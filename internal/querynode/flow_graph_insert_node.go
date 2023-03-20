// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package querynode

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"sync"

	"github.com/cockroachdb/errors"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// insertNode is one of the nodes in query flow graph
type insertNode struct {
	baseNode
	collectionID UniqueID
	metaReplica  ReplicaInterface // streaming
	vchannel     Channel
}

// insertData stores the valid insert data
type insertData struct {
	insertIDs        map[UniqueID][]int64 // rowIDs
	insertTimestamps map[UniqueID][]Timestamp
	insertRecords    map[UniqueID][]*schemapb.FieldData
	insertOffset     map[UniqueID]int64
	insertPKs        map[UniqueID][]primaryKey // pks
}

// deleteData stores the valid delete data
type deleteData struct {
	deleteIDs        map[UniqueID][]primaryKey // pks
	deleteTimestamps map[UniqueID][]Timestamp
	deleteOffset     map[UniqueID]int64
}

// Name returns the name of insertNode
func (iNode *insertNode) Name() string {
	return fmt.Sprintf("iNode-%s", iNode.vchannel)
}

func (iNode *insertNode) IsValidInMsg(in []Msg) bool {
	if !iNode.baseNode.IsValidInMsg(in) {
		return false
	}
	_, ok := in[0].(*insertMsg)
	if !ok {
		log.Warn("type assertion failed for insertMsg", zap.String("msgType", reflect.TypeOf(in[0]).Name()), zap.String("name", iNode.Name()))
		return false
	}
	return true
}

// Operate handles input messages, to execute insert operations
func (iNode *insertNode) Operate(in []Msg) []Msg {
	iMsg := in[0].(*insertMsg)

	var spans []trace.Span
	for _, msg := range iMsg.insertMessages {
		ctx := msg.TraceCtx()
		sp := trace.SpanFromContext(msg.TraceCtx())
		spans = append(spans, sp)
		msg.SetTraceCtx(ctx)
	}
	defer func() {
		for _, sp := range spans {
			sp.End()
		}
	}()

	if iMsg.IsCloseMsg() {
		return []Msg{
			&serviceTimeMsg{BaseMsg: flowgraph.NewBaseMsg(true)},
		}
	}

	iData := insertData{
		insertIDs:        make(map[UniqueID][]int64),
		insertTimestamps: make(map[UniqueID][]Timestamp),
		insertRecords:    make(map[UniqueID][]*schemapb.FieldData),
		insertOffset:     make(map[UniqueID]int64),
		insertPKs:        make(map[UniqueID][]primaryKey),
	}

	collection, err := iNode.metaReplica.getCollectionByID(iNode.collectionID)
	if err != nil {
		// QueryNode should add collection before start flow graph
		panic(fmt.Errorf("%s getCollectionByID failed, collectionID = %d, vchannel: %s", iNode.Name(), iNode.collectionID, iNode.vchannel))
	}

	// 1. hash insertMessages to insertData
	// sort timestamps ensures that the data in iData.insertRecords is sorted in ascending order of timestamp
	// avoiding re-sorting in segCore, which will need data copying
	sort.Slice(iMsg.insertMessages, func(i, j int) bool {
		return iMsg.insertMessages[i].BeginTs() < iMsg.insertMessages[j].BeginTs()
	})
	for _, insertMsg := range iMsg.insertMessages {
		// if loadType is loadCollection, check if partition exists, if not, create partition
		if collection.getLoadType() == loadTypeCollection {
			err = iNode.metaReplica.addPartition(insertMsg.CollectionID, insertMsg.PartitionID)
			if err != nil {
				// error occurs only when collection cannot be found, should not happen
				err = fmt.Errorf("insertNode addPartition failed, err = %s", err)
				log.Error(err.Error(), zap.Int64("collection", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
				panic(err)
			}
		}

		// check if segment exists, if not, create this segment
		has, err := iNode.metaReplica.hasSegment(insertMsg.SegmentID, segmentTypeGrowing)
		if err != nil {
			log.Error(err.Error()) // never gonna happen
			panic(err)
		}
		if !has {
			log.Info("Add growing segment",
				zap.Int64("collectionID", insertMsg.CollectionID),
				zap.Int64("segmentID", insertMsg.SegmentID),
				zap.Uint64("startPosition", insertMsg.BeginTs()),
			)
			startPosition := &msgpb.MsgPosition{
				ChannelName: insertMsg.ShardName,
				Timestamp:   insertMsg.BeginTs(),
			}
			err = iNode.metaReplica.addSegment(insertMsg.SegmentID, insertMsg.PartitionID, insertMsg.CollectionID, insertMsg.ShardName, 0, startPosition, segmentTypeGrowing)
			if err != nil {
				// error occurs when collection or partition cannot be found, collection and partition should be created before
				err = fmt.Errorf("insertNode addSegment failed, err = %s", err)
				log.Error(err.Error(), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
				panic(err)
			}
		}

		insertRecord, err := storage.TransferInsertMsgToInsertRecord(collection.schema, insertMsg)
		if err != nil {
			// occurs only when schema doesn't have dim param, this should not happen
			err = fmt.Errorf("failed to transfer msgStream.insertMsg to storage.InsertRecord, err = %s", err)
			log.Error(err.Error(), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
			panic(err)
		}

		iData.insertIDs[insertMsg.SegmentID] = append(iData.insertIDs[insertMsg.SegmentID], insertMsg.RowIDs...)
		iData.insertTimestamps[insertMsg.SegmentID] = append(iData.insertTimestamps[insertMsg.SegmentID], insertMsg.Timestamps...)
		if _, ok := iData.insertRecords[insertMsg.SegmentID]; !ok {
			iData.insertRecords[insertMsg.SegmentID] = insertRecord.FieldsData
		} else {
			typeutil.MergeFieldData(iData.insertRecords[insertMsg.SegmentID], insertRecord.FieldsData)
		}
		pks, err := getPrimaryKeys(insertMsg, iNode.metaReplica)
		if err != nil {
			// error occurs when cannot find collection or data is misaligned, should not happen
			err = fmt.Errorf("failed to get primary keys, err = %d", err)
			log.Error(err.Error(), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
			panic(err)
		}
		iData.insertPKs[insertMsg.SegmentID] = append(iData.insertPKs[insertMsg.SegmentID], pks...)
	}

	// 2. do preInsert
	for segmentID := range iData.insertRecords {
		log := log.With(
			zap.Int64("segmentID", segmentID))
		var targetSegment, err = iNode.metaReplica.getSegmentByID(segmentID, segmentTypeGrowing)
		if err != nil {
			// should not happen, segment should be created before
			err = fmt.Errorf("insertNode getSegmentByID failed, err = %s", err)
			log.Error(err.Error(), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))

			if !errors.Is(err, ErrSegmentNotFound) {
				panic(err)
			}
		}

		var numOfRecords = len(iData.insertIDs[segmentID])
		if targetSegment != nil {
			offset, err := targetSegment.segmentPreInsert(numOfRecords)
			if err != nil {
				if errors.Is(err, ErrSegmentUnhealthy) {
					log.Warn("segment removed before preInsert")
					continue
				}
				// error occurs when cgo function `PreInsert` failed
				err = fmt.Errorf("segmentPreInsert failed, segmentID = %d, err = %s", segmentID, err)
				log.Error(err.Error(), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
				panic(err)
			}
			iData.insertOffset[segmentID] = offset
			log.Debug("insertNode operator", zap.Int("insert size", numOfRecords), zap.Int64("insert offset", offset), zap.Int64("segmentID", segmentID), zap.Int64("collectionID", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
			targetSegment.updateBloomFilter(iData.insertPKs[segmentID])
		}
	}

	// 3. do insert
	wg := sync.WaitGroup{}
	for segmentID := range iData.insertRecords {
		segmentID := segmentID
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := iNode.insert(&iData, segmentID)
			if err != nil {
				// error occurs when segment cannot be found or cgo function `Insert` failed
				err = fmt.Errorf("segment insert failed, segmentID = %d, err = %s", segmentID, err)
				log.Error(err.Error(), zap.Int64("collection", iNode.collectionID), zap.String("vchannel", iNode.vchannel))
				panic(err)
			}
		}()
	}
	wg.Wait()

	delData := &deleteData{
		deleteIDs:        make(map[UniqueID][]primaryKey),
		deleteTimestamps: make(map[UniqueID][]Timestamp),
		deleteOffset:     make(map[UniqueID]int64),
	}
	// 1. filter segment by bloom filter
	for _, delMsg := range iMsg.deleteMessages {
		if iNode.metaReplica.getSegmentNum(segmentTypeGrowing) != 0 {
			log.Debug("delete in streaming replica",
				zap.String("vchannel", iNode.vchannel),
				zap.Int64("collectionID", delMsg.CollectionID),
				zap.String("collectionName", delMsg.CollectionName),
				zap.Int64("numPKs", delMsg.NumRows))
			err := processDeleteMessages(iNode.metaReplica, segmentTypeGrowing, delMsg, delData, iNode.vchannel)
			if err != nil {
				// error occurs when missing meta info or unexpected pk type, should not happen
				err = fmt.Errorf("insertNode processDeleteMessages failed, collectionID = %d, err = %s, vchannel: %s", delMsg.CollectionID, err, iNode.vchannel)
				log.Error(err.Error())
				panic(err)
			}
		}
	}

	// 2. do preDelete
	for segmentID, pks := range delData.deleteIDs {
		segment, err := iNode.metaReplica.getSegmentByID(segmentID, segmentTypeGrowing)
		if err != nil {
			if errors.Is(err, ErrSegmentNotFound) {
				log.Warn("segment not found when do preDelete, it may have been released due to compaction",
					zap.Int64("segmentID", segmentID),
					zap.Error(err),
				)
				continue
			}

			err = fmt.Errorf("insertNode getSegmentByID failed, err = %s", err)
			log.Error(err.Error())
			panic(err)
		}
		offset := segment.segmentPreDelete(len(pks))
		if offset >= 0 {
			delData.deleteOffset[segmentID] = offset
		}
	}

	// 3. do delete
	for segmentID := range delData.deleteOffset {
		segmentID := segmentID
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := iNode.delete(delData, segmentID)
			if err != nil {
				// error occurs when segment cannot be found, calling cgo function delete failed and etc...
				err = fmt.Errorf("segment delete failed, segmentID = %d, err = %s", segmentID, err)
				log.Error(err.Error())
				panic(err)
			}
		}()
	}
	wg.Wait()

	var res Msg = &serviceTimeMsg{
		timeRange: iMsg.timeRange,
	}

	return []Msg{res}
}

// processDeleteMessages would execute delete operations for growing segments
func processDeleteMessages(replica ReplicaInterface, segType segmentType, msg *msgstream.DeleteMsg, delData *deleteData, vchannelName string) error {
	var partitionIDs []UniqueID
	var err error
	if msg.PartitionID != -1 {
		partitionIDs = []UniqueID{msg.GetPartitionID()}
	}
	var resultSegmentIDs []UniqueID
	resultSegmentIDs, err = replica.getSegmentIDsByVChannel(partitionIDs, vchannelName, segType)
	log.Warn("processDeleteMessage", zap.String("vchannel", vchannelName), zap.Int64s("segmentIDs", resultSegmentIDs), zap.Int64s("paritions", partitionIDs))
	if err != nil {
		if !errors.Is(err, ErrPartitionNotFound) {
			return err
		}
	}

	primaryKeys := storage.ParseIDs2PrimaryKeys(msg.PrimaryKeys)
	for _, segmentID := range resultSegmentIDs {
		segment, err := replica.getSegmentByID(segmentID, segType)
		if err != nil {
			if errors.Is(err, ErrSegmentNotFound) {
				log.Warn("segment not found when process delete messages, it may have been released due to compaction",
					zap.Int64("segmentID", segmentID),
					zap.Error(err),
				)
				continue
			}
			return err
		}
		pks, tss, err := filterSegmentsByPKs(primaryKeys, msg.Timestamps, segment)
		if err != nil {
			return err
		}
		if len(pks) > 0 {
			delData.deleteIDs[segmentID] = append(delData.deleteIDs[segmentID], pks...)
			delData.deleteTimestamps[segmentID] = append(delData.deleteTimestamps[segmentID], tss...)
		}
	}
	return nil
}

// filterSegmentsByPKs would filter segments by primary keys
func filterSegmentsByPKs(pks []primaryKey, timestamps []Timestamp, segment *Segment) ([]primaryKey, []Timestamp, error) {
	if segment == nil {
		return nil, nil, fmt.Errorf("segments is nil when getSegmentsByPKs")
	}

	retPks := make([]primaryKey, 0)
	retTss := make([]Timestamp, 0)
	for index, pk := range pks {
		if segment.isPKExist(pk) {
			retPks = append(retPks, pk)
			retTss = append(retTss, timestamps[index])
		}
	}
	return retPks, retTss, nil
}

// insert would execute insert operations for specific growing segment
func (iNode *insertNode) insert(iData *insertData, segmentID UniqueID) error {
	log := log.With(
		zap.Int64("collectionID", iNode.collectionID),
		zap.Int64("segmentID", segmentID))
	var targetSegment, err = iNode.metaReplica.getSegmentByID(segmentID, segmentTypeGrowing)
	if err != nil {
		if errors.Is(err, ErrSegmentNotFound) {
			log.Warn("the segment has been released, ignore it",
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			return nil
		}
		return fmt.Errorf("getSegmentByID failed, err = %s", err)
	}

	ids := iData.insertIDs[segmentID]
	timestamps := iData.insertTimestamps[segmentID]
	offsets := iData.insertOffset[segmentID]
	insertRecord := &segcorepb.InsertRecord{
		FieldsData: iData.insertRecords[segmentID],
		NumRows:    int64(len(ids)),
	}

	err = targetSegment.segmentInsert(offsets, ids, timestamps, insertRecord)
	if err != nil {
		if errors.Is(err, ErrSegmentUnhealthy) {
			log.Warn("segment removed before insert")
			return nil
		}
		return fmt.Errorf("segmentInsert failed, segmentID = %d, err = %s", segmentID, err)
	}

	log.Debug("Do insert done", zap.Int("len", len(iData.insertIDs[segmentID])))
	return nil
}

// delete would execute delete operations for specific growing segment
func (iNode *insertNode) delete(deleteData *deleteData, segmentID UniqueID) error {
	log := log.With(
		zap.Int64("collectionID", iNode.collectionID),
		zap.Int64("segmentID", segmentID))
	targetSegment, err := iNode.metaReplica.getSegmentByID(segmentID, segmentTypeGrowing)
	if err != nil {
		if errors.Is(err, ErrSegmentNotFound) {
			log.Warn("segment not found when applying delete message, it may have been released due to compaction",
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			return nil
		}
		return fmt.Errorf("getSegmentByID failed, err = %s", err)
	}

	if targetSegment.getType() != segmentTypeGrowing {
		return fmt.Errorf("unexpected segmentType when delete, segmentType = %s", targetSegment.segmentType.String())
	}

	ids := deleteData.deleteIDs[segmentID]
	timestamps := deleteData.deleteTimestamps[segmentID]
	offset := deleteData.deleteOffset[segmentID]

	err = targetSegment.segmentDelete(offset, ids, timestamps)
	if err != nil {
		if errors.Is(err, ErrSegmentUnhealthy) {
			log.Warn("segment removed before delete")
			return nil
		}
		return fmt.Errorf("segmentDelete failed, err = %s", err)
	}

	log.Debug("Do delete done", zap.Int("len", len(deleteData.deleteIDs[segmentID])))
	return nil
}

// TODO: remove this function to proper file
// getPrimaryKeys would get primary keys by insert messages
func getPrimaryKeys(msg *msgstream.InsertMsg, metaReplica ReplicaInterface) ([]primaryKey, error) {
	if err := msg.CheckAligned(); err != nil {
		log.Warn("misaligned messages detected", zap.Error(err))
		return nil, err
	}
	collectionID := msg.GetCollectionID()

	collection, err := metaReplica.getCollectionByID(collectionID)
	if err != nil {
		log.Warn(err.Error())
		return nil, err
	}

	return getPKs(msg, collection.schema)
}

func getPKs(msg *msgstream.InsertMsg, schema *schemapb.CollectionSchema) ([]primaryKey, error) {
	if msg.IsRowBased() {
		return getPKsFromRowBasedInsertMsg(msg, schema)
	}
	return getPKsFromColumnBasedInsertMsg(msg, schema)
}

func getPKsFromRowBasedInsertMsg(msg *msgstream.InsertMsg, schema *schemapb.CollectionSchema) ([]primaryKey, error) {
	offset := 0
	for _, field := range schema.Fields {
		if field.IsPrimaryKey {
			break
		}
		switch field.DataType {
		case schemapb.DataType_Bool:
			offset++
		case schemapb.DataType_Int8:
			offset++
		case schemapb.DataType_Int16:
			offset += 2
		case schemapb.DataType_Int32:
			offset += 4
		case schemapb.DataType_Int64:
			offset += 8
		case schemapb.DataType_Float:
			offset += 4
		case schemapb.DataType_Double:
			offset += 8
		case schemapb.DataType_FloatVector:
			for _, t := range field.TypeParams {
				if t.Key == "dim" {
					dim, err := strconv.Atoi(t.Value)
					if err != nil {
						return nil, fmt.Errorf("strconv wrong on get dim, err = %s", err)
					}
					offset += dim * 4
					break
				}
			}
		case schemapb.DataType_BinaryVector:
			for _, t := range field.TypeParams {
				if t.Key == "dim" {
					dim, err := strconv.Atoi(t.Value)
					if err != nil {
						return nil, fmt.Errorf("strconv wrong on get dim, err = %s", err)
					}
					offset += dim / 8
					break
				}
			}
		}
	}

	blobReaders := make([]io.Reader, len(msg.RowData))
	for i, blob := range msg.RowData {
		blobReaders[i] = bytes.NewReader(blob.GetValue()[offset : offset+8])
	}
	pks := make([]primaryKey, len(blobReaders))

	for i, reader := range blobReaders {
		var int64PkValue int64
		err := binary.Read(reader, common.Endian, &int64PkValue)
		if err != nil {
			log.Warn("binary read blob value failed", zap.Error(err))
			return nil, err
		}
		pks[i] = newInt64PrimaryKey(int64PkValue)
	}

	return pks, nil
}

func getPKsFromColumnBasedInsertMsg(msg *msgstream.InsertMsg, schema *schemapb.CollectionSchema) ([]primaryKey, error) {
	primaryFieldSchema, err := typeutil.GetPrimaryFieldSchema(schema)
	if err != nil {
		return nil, err
	}

	primaryFieldData, err := typeutil.GetPrimaryFieldData(msg.GetFieldsData(), primaryFieldSchema)
	if err != nil {
		return nil, err
	}

	pks, err := storage.ParseFieldData2PrimaryKeys(primaryFieldData)
	if err != nil {
		return nil, err
	}

	return pks, nil
}

// newInsertNode returns a new insertNode
func newInsertNode(metaReplica ReplicaInterface, collectionID UniqueID, vchannel Channel) *insertNode {
	maxQueueLength := Params.QueryNodeCfg.FlowGraphMaxQueueLength.GetAsInt32()
	maxParallelism := Params.QueryNodeCfg.FlowGraphMaxParallelism.GetAsInt32()

	baseNode := baseNode{}
	baseNode.SetMaxQueueLength(maxQueueLength)
	baseNode.SetMaxParallelism(maxParallelism)

	return &insertNode{
		baseNode:     baseNode,
		collectionID: collectionID,
		metaReplica:  metaReplica,
		vchannel:     vchannel,
	}
}
