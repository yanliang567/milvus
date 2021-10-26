// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package querynode

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strconv"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/kv"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	minioKV "github.com/milvus-io/milvus/internal/kv/minio"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/funcutil"
)

const (
	queryNodeSegmentMetaPrefix = "queryNode-segmentMeta"
)

// segmentLoader is only responsible for loading the field data from binlog
type segmentLoader struct {
	historicalReplica ReplicaInterface

	dataCoord types.DataCoord

	minioKV kv.DataKV // minio minioKV
	etcdKV  *etcdkv.EtcdKV

	indexLoader *indexLoader
}

func (loader *segmentLoader) loadSegment(req *querypb.LoadSegmentsRequest) error {
	// no segment needs to load, return
	if len(req.Infos) == 0 {
		return nil
	}

	newSegments := make(map[UniqueID]*Segment)
	segmentGC := func() {
		for _, s := range newSegments {
			deleteSegment(s)
		}
	}

	segmentFieldBinLogs := make(map[UniqueID][]*datapb.FieldBinlog)
	segmentIndexedFieldIDs := make(map[UniqueID][]FieldID)
	segmentSizes := make(map[UniqueID]int64)

	// prepare and estimate segments size
	for _, info := range req.Infos {
		segmentID := info.SegmentID
		partitionID := info.PartitionID
		collectionID := info.CollectionID

		collection, err := loader.historicalReplica.getCollectionByID(collectionID)
		if err != nil {
			segmentGC()
			return err
		}
		segment := newSegment(collection, segmentID, partitionID, collectionID, "", segmentTypeSealed, true)
		newSegments[segmentID] = segment
		fieldBinlog, indexedFieldID, err := loader.getFieldAndIndexInfo(segment, info)
		if err != nil {
			segmentGC()
			return err
		}
		segmentSize, err := loader.estimateSegmentSize(segment, fieldBinlog, indexedFieldID)
		if err != nil {
			segmentGC()
			return err
		}
		segmentFieldBinLogs[segmentID] = fieldBinlog
		segmentIndexedFieldIDs[segmentID] = indexedFieldID
		segmentSizes[segmentID] = segmentSize
	}

	// check memory limit
	err := loader.checkSegmentSize(req.Infos[0].CollectionID, segmentSizes)
	if err != nil {
		segmentGC()
		return err
	}

	// start to load
	for _, info := range req.Infos {
		segmentID := info.SegmentID
		if newSegments[segmentID] == nil || segmentFieldBinLogs[segmentID] == nil || segmentIndexedFieldIDs[segmentID] == nil {
			segmentGC()
			return errors.New(fmt.Sprintln("unexpected error, cannot find load infos, this error should not happen, collectionID = ", req.Infos[0].CollectionID))
		}
		err = loader.loadSegmentInternal(newSegments[segmentID],
			segmentFieldBinLogs[segmentID],
			segmentIndexedFieldIDs[segmentID],
			info)
		if err != nil {
			segmentGC()
			return err
		}
	}

	// set segments
	for _, s := range newSegments {
		err := loader.historicalReplica.setSegment(s)
		if err != nil {
			segmentGC()
			return err
		}
	}
	return nil
}

func (loader *segmentLoader) loadSegmentInternal(segment *Segment,
	fieldBinLogs []*datapb.FieldBinlog,
	indexFieldIDs []FieldID,
	segmentLoadInfo *querypb.SegmentLoadInfo) error {
	log.Debug("loading insert...")
	err := loader.loadSegmentFieldsData(segment, fieldBinLogs)
	if err != nil {
		return err
	}

	pkIDField, err := loader.historicalReplica.getPKFieldIDByCollectionID(segment.collectionID)
	if err != nil {
		return err
	}
	if pkIDField == common.InvalidFieldID {
		log.Warn("segment primary key field doesn't exist when load segment")
	} else {
		log.Debug("loading bloom filter...")
		pkStatsBinlogs := loader.filterPKStatsBinlogs(segmentLoadInfo.Statslogs, pkIDField)
		err = loader.loadSegmentBloomFilter(segment, pkStatsBinlogs)
		if err != nil {
			return err
		}
	}

	log.Debug("loading delta...")
	err = loader.loadDeltaLogs(segment, segmentLoadInfo.Deltalogs)
	if err != nil {
		return err
	}

	for _, id := range indexFieldIDs {
		log.Debug("loading index...")
		err = loader.indexLoader.loadIndex(segment, id)
		if err != nil {
			return err
		}
	}

	return nil
}

func (loader *segmentLoader) filterPKStatsBinlogs(fieldBinlogs []*datapb.FieldBinlog, pkFieldID int64) []string {
	result := make([]string, 0)
	for _, fieldBinlog := range fieldBinlogs {
		if fieldBinlog.FieldID == pkFieldID {
			result = append(result, fieldBinlog.Binlogs...)
		}
	}
	return result
}

func (loader *segmentLoader) filterFieldBinlogs(fieldBinlogs []*datapb.FieldBinlog, skipFieldIDs []int64) []*datapb.FieldBinlog {
	result := make([]*datapb.FieldBinlog, 0)
	for _, fieldBinlog := range fieldBinlogs {
		if !funcutil.SliceContain(skipFieldIDs, fieldBinlog.FieldID) {
			result = append(result, fieldBinlog)
		}
	}
	return result
}

func (loader *segmentLoader) loadSegmentFieldsData(segment *Segment, fieldBinlogs []*datapb.FieldBinlog) error {
	iCodec := storage.InsertCodec{}
	defer func() {
		err := iCodec.Close()
		if err != nil {
			log.Warn(err.Error())
		}
	}()
	blobs := make([]*storage.Blob, 0)
	for _, fb := range fieldBinlogs {
		log.Debug("load segment fields data",
			zap.Int64("segmentID", segment.segmentID),
			zap.Any("fieldID", fb.FieldID),
			zap.String("paths", fmt.Sprintln(fb.Binlogs)),
		)
		for _, path := range fb.Binlogs {
			p := path
			binLog, err := loader.minioKV.Load(path)
			if err != nil {
				// TODO: return or continue?
				return err
			}
			blob := &storage.Blob{
				Key:   p,
				Value: []byte(binLog),
			}
			blobs = append(blobs, blob)
		}
	}

	_, _, insertData, err := iCodec.Deserialize(blobs)
	if err != nil {
		log.Warn(err.Error())
		return err
	}

	for fieldID, value := range insertData.Data {
		var numRows []int64
		var data interface{}
		switch fieldData := value.(type) {
		case *storage.BoolFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.Int8FieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.Int16FieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.Int32FieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.Int64FieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.FloatFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.DoubleFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.StringFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.FloatVectorFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		case *storage.BinaryVectorFieldData:
			numRows = fieldData.NumRows
			data = fieldData.Data
		default:
			return errors.New("unexpected field data type")
		}
		if fieldID == common.TimeStampField {
			segment.setIDBinlogRowSizes(numRows)
		}
		totalNumRows := int64(0)
		for _, numRow := range numRows {
			totalNumRows += numRow
		}
		err = segment.segmentLoadFieldData(fieldID, int(totalNumRows), data)
		if err != nil {
			// TODO: return or continue?
			return err
		}
	}

	return nil
}

func (loader *segmentLoader) loadSegmentBloomFilter(segment *Segment, binlogPaths []string) error {
	if len(binlogPaths) == 0 {
		log.Info("there are no stats logs saved with segment", zap.Any("segmentID", segment.segmentID))
		return nil
	}

	values, err := loader.minioKV.MultiLoad(binlogPaths)
	if err != nil {
		return err
	}
	blobs := make([]*storage.Blob, 0)
	for i := 0; i < len(values); i++ {
		blobs = append(blobs, &storage.Blob{Value: []byte(values[i])})
	}

	stats, err := storage.DeserializeStats(blobs)
	if err != nil {
		return err
	}
	for _, stat := range stats {
		if stat.BF == nil {
			log.Warn("stat log with nil bloom filter", zap.Int64("segmentID", segment.segmentID), zap.Any("stat", stat))
			continue
		}
		err = segment.pkFilter.Merge(stat.BF)
		if err != nil {
			return err
		}
	}
	return nil
}

func (loader *segmentLoader) loadDeltaLogs(segment *Segment, deltaLogs []*datapb.DeltaLogInfo) error {
	if len(deltaLogs) == 0 {
		log.Info("there are no delta logs saved with segment", zap.Any("segmentID", segment.segmentID))
		return nil
	}
	dCodec := storage.DeleteCodec{}
	blobs := make([]*storage.Blob, 0)
	for _, deltaLog := range deltaLogs {
		value, err := loader.minioKV.Load(deltaLog.DeltaLogPath)
		if err != nil {
			return err
		}
		blob := &storage.Blob{
			Key:   deltaLog.DeltaLogPath,
			Value: []byte(value),
		}
		blobs = append(blobs, blob)
	}
	_, _, deltaData, err := dCodec.Deserialize(blobs)
	if err != nil {
		return err
	}

	rowCount := len(deltaData.Data)
	pks := make([]int64, 0)
	tss := make([]Timestamp, 0)
	for pk, ts := range deltaData.Data {
		pks = append(pks, pk)
		tss = append(tss, Timestamp(ts))
	}
	err = segment.segmentLoadDeletedRecord(pks, tss, int64(rowCount))
	if err != nil {
		return err
	}
	return nil
}

// JoinIDPath joins ids to path format.
func JoinIDPath(ids ...UniqueID) string {
	idStr := make([]string, len(ids))
	for _, id := range ids {
		idStr = append(idStr, strconv.FormatInt(id, 10))
	}
	return path.Join(idStr...)
}

func (loader *segmentLoader) getFieldAndIndexInfo(segment *Segment,
	segmentLoadInfo *querypb.SegmentLoadInfo) ([]*datapb.FieldBinlog, []FieldID, error) {
	collectionID := segment.collectionID
	vectorFieldIDs, err := loader.historicalReplica.getVecFieldIDsByCollectionID(collectionID)
	if err != nil {
		return nil, nil, err
	}
	if len(vectorFieldIDs) <= 0 {
		return nil, nil, fmt.Errorf("no vector field in collection %d", collectionID)
	}

	// add VectorFieldInfo for vector fields
	for _, fieldBinlog := range segmentLoadInfo.BinlogPaths {
		if funcutil.SliceContain(vectorFieldIDs, fieldBinlog.FieldID) {
			vectorFieldInfo := newVectorFieldInfo(fieldBinlog)
			segment.setVectorFieldInfo(fieldBinlog.FieldID, vectorFieldInfo)
		}
	}

	indexedFieldIDs := make([]FieldID, 0)
	for _, vecFieldID := range vectorFieldIDs {
		err = loader.indexLoader.setIndexInfo(collectionID, segment, vecFieldID)
		if err != nil {
			log.Warn(err.Error())
			continue
		}
		indexedFieldIDs = append(indexedFieldIDs, vecFieldID)
	}

	// we don't need to load raw data for indexed vector field
	fieldBinlogs := loader.filterFieldBinlogs(segmentLoadInfo.BinlogPaths, indexedFieldIDs)
	return fieldBinlogs, indexedFieldIDs, nil
}

func (loader *segmentLoader) estimateSegmentSize(segment *Segment,
	fieldBinLogs []*datapb.FieldBinlog,
	indexFieldIDs []FieldID) (int64, error) {
	segmentSize := int64(0)
	// get fields data size, if len(indexFieldIDs) == 0, vector field would be involved in fieldBinLogs
	for _, fb := range fieldBinLogs {
		log.Debug("estimate segment fields size",
			zap.Any("collectionID", segment.collectionID),
			zap.Any("segmentID", segment.ID()),
			zap.Any("fieldID", fb.FieldID),
			zap.Any("paths", fb.Binlogs),
		)
		for _, binlogPath := range fb.Binlogs {
			logSize, err := storage.EstimateMemorySize(loader.minioKV, binlogPath)
			if err != nil {
				logSize, err = storage.GetBinlogSize(loader.minioKV, binlogPath)
				if err != nil {
					return 0, err
				}
			}
			segmentSize += logSize
		}
	}
	// get index size
	for _, fieldID := range indexFieldIDs {
		indexSize, err := loader.indexLoader.estimateIndexBinlogSize(segment, fieldID)
		if err != nil {
			return 0, err
		}
		segmentSize += indexSize
	}
	return segmentSize, nil
}

func (loader *segmentLoader) checkSegmentSize(collectionID UniqueID, segmentSizes map[UniqueID]int64) error {
	const thresholdFactor = 0.9
	usedMem, err := getUsedMemory()
	if err != nil {
		return err
	}
	totalMem, err := getTotalMemory()
	if err != nil {
		return err
	}

	segmentTotalSize := int64(0)
	for _, size := range segmentSizes {
		segmentTotalSize += size
	}

	for segmentID, size := range segmentSizes {
		log.Debug("memory stats when load segment",
			zap.Any("collectionIDs", collectionID),
			zap.Any("segmentID", segmentID),
			zap.Any("totalMem", totalMem),
			zap.Any("usedMem", usedMem),
			zap.Any("segmentTotalSize", segmentTotalSize),
			zap.Any("currentSegmentSize", size),
			zap.Any("thresholdFactor", thresholdFactor),
		)
		if int64(usedMem)+segmentTotalSize+size > int64(float64(totalMem)*thresholdFactor) {
			return errors.New(fmt.Sprintln("load segment failed, OOM if load, "+
				"collectionID = ", collectionID, ", ",
				"usedMem = ", usedMem, ", ",
				"segmentTotalSize = ", segmentTotalSize, ", ",
				"currentSegmentSize = ", size, ", ",
				"totalMem = ", totalMem, ", ",
				"thresholdFactor = ", thresholdFactor,
			))
		}
	}

	return nil
}

func newSegmentLoader(ctx context.Context, rootCoord types.RootCoord, indexCoord types.IndexCoord, replica ReplicaInterface, etcdKV *etcdkv.EtcdKV) *segmentLoader {
	option := &minioKV.Option{
		Address:           Params.MinioEndPoint,
		AccessKeyID:       Params.MinioAccessKeyID,
		SecretAccessKeyID: Params.MinioSecretAccessKey,
		UseSSL:            Params.MinioUseSSLStr,
		CreateBucket:      true,
		BucketName:        Params.MinioBucketName,
	}

	client, err := minioKV.NewMinIOKV(ctx, option)
	if err != nil {
		panic(err)
	}

	iLoader := newIndexLoader(ctx, rootCoord, indexCoord, replica)
	return &segmentLoader{
		historicalReplica: replica,

		minioKV: client,
		etcdKV:  etcdKV,

		indexLoader: iLoader,
	}
}
