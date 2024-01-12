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

package delegator

import (
	"context"
	"fmt"
	"math/rand"
	"sort"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/querynodev2/cluster"
	"github.com/milvus-io/milvus/internal/querynodev2/delegator/deletebuffer"
	"github.com/milvus-io/milvus/internal/querynodev2/pkoracle"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/mq/msgstream"
	"github.com/milvus-io/milvus/pkg/mq/msgstream/mqwrapper"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/retry"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

// delegator data related part

// InsertData
type InsertData struct {
	RowIDs        []int64
	PrimaryKeys   []storage.PrimaryKey
	Timestamps    []uint64
	InsertRecord  *segcorepb.InsertRecord
	StartPosition *msgpb.MsgPosition
	PartitionID   int64
}

type DeleteData struct {
	PartitionID int64
	PrimaryKeys []storage.PrimaryKey
	Timestamps  []uint64
	RowCount    int64
}

// Append appends another delete data into this one.
func (d *DeleteData) Append(ad DeleteData) {
	d.PrimaryKeys = append(d.PrimaryKeys, ad.PrimaryKeys...)
	d.Timestamps = append(d.Timestamps, ad.Timestamps...)
	d.RowCount += ad.RowCount
}

// ProcessInsert handles insert data in delegator.
func (sd *shardDelegator) ProcessInsert(insertRecords map[int64]*InsertData) {
	method := "ProcessInsert"
	tr := timerecord.NewTimeRecorder(method)
	log := sd.getLogger(context.Background())
	for segmentID, insertData := range insertRecords {
		growing := sd.segmentManager.GetGrowing(segmentID)
		if growing == nil {
			var err error
			growing, err = segments.NewSegment(
				context.Background(),
				sd.collection,
				segmentID,
				insertData.PartitionID,
				sd.collectionID,
				sd.vchannelName,
				segments.SegmentTypeGrowing,
				0,
				insertData.StartPosition,
				insertData.StartPosition,
				datapb.SegmentLevel_L1,
			)
			if err != nil {
				log.Error("failed to create new segment",
					zap.Int64("segmentID", segmentID),
					zap.Error(err))
				panic(err)
			}
		}

		err := growing.Insert(context.Background(), insertData.RowIDs, insertData.Timestamps, insertData.InsertRecord)
		if err != nil {
			log.Error("failed to insert data into growing segment",
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			if errors.IsAny(err, merr.ErrSegmentNotLoaded, merr.ErrSegmentNotFound) {
				log.Warn("try to insert data into released segment, skip it", zap.Error(err))
				continue
			}
			// panic here, insert failure
			panic(err)
		}
		metrics.QueryNodeNumEntities.WithLabelValues(
			fmt.Sprint(paramtable.GetNodeID()),
			fmt.Sprint(growing.Collection()),
			fmt.Sprint(growing.Partition()),
			growing.Type().String(),
			fmt.Sprint(0),
		).Add(float64(len(insertData.RowIDs)))
		growing.UpdateBloomFilter(insertData.PrimaryKeys)

		if !sd.pkOracle.Exists(growing, paramtable.GetNodeID()) {
			// register created growing segment after insert, avoid to add empty growing to delegator
			sd.pkOracle.Register(growing, paramtable.GetNodeID())
			sd.segmentManager.Put(segments.SegmentTypeGrowing, growing)
			sd.addGrowing(SegmentEntry{
				NodeID:        paramtable.GetNodeID(),
				SegmentID:     segmentID,
				PartitionID:   insertData.PartitionID,
				Version:       0,
				TargetVersion: initialTargetVersion,
			})
		}

		log.Debug("insert into growing segment",
			zap.Int64("collectionID", growing.Collection()),
			zap.Int64("segmentID", segmentID),
			zap.Int("rowCount", len(insertData.RowIDs)),
			zap.Uint64("maxTimestamp", insertData.Timestamps[len(insertData.Timestamps)-1]),
		)
	}
	metrics.QueryNodeProcessCost.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), metrics.InsertLabel).
		Observe(float64(tr.ElapseSpan().Milliseconds()))
}

// ProcessDelete handles delete data in delegator.
// delegator puts deleteData into buffer first,
// then dispatch data to segments acoording to the result of pkOracle.
func (sd *shardDelegator) ProcessDelete(deleteData []*DeleteData, ts uint64) {
	method := "ProcessDelete"
	tr := timerecord.NewTimeRecorder(method)
	// block load segment handle delete buffer
	sd.deleteMut.Lock()
	defer sd.deleteMut.Unlock()

	log := sd.getLogger(context.Background())

	log.Debug("start to process delete", zap.Uint64("ts", ts))
	// add deleteData into buffer.
	cacheItems := make([]deletebuffer.BufferItem, 0, len(deleteData))
	for _, entry := range deleteData {
		cacheItems = append(cacheItems, deletebuffer.BufferItem{
			PartitionID: entry.PartitionID,
			DeleteData: storage.DeleteData{
				Pks:      entry.PrimaryKeys,
				Tss:      entry.Timestamps,
				RowCount: entry.RowCount,
			},
		})
	}

	sd.deleteBuffer.Put(&deletebuffer.Item{
		Ts:   ts,
		Data: cacheItems,
	})

	// segment => delete data
	delRecords := make(map[int64]DeleteData)
	for _, data := range deleteData {
		for i, pk := range data.PrimaryKeys {
			segmentIDs, err := sd.pkOracle.Get(pk, pkoracle.WithPartitionID(data.PartitionID))
			if err != nil {
				log.Warn("failed to get delete candidates for pk", zap.Any("pk", pk.GetValue()))
				continue
			}
			for _, segmentID := range segmentIDs {
				delRecord := delRecords[segmentID]
				delRecord.PrimaryKeys = append(delRecord.PrimaryKeys, pk)
				delRecord.Timestamps = append(delRecord.Timestamps, data.Timestamps[i])
				delRecord.RowCount++
				delRecords[segmentID] = delRecord
			}
		}
	}

	offlineSegments := typeutil.NewConcurrentSet[int64]()

	sealed, growing, version := sd.distribution.PinOnlineSegments()

	eg, ctx := errgroup.WithContext(context.Background())
	for _, entry := range sealed {
		entry := entry
		eg.Go(func() error {
			worker, err := sd.workerManager.GetWorker(ctx, entry.NodeID)
			if err != nil {
				log.Warn("failed to get worker",
					zap.Int64("nodeID", paramtable.GetNodeID()),
					zap.Error(err),
				)
				// skip if node down
				// delete will be processed after loaded again
				return nil
			}
			offlineSegments.Upsert(sd.applyDelete(ctx, entry.NodeID, worker, delRecords, entry.Segments, querypb.DataScope_Historical)...)
			return nil
		})
	}
	if len(growing) > 0 {
		eg.Go(func() error {
			worker, err := sd.workerManager.GetWorker(ctx, paramtable.GetNodeID())
			if err != nil {
				log.Error("failed to get worker(local)",
					zap.Int64("nodeID", paramtable.GetNodeID()),
					zap.Error(err),
				)
				// panic here, local worker shall not have error
				panic(err)
			}
			offlineSegments.Upsert(sd.applyDelete(ctx, paramtable.GetNodeID(), worker, delRecords, growing, querypb.DataScope_Streaming)...)
			return nil
		})
	}

	// not error return in apply delete
	_ = eg.Wait()

	sd.distribution.Unpin(version)
	offlineSegIDs := offlineSegments.Collect()
	if len(offlineSegIDs) > 0 {
		log.Warn("failed to apply delete, mark segment offline", zap.Int64s("offlineSegments", offlineSegIDs))
		sd.markSegmentOffline(offlineSegIDs...)
	}

	metrics.QueryNodeProcessCost.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), metrics.DeleteLabel).
		Observe(float64(tr.ElapseSpan().Milliseconds()))
}

// applyDelete handles delete record and apply them to corresponding workers.
func (sd *shardDelegator) applyDelete(ctx context.Context, nodeID int64, worker cluster.Worker, delRecords map[int64]DeleteData, entries []SegmentEntry, scope querypb.DataScope) []int64 {
	var offlineSegments []int64
	log := sd.getLogger(ctx)
	for _, segmentEntry := range entries {
		log := log.With(
			zap.Int64("segmentID", segmentEntry.SegmentID),
			zap.Int64("workerID", nodeID),
		)
		delRecord, ok := delRecords[segmentEntry.SegmentID]
		if ok {
			log.Debug("delegator plan to applyDelete via worker")
			err := retry.Do(ctx, func() error {
				if sd.Stopped() {
					return retry.Unrecoverable(merr.WrapErrChannelNotAvailable(sd.vchannelName, "channel is unsubscribing"))
				}

				err := worker.Delete(ctx, &querypb.DeleteRequest{
					Base:         commonpbutil.NewMsgBase(commonpbutil.WithTargetID(nodeID)),
					CollectionId: sd.collectionID,
					PartitionId:  segmentEntry.PartitionID,
					VchannelName: sd.vchannelName,
					SegmentId:    segmentEntry.SegmentID,
					PrimaryKeys:  storage.ParsePrimaryKeys2IDs(delRecord.PrimaryKeys),
					Timestamps:   delRecord.Timestamps,
					Scope:        scope,
				})
				if errors.Is(err, merr.ErrNodeNotFound) {
					log.Warn("try to delete data on non-exist node")
					return retry.Unrecoverable(err)
				} else if errors.IsAny(err, merr.ErrSegmentNotFound, merr.ErrSegmentNotLoaded) {
					log.Warn("try to delete data of released segment")
					return nil
				} else if err != nil {
					log.Warn("worker failed to delete on segment",
						zap.Error(err),
					)
					return err
				}
				return nil
			}, retry.Attempts(10))
			if err != nil {
				log.Warn("apply delete for segment failed, marking it offline")
				offlineSegments = append(offlineSegments, segmentEntry.SegmentID)
			}
		}
	}
	return offlineSegments
}

// markSegmentOffline makes segment go offline and waits for QueryCoord to fix.
func (sd *shardDelegator) markSegmentOffline(segmentIDs ...int64) {
	sd.distribution.AddOfflines(segmentIDs...)
}

// addGrowing add growing segment record for delegator.
func (sd *shardDelegator) addGrowing(entries ...SegmentEntry) {
	log := sd.getLogger(context.Background())
	log.Info("add growing segments to delegator", zap.Int64s("segmentIDs", lo.Map(entries, func(entry SegmentEntry, _ int) int64 {
		return entry.SegmentID
	})))
	sd.distribution.AddGrowing(entries...)
}

// LoadGrowing load growing segments locally.
func (sd *shardDelegator) LoadGrowing(ctx context.Context, infos []*querypb.SegmentLoadInfo, version int64) error {
	log := sd.getLogger(ctx)

	segmentIDs := lo.Map(infos, func(info *querypb.SegmentLoadInfo, _ int) int64 { return info.GetSegmentID() })
	log.Info("loading growing segments...", zap.Int64s("segmentIDs", segmentIDs))
	loaded, err := sd.loader.Load(ctx, sd.collectionID, segments.SegmentTypeGrowing, version, infos...)
	if err != nil {
		log.Warn("failed to load growing segment", zap.Error(err))
		return err
	}

	for _, segment := range loaded {
		log := log.With(
			zap.Int64("segmentID", segment.ID()),
		)
		deletedPks, deletedTss := sd.GetLevel0Deletions(segment.Partition())
		if len(deletedPks) == 0 {
			continue
		}

		log.Info("forwarding L0 delete records...", zap.Int("deletionCount", len(deletedPks)))
		err = segment.Delete(ctx, deletedPks, deletedTss)
		if err != nil {
			log.Warn("failed to forward L0 deletions to growing segment",
				zap.Error(err),
			)

			// clear loaded growing segments
			for _, segment := range loaded {
				segment.Release()
			}
			return err
		}
	}

	segmentIDs = lo.Map(loaded, func(segment segments.Segment, _ int) int64 { return segment.ID() })
	log.Info("load growing segments done", zap.Int64s("segmentIDs", segmentIDs))

	for _, candidate := range loaded {
		sd.pkOracle.Register(candidate, paramtable.GetNodeID())
	}
	sd.addGrowing(lo.Map(loaded, func(segment segments.Segment, _ int) SegmentEntry {
		return SegmentEntry{
			NodeID:        paramtable.GetNodeID(),
			SegmentID:     segment.ID(),
			PartitionID:   segment.Partition(),
			Version:       version,
			TargetVersion: sd.distribution.getTargetVersion(),
		}
	})...)
	return nil
}

// LoadSegments load segments local or remotely depends on the target node.
func (sd *shardDelegator) LoadSegments(ctx context.Context, req *querypb.LoadSegmentsRequest) error {
	if len(req.GetInfos()) == 0 {
		return nil
	}

	log := sd.getLogger(ctx)

	targetNodeID := req.GetDstNodeID()
	// add common log fields
	log = log.With(
		zap.Int64("workID", req.GetDstNodeID()),
		zap.Int64s("segments", lo.Map(req.GetInfos(), func(info *querypb.SegmentLoadInfo, _ int) int64 { return info.GetSegmentID() })),
	)

	worker, err := sd.workerManager.GetWorker(ctx, targetNodeID)
	if err != nil {
		log.Warn("delegator failed to find worker", zap.Error(err))
		return err
	}

	// load bloom filter only when candidate not exists
	infos := lo.Filter(req.GetInfos(), func(info *querypb.SegmentLoadInfo, _ int) bool {
		return !sd.pkOracle.Exists(pkoracle.NewCandidateKey(info.GetSegmentID(), info.GetPartitionID(), commonpb.SegmentState_Sealed), targetNodeID)
	})
	candidates, err := sd.loader.LoadBloomFilterSet(ctx, req.GetCollectionID(), req.GetVersion(), infos...)
	if err != nil {
		log.Warn("failed to load bloom filter set for segment", zap.Error(err))
		return err
	}

	req.Base.TargetID = req.GetDstNodeID()
	log.Debug("worker loads segments...")

	sLoad := func(ctx context.Context, req *querypb.LoadSegmentsRequest) error {
		segmentID := req.GetInfos()[0].GetSegmentID()
		nodeID := req.GetDstNodeID()
		_, err, _ := sd.sf.Do(fmt.Sprintf("%d-%d", nodeID, segmentID), func() (struct{}, error) {
			err := worker.LoadSegments(ctx, req)
			return struct{}{}, err
		})
		return err
	}

	// separate infos into different load task
	if len(req.GetInfos()) > 1 {
		var reqs []*querypb.LoadSegmentsRequest
		for _, info := range req.GetInfos() {
			newReq := typeutil.Clone(req)
			newReq.Infos = []*querypb.SegmentLoadInfo{info}
			reqs = append(reqs, newReq)
		}

		group, ctx := errgroup.WithContext(ctx)
		for _, req := range reqs {
			req := req
			group.Go(func() error {
				return sLoad(ctx, req)
			})
		}
		err = group.Wait()
	} else {
		err = sLoad(ctx, req)
	}

	if err != nil {
		log.Warn("worker failed to load segments", zap.Error(err))
		return err
	}
	log.Debug("work loads segments done")

	// load index segment need no stream delete and distribution change
	if req.GetLoadScope() == querypb.LoadScope_Index {
		return nil
	}

	entries := lo.Map(req.GetInfos(), func(info *querypb.SegmentLoadInfo, _ int) SegmentEntry {
		return SegmentEntry{
			SegmentID:   info.GetSegmentID(),
			PartitionID: info.GetPartitionID(),
			NodeID:      req.GetDstNodeID(),
			Version:     req.GetVersion(),
		}
	})
	if req.GetInfos()[0].GetLevel() == datapb.SegmentLevel_L0 {
		sd.GenerateLevel0DeletionCache()
	} else {
		log.Debug("load delete...")
		err = sd.loadStreamDelete(ctx, candidates, infos, req.GetDeltaPositions(), targetNodeID, worker, entries)
		if err != nil {
			log.Warn("load stream delete failed", zap.Error(err))
			return err
		}
	}

	// alter distribution
	sd.distribution.AddDistributions(entries...)

	return nil
}

func (sd *shardDelegator) GetLevel0Deletions(partitionID int64) ([]storage.PrimaryKey, []storage.Timestamp) {
	sd.level0Mut.RLock()
	deleteData, ok1 := sd.level0Deletions[partitionID]
	allPartitionsDeleteData, ok2 := sd.level0Deletions[common.InvalidPartitionID]
	sd.level0Mut.RUnlock()
	// we may need to merge the specified partition deletions and the all partitions deletions,
	// so release the mutex as early as possible.

	if ok1 && ok2 {
		pks := make([]storage.PrimaryKey, 0, deleteData.RowCount+allPartitionsDeleteData.RowCount)
		tss := make([]storage.Timestamp, 0, deleteData.RowCount+allPartitionsDeleteData.RowCount)

		i := 0
		j := 0
		for i < int(deleteData.RowCount) || j < int(allPartitionsDeleteData.RowCount) {
			if i == int(deleteData.RowCount) {
				pks = append(pks, allPartitionsDeleteData.Pks[j])
				tss = append(tss, allPartitionsDeleteData.Tss[j])
				j++
			} else if j == int(allPartitionsDeleteData.RowCount) {
				pks = append(pks, deleteData.Pks[i])
				tss = append(tss, deleteData.Tss[i])
				i++
			} else if deleteData.Tss[i] < allPartitionsDeleteData.Tss[j] {
				pks = append(pks, deleteData.Pks[i])
				tss = append(tss, deleteData.Tss[i])
				i++
			} else {
				pks = append(pks, allPartitionsDeleteData.Pks[j])
				tss = append(tss, allPartitionsDeleteData.Tss[j])
				j++
			}
		}

		return pks, tss
	} else if ok1 {
		return deleteData.Pks, deleteData.Tss
	} else if ok2 {
		return allPartitionsDeleteData.Pks, allPartitionsDeleteData.Tss
	}

	return nil, nil
}

func (sd *shardDelegator) GenerateLevel0DeletionCache() {
	level0Segments := sd.segmentManager.GetBy(segments.WithLevel(datapb.SegmentLevel_L0))
	deletions := make(map[int64]*storage.DeleteData)
	for _, segment := range level0Segments {
		segment := segment.(*segments.L0Segment)
		pks, tss := segment.DeleteRecords()
		deleteData, ok := deletions[segment.Partition()]
		if !ok {
			deleteData = storage.NewDeleteData(pks, tss)
		} else {
			deleteData.AppendBatch(pks, tss)
		}
		deletions[segment.Partition()] = deleteData
	}

	type DeletePair struct {
		Pk storage.PrimaryKey
		Ts storage.Timestamp
	}
	for _, deleteData := range deletions {
		pairs := make([]DeletePair, deleteData.RowCount)
		for i := range deleteData.Pks {
			pairs[i] = DeletePair{deleteData.Pks[i], deleteData.Tss[i]}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Ts < pairs[j].Ts
		})
		for i := range pairs {
			deleteData.Pks[i], deleteData.Tss[i] = pairs[i].Pk, pairs[i].Ts
		}
	}

	sd.level0Mut.Lock()
	defer sd.level0Mut.Unlock()
	totalSize := int64(0)
	for _, delete := range deletions {
		totalSize += delete.Size()
	}
	metrics.QueryNodeLevelZeroSize.WithLabelValues(
		fmt.Sprint(paramtable.GetNodeID()),
		fmt.Sprint(sd.collectionID),
		sd.vchannelName,
	).Set(float64(totalSize))
	sd.level0Deletions = deletions
}

func (sd *shardDelegator) loadStreamDelete(ctx context.Context,
	candidates []*pkoracle.BloomFilterSet,
	infos []*querypb.SegmentLoadInfo,
	deltaPositions []*msgpb.MsgPosition,
	targetNodeID int64,
	worker cluster.Worker,
	entries []SegmentEntry,
) error {
	log := sd.getLogger(ctx)

	idCandidates := lo.SliceToMap(candidates, func(candidate *pkoracle.BloomFilterSet) (int64, *pkoracle.BloomFilterSet) {
		return candidate.ID(), candidate
	})

	sd.deleteMut.RLock()
	defer sd.deleteMut.RUnlock()
	// apply buffered delete for new segments
	// no goroutines here since qnv2 has no load merging logic
	for _, info := range infos {
		log := log.With(
			zap.Int64("segmentID", info.GetSegmentID()),
		)
		candidate := idCandidates[info.GetSegmentID()]
		position := info.GetDeltaPosition()
		if position == nil { // for compatibility of rolling upgrade from 2.2.x to 2.3
			// During rolling upgrade, Querynode(2.3) may receive merged LoadSegmentRequest
			// from QueryCoord(2.2); In version 2.2.x, only segments with the same dmlChannel
			// can be merged, and deltaPositions will be merged into a single deltaPosition,
			// so we should use `deltaPositions[0]` as the seek position for all the segments
			// within the same LoadSegmentRequest.
			position = deltaPositions[0]
		}

		deletedPks, deletedTss := sd.GetLevel0Deletions(candidate.Partition())
		deleteData := &storage.DeleteData{}
		for i, pk := range deletedPks {
			if candidate.MayPkExist(pk) {
				deleteData.Append(pk, deletedTss[i])
			}
		}

		if deleteData.RowCount > 0 {
			log.Info("forward L0 delete to worker...",
				zap.Int64("deleteRowNum", deleteData.RowCount),
			)
			err := worker.Delete(ctx, &querypb.DeleteRequest{
				Base:         commonpbutil.NewMsgBase(commonpbutil.WithTargetID(targetNodeID)),
				CollectionId: info.GetCollectionID(),
				PartitionId:  info.GetPartitionID(),
				SegmentId:    info.GetSegmentID(),
				PrimaryKeys:  storage.ParsePrimaryKeys2IDs(deleteData.Pks),
				Timestamps:   deleteData.Tss,
			})
			if err != nil {
				log.Warn("failed to apply delete when LoadSegment", zap.Error(err))
				return err
			}
		}

		deleteData = &storage.DeleteData{}
		// start position is dml position for segment
		// if this position is before deleteBuffer's safe ts, it means some delete shall be read from msgstream
		if position.GetTimestamp() < sd.deleteBuffer.SafeTs() {
			log.Info("load delete from stream...")
			streamDeleteData, err := sd.readDeleteFromMsgstream(ctx, position, sd.deleteBuffer.SafeTs(), candidate)
			if err != nil {
				log.Warn("failed to read delete data from msgstream", zap.Error(err))
				return err
			}

			deleteData.Merge(streamDeleteData)
			log.Info("load delete from stream done")
		}

		// list buffered delete
		deleteRecords := sd.deleteBuffer.ListAfter(position.GetTimestamp())
		for _, entry := range deleteRecords {
			for _, record := range entry.Data {
				if record.PartitionID != common.InvalidPartitionID && candidate.Partition() != record.PartitionID {
					continue
				}
				for i, pk := range record.DeleteData.Pks {
					if candidate.MayPkExist(pk) {
						deleteData.Append(pk, record.DeleteData.Tss[i])
					}
				}
			}
		}
		// if delete count not empty, apply
		if deleteData.RowCount > 0 {
			log.Info("forward delete to worker...", zap.Int64("deleteRowNum", deleteData.RowCount))
			err := worker.Delete(ctx, &querypb.DeleteRequest{
				Base:         commonpbutil.NewMsgBase(commonpbutil.WithTargetID(targetNodeID)),
				CollectionId: info.GetCollectionID(),
				PartitionId:  info.GetPartitionID(),
				SegmentId:    info.GetSegmentID(),
				PrimaryKeys:  storage.ParsePrimaryKeys2IDs(deleteData.Pks),
				Timestamps:   deleteData.Tss,
			})
			if err != nil {
				log.Warn("failed to apply delete when LoadSegment", zap.Error(err))
				return err
			}
		}
	}

	// add candidate after load success
	for _, candidate := range candidates {
		log.Info("register sealed segment bfs into pko candidates",
			zap.Int64("segmentID", candidate.ID()),
		)
		sd.pkOracle.Register(candidate, targetNodeID)
	}
	log.Info("load delete done")

	return nil
}

func (sd *shardDelegator) readDeleteFromMsgstream(ctx context.Context, position *msgpb.MsgPosition, safeTs uint64, candidate *pkoracle.BloomFilterSet) (*storage.DeleteData, error) {
	log := sd.getLogger(ctx).With(
		zap.String("channel", position.ChannelName),
		zap.Int64("segmentID", candidate.ID()),
	)
	stream, err := sd.factory.NewTtMsgStream(ctx)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	vchannelName := position.ChannelName
	pChannelName := funcutil.ToPhysicalChannel(vchannelName)
	position.ChannelName = pChannelName

	ts, _ := tsoutil.ParseTS(position.Timestamp)

	// Random the subname in case we trying to load same delta at the same time
	subName := fmt.Sprintf("querynode-delta-loader-%d-%d-%d", paramtable.GetNodeID(), sd.collectionID, rand.Int())
	log.Info("from dml check point load delete", zap.Any("position", position), zap.String("vChannel", vchannelName), zap.String("subName", subName), zap.Time("positionTs", ts))
	err = stream.AsConsumer(context.TODO(), []string{pChannelName}, subName, mqwrapper.SubscriptionPositionUnknown)
	if err != nil {
		return nil, err
	}

	err = stream.Seek(context.TODO(), []*msgpb.MsgPosition{position})
	if err != nil {
		return nil, err
	}

	result := &storage.DeleteData{}
	hasMore := true
	for hasMore {
		select {
		case <-ctx.Done():
			log.Debug("read delta msg from seek position done", zap.Error(ctx.Err()))
			return nil, ctx.Err()
		case msgPack, ok := <-stream.Chan():
			if !ok {
				err = fmt.Errorf("stream channel closed, pChannelName=%v, msgID=%v", pChannelName, position.GetMsgID())
				log.Warn("fail to read delta msg",
					zap.String("pChannelName", pChannelName),
					zap.Binary("msgID", position.GetMsgID()),
					zap.Error(err),
				)
				return nil, err
			}

			if msgPack == nil {
				continue
			}

			for _, tsMsg := range msgPack.Msgs {
				if tsMsg.Type() == commonpb.MsgType_Delete {
					dmsg := tsMsg.(*msgstream.DeleteMsg)
					if dmsg.CollectionID != sd.collectionID || dmsg.GetPartitionID() != candidate.Partition() {
						continue
					}

					for idx, pk := range storage.ParseIDs2PrimaryKeys(dmsg.GetPrimaryKeys()) {
						if candidate.MayPkExist(pk) {
							result.Pks = append(result.Pks, pk)
							result.Tss = append(result.Tss, dmsg.Timestamps[idx])
						}
					}
				}
			}

			// reach safe ts
			if safeTs <= msgPack.EndPositions[0].GetTimestamp() {
				hasMore = false
			}
		}
	}

	return result, nil
}

// ReleaseSegments releases segments local or remotely depending on the target node.
func (sd *shardDelegator) ReleaseSegments(ctx context.Context, req *querypb.ReleaseSegmentsRequest, force bool) error {
	log := sd.getLogger(ctx)

	targetNodeID := req.GetNodeID()
	level0Segments := typeutil.NewSet(lo.Map(sd.segmentManager.GetBy(segments.WithLevel(datapb.SegmentLevel_L0)), func(segment segments.Segment, _ int) int64 {
		return segment.ID()
	})...)
	hasLevel0 := false
	for _, segmentID := range req.GetSegmentIDs() {
		hasLevel0 = level0Segments.Contain(segmentID)
		if hasLevel0 {
			break
		}
	}

	// add common log fields
	log = log.With(
		zap.Int64s("segmentIDs", req.GetSegmentIDs()),
		zap.Int64("nodeID", req.GetNodeID()),
		zap.String("scope", req.GetScope().String()),
		zap.Bool("force", force))

	log.Info("delegator start to release segments")
	// alter distribution first
	var sealed, growing []SegmentEntry
	convertSealed := func(segmentID int64, _ int) SegmentEntry {
		return SegmentEntry{
			SegmentID: segmentID,
			NodeID:    targetNodeID,
		}
	}
	convertGrowing := func(segmentID int64, _ int) SegmentEntry {
		return SegmentEntry{
			SegmentID: segmentID,
		}
	}
	switch req.GetScope() {
	case querypb.DataScope_All:
		sealed = lo.Map(req.GetSegmentIDs(), convertSealed)
		growing = lo.Map(req.GetSegmentIDs(), convertGrowing)
	case querypb.DataScope_Streaming:
		growing = lo.Map(req.GetSegmentIDs(), convertGrowing)
	case querypb.DataScope_Historical:
		sealed = lo.Map(req.GetSegmentIDs(), convertSealed)
	}

	signal := sd.distribution.RemoveDistributions(sealed, growing)
	// wait cleared signal
	<-signal
	if len(sealed) > 0 {
		sd.pkOracle.Remove(
			pkoracle.WithSegmentIDs(lo.Map(sealed, func(entry SegmentEntry, _ int) int64 { return entry.SegmentID })...),
			pkoracle.WithSegmentType(commonpb.SegmentState_Sealed),
			pkoracle.WithWorkerID(targetNodeID),
		)
	}
	if len(growing) > 0 {
		sd.pkOracle.Remove(
			pkoracle.WithSegmentIDs(lo.Map(growing, func(entry SegmentEntry, _ int) int64 { return entry.SegmentID })...),
			pkoracle.WithSegmentType(commonpb.SegmentState_Growing),
		)
	}

	if !force {
		worker, err := sd.workerManager.GetWorker(ctx, targetNodeID)
		if err != nil {
			log.Warn("delegator failed to find worker",
				zap.Error(err),
			)
			return err
		}
		req.Base.TargetID = targetNodeID
		err = worker.ReleaseSegments(ctx, req)
		if err != nil {
			log.Warn("worker failed to release segments",
				zap.Error(err),
			)
		}
		return err
	}

	if hasLevel0 {
		sd.GenerateLevel0DeletionCache()
	}

	return nil
}

func (sd *shardDelegator) SyncTargetVersion(newVersion int64, growingInTarget []int64,
	sealedInTarget []int64, droppedInTarget []int64, checkpoint *msgpb.MsgPosition,
) {
	growings := sd.segmentManager.GetBy(
		segments.WithType(segments.SegmentTypeGrowing),
		segments.WithChannel(sd.vchannelName),
	)

	sealedSet := typeutil.NewUniqueSet(sealedInTarget...)
	growingSet := typeutil.NewUniqueSet(growingInTarget...)
	droppedSet := typeutil.NewUniqueSet(droppedInTarget...)
	redundantGrowing := typeutil.NewUniqueSet()
	for _, s := range growings {
		if growingSet.Contain(s.ID()) {
			continue
		}

		// sealed segment already exists, make growing segment redundant
		if sealedSet.Contain(s.ID()) {
			redundantGrowing.Insert(s.ID())
		}

		// sealed segment already dropped, make growing segment redundant
		if droppedSet.Contain(s.ID()) {
			redundantGrowing.Insert(s.ID())
		}
	}
	redundantGrowingIDs := redundantGrowing.Collect()
	if len(redundantGrowing) > 0 {
		log.Warn("found redundant growing segments",
			zap.Int64s("growingSegments", redundantGrowingIDs))
	}
	sd.distribution.SyncTargetVersion(newVersion, growingInTarget, sealedInTarget, redundantGrowingIDs)
	sd.deleteBuffer.TryDiscard(checkpoint.GetTimestamp())
}

func (sd *shardDelegator) GetTargetVersion() int64 {
	return sd.distribution.getTargetVersion()
}
