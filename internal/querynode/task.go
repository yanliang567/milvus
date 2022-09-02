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
	"context"
	"errors"
	"fmt"
	"runtime/debug"

	"go.uber.org/zap"

	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type task interface {
	ID() UniqueID // return ReqID
	Timestamp() Timestamp
	PreExecute(ctx context.Context) error
	Execute(ctx context.Context) error
	PostExecute(ctx context.Context) error
	WaitToFinish() error
	Notify(err error)
	OnEnqueue() error
}

type baseTask struct {
	done chan error
	ctx  context.Context
	id   UniqueID
	ts   Timestamp
}

func (b *baseTask) Ctx() context.Context {
	return b.ctx
}

func (b *baseTask) OnEnqueue() error {
	return nil
}

func (b *baseTask) ID() UniqueID {
	return b.id
}

func (b *baseTask) Timestamp() Timestamp {
	return b.ts
}

func (b *baseTask) PreExecute(ctx context.Context) error {
	return nil
}

func (b *baseTask) PostExecute(ctx context.Context) error {
	return nil
}

func (b *baseTask) WaitToFinish() error {
	err := <-b.done
	return err
}

func (b *baseTask) Notify(err error) {
	b.done <- err
}

type watchDmChannelsTask struct {
	baseTask
	req  *queryPb.WatchDmChannelsRequest
	node *QueryNode
}

type loadSegmentsTask struct {
	baseTask
	req  *queryPb.LoadSegmentsRequest
	node *QueryNode
}

type releaseCollectionTask struct {
	baseTask
	req  *queryPb.ReleaseCollectionRequest
	node *QueryNode
}

type releasePartitionsTask struct {
	baseTask
	req  *queryPb.ReleasePartitionsRequest
	node *QueryNode
}

// watchDmChannelsTask
func (w *watchDmChannelsTask) Execute(ctx context.Context) (err error) {
	collectionID := w.req.CollectionID
	partitionIDs := w.req.GetPartitionIDs()

	lType := w.req.GetLoadMeta().GetLoadType()
	if lType == queryPb.LoadType_UnKnownType {
		// if no partitionID is specified, load type is load collection
		if len(partitionIDs) != 0 {
			lType = queryPb.LoadType_LoadPartition
		} else {
			lType = queryPb.LoadType_LoadCollection
		}
	}

	// get all vChannels
	var vChannels, pChannels []Channel
	VPChannels := make(map[string]string) // map[vChannel]pChannel
	for _, info := range w.req.Infos {
		v := info.ChannelName
		p := funcutil.ToPhysicalChannel(info.ChannelName)
		vChannels = append(vChannels, v)
		pChannels = append(pChannels, p)
		VPChannels[v] = p
	}

	if len(VPChannels) != len(vChannels) {
		return errors.New("get physical channels failed, illegal channel length, collectionID = " + fmt.Sprintln(collectionID))
	}

	log.Info("Starting WatchDmChannels ...",
		zap.String("collectionName", w.req.Schema.Name),
		zap.Int64("collectionID", collectionID),
		zap.Int64("replicaID", w.req.GetReplicaID()),
		zap.Any("load type", lType),
		zap.Strings("vChannels", vChannels),
		zap.Strings("pChannels", pChannels),
	)

	// init collection meta
	coll := w.node.metaReplica.addCollection(collectionID, w.req.Schema)

	loadedChannelCounter := 0
	for _, toLoadChannel := range vChannels {
		for _, loadedChannel := range coll.vChannels {
			if toLoadChannel == loadedChannel {
				loadedChannelCounter++
				break
			}
		}
	}

	// check if all channels has been loaded, if YES, should do nothing and return
	// in case of query coord trigger same watchDmChannelTask on multi
	if len(vChannels) == loadedChannelCounter {
		log.Warn("All channel has been loaded, skip this watchDmChannelsTask")
		return nil
	}

	//add shard cluster
	for _, vchannel := range vChannels {
		w.node.ShardClusterService.addShardCluster(w.req.GetCollectionID(), w.req.GetReplicaID(), vchannel)
	}

	defer func() {
		if err != nil {
			for _, vchannel := range vChannels {
				w.node.ShardClusterService.releaseShardCluster(vchannel)
			}
		}
	}()

	// load growing segments
	unFlushedSegments := make([]*queryPb.SegmentLoadInfo, 0)
	unFlushedSegmentIDs := make([]UniqueID, 0)
	for _, info := range w.req.Infos {
		for _, ufInfoID := range info.GetUnflushedSegmentIds() {
			// unFlushed segment may not have binLogs, skip loading
			ufInfo := w.req.GetSegmentInfos()[ufInfoID]
			if ufInfo == nil {
				log.Warn("an unflushed segment is not found in segment infos", zap.Int64("segment ID", ufInfoID))
				continue
			}
			if len(ufInfo.GetBinlogs()) > 0 {
				unFlushedSegments = append(unFlushedSegments, &queryPb.SegmentLoadInfo{
					SegmentID:     ufInfo.ID,
					PartitionID:   ufInfo.PartitionID,
					CollectionID:  ufInfo.CollectionID,
					BinlogPaths:   ufInfo.Binlogs,
					NumOfRows:     ufInfo.NumOfRows,
					Statslogs:     ufInfo.Statslogs,
					Deltalogs:     ufInfo.Deltalogs,
					InsertChannel: ufInfo.InsertChannel,
				})
				unFlushedSegmentIDs = append(unFlushedSegmentIDs, ufInfo.GetID())
			}
		}
	}
	req := &queryPb.LoadSegmentsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_LoadSegments,
			MsgID:   w.req.Base.MsgID, // use parent task's msgID
		},
		Infos:        unFlushedSegments,
		CollectionID: collectionID,
		Schema:       w.req.GetSchema(),
		LoadMeta:     w.req.GetLoadMeta(),
	}

	// update partition info from unFlushedSegments and loadMeta
	for _, info := range req.Infos {
		err = w.node.metaReplica.addPartition(collectionID, info.PartitionID)
		if err != nil {
			return err
		}
	}
	for _, partitionID := range req.GetLoadMeta().GetPartitionIDs() {
		err = w.node.metaReplica.addPartition(collectionID, partitionID)
		if err != nil {
			return err
		}
	}

	log.Info("loading growing segments in WatchDmChannels...",
		zap.Int64("collectionID", collectionID),
		zap.Int64s("unFlushedSegmentIDs", unFlushedSegmentIDs),
	)
	err = w.node.loader.LoadSegment(req, segmentTypeGrowing)
	if err != nil {
		log.Warn(err.Error())
		return err
	}
	log.Info("successfully load growing segments done in WatchDmChannels",
		zap.Int64("collectionID", collectionID),
		zap.Int64s("unFlushedSegmentIDs", unFlushedSegmentIDs),
	)

	// remove growing segment if watch dmChannels failed
	defer func() {
		if err != nil {
			collection, err2 := w.node.metaReplica.getCollectionByID(collectionID)
			if err2 == nil {
				collection.Lock()
				defer collection.Unlock()
				for _, segmentID := range unFlushedSegmentIDs {
					w.node.metaReplica.removeSegment(segmentID, segmentTypeGrowing)
				}
			}
		}
	}()

	consumeSubName := funcutil.GenChannelSubName(Params.CommonCfg.QueryNodeSubName, collectionID, Params.QueryNodeCfg.GetNodeID())

	// group channels by to seeking or consuming
	channel2SeekPosition := make(map[string]*internalpb.MsgPosition)
	channel2AsConsumerPosition := make(map[string]*internalpb.MsgPosition)
	for _, info := range w.req.Infos {
		if info.SeekPosition == nil || len(info.SeekPosition.MsgID) == 0 {
			channel2AsConsumerPosition[info.ChannelName] = info.SeekPosition
			continue
		}
		info.SeekPosition.MsgGroup = consumeSubName
		channel2SeekPosition[info.ChannelName] = info.SeekPosition
	}
	log.Info("watchDMChannel, group channels done", zap.Int64("collectionID", collectionID))

	// add excluded segments for unFlushed segments,
	// unFlushed segments before check point should be filtered out.
	unFlushedCheckPointInfos := make([]*datapb.SegmentInfo, 0)
	for _, info := range w.req.Infos {
		for _, ufsID := range info.GetUnflushedSegmentIds() {
			unFlushedCheckPointInfos = append(unFlushedCheckPointInfos, w.req.SegmentInfos[ufsID])
		}
	}
	w.node.metaReplica.addExcludedSegments(collectionID, unFlushedCheckPointInfos)
	unflushedSegmentIDs := make([]UniqueID, len(unFlushedCheckPointInfos))
	for i, segInfo := range unFlushedCheckPointInfos {
		unflushedSegmentIDs[i] = segInfo.GetID()
	}
	log.Info("watchDMChannel, add check points info for unflushed segments done",
		zap.Int64("collectionID", collectionID),
		zap.Any("unflushedSegmentIDs", unflushedSegmentIDs),
	)

	// add excluded segments for flushed segments,
	// flushed segments with later check point than seekPosition should be filtered out.
	flushedCheckPointInfos := make([]*datapb.SegmentInfo, 0)
	for _, info := range w.req.Infos {
		for _, flushedSegmentID := range info.GetFlushedSegmentIds() {
			flushedSegment := w.req.SegmentInfos[flushedSegmentID]
			for _, position := range channel2SeekPosition {
				if flushedSegment.DmlPosition != nil &&
					flushedSegment.DmlPosition.ChannelName == position.ChannelName &&
					flushedSegment.DmlPosition.Timestamp > position.Timestamp {
					flushedCheckPointInfos = append(flushedCheckPointInfos, flushedSegment)
				}
			}
		}
	}
	w.node.metaReplica.addExcludedSegments(collectionID, flushedCheckPointInfos)
	flushedSegmentIDs := make([]UniqueID, len(flushedCheckPointInfos))
	for i, segInfo := range flushedCheckPointInfos {
		flushedSegmentIDs[i] = segInfo.GetID()
	}
	log.Info("watchDMChannel, add check points info for flushed segments done",
		zap.Int64("collectionID", collectionID),
		zap.Any("flushedSegmentIDs", flushedSegmentIDs),
	)

	// add excluded segments for dropped segments,
	// dropped segments with later check point than seekPosition should be filtered out.
	droppedCheckPointInfos := make([]*datapb.SegmentInfo, 0)
	for _, info := range w.req.Infos {
		for _, droppedSegmentID := range info.GetDroppedSegmentIds() {
			droppedSegment := w.req.SegmentInfos[droppedSegmentID]
			for _, position := range channel2SeekPosition {
				if droppedSegment != nil &&
					droppedSegment.DmlPosition.ChannelName == position.ChannelName &&
					droppedSegment.DmlPosition.Timestamp > position.Timestamp {
					droppedCheckPointInfos = append(droppedCheckPointInfos, droppedSegment)
				}
			}
		}
	}
	w.node.metaReplica.addExcludedSegments(collectionID, droppedCheckPointInfos)
	droppedSegmentIDs := make([]UniqueID, len(droppedCheckPointInfos))
	for i, segInfo := range droppedCheckPointInfos {
		droppedSegmentIDs[i] = segInfo.GetID()
	}
	log.Info("watchDMChannel, add check points info for dropped segments done",
		zap.Int64("collectionID", collectionID),
		zap.Any("droppedSegmentIDs", droppedSegmentIDs),
	)

	// add flow graph
	channel2FlowGraph, err := w.node.dataSyncService.addFlowGraphsForDMLChannels(collectionID, vChannels)
	if err != nil {
		log.Warn("watchDMChannel, add flowGraph for dmChannels failed", zap.Int64("collectionID", collectionID), zap.Strings("vChannels", vChannels), zap.Error(err))
		return err
	}
	log.Info("Query node add DML flow graphs", zap.Int64("collectionID", collectionID), zap.Any("channels", vChannels))

	// channels as consumer
	for channel, fg := range channel2FlowGraph {
		if _, ok := channel2AsConsumerPosition[channel]; ok {
			// use pChannel to consume
			err = fg.consumeFlowGraph(VPChannels[channel], consumeSubName)
			if err != nil {
				log.Error("msgStream as consumer failed for dmChannels", zap.Int64("collectionID", collectionID), zap.String("vChannel", channel))
				break
			}
		}

		if pos, ok := channel2SeekPosition[channel]; ok {
			pos.MsgGroup = consumeSubName
			// use pChannel to seek
			pos.ChannelName = VPChannels[channel]
			err = fg.seekQueryNodeFlowGraph(pos)
			if err != nil {
				log.Error("msgStream seek failed for dmChannels", zap.Int64("collectionID", collectionID), zap.String("vChannel", channel))
				break
			}
		}
	}

	if err != nil {
		log.Warn("watchDMChannel, add flowGraph for dmChannels failed", zap.Int64("collectionID", collectionID), zap.Strings("vChannels", vChannels), zap.Error(err))
		for _, fg := range channel2FlowGraph {
			fg.flowGraph.Close()
		}
		gcChannels := make([]Channel, 0)
		for channel := range channel2FlowGraph {
			gcChannels = append(gcChannels, channel)
		}
		w.node.dataSyncService.removeFlowGraphsByDMLChannels(gcChannels)
		return err
	}

	log.Info("watchDMChannel, add flowGraph for dmChannels success", zap.Int64("collectionID", collectionID), zap.Strings("vChannels", vChannels))

	coll.addVChannels(vChannels)
	coll.addPChannels(pChannels)
	coll.setLoadType(lType)

	log.Info("watchDMChannel, init replica done", zap.Int64("collectionID", collectionID), zap.Strings("vChannels", vChannels))

	// create tSafe
	for _, channel := range vChannels {
		w.node.tSafeReplica.addTSafe(channel)
	}

	// add tsafe watch in query shard if exists
	for _, dmlChannel := range vChannels {
		w.node.queryShardService.addQueryShard(collectionID, dmlChannel, w.req.GetReplicaID())
	}

	// start flow graphs
	for _, fg := range channel2FlowGraph {
		fg.flowGraph.Start()
	}

	log.Info("WatchDmChannels done", zap.Int64("collectionID", collectionID), zap.Strings("vChannels", vChannels))
	return nil
}

// internal helper function to subscribe delta channel
func (l *loadSegmentsTask) watchDeltaChannel(vchanName []string) error {
	collectionID := l.req.CollectionID
	var vDeltaChannels, pDeltaChannels []string
	VPDeltaChannels := make(map[string]string)
	for _, v := range vchanName {
		dc, err := funcutil.ConvertChannelName(v, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta)
		if err != nil {
			log.Warn("watchDeltaChannels, failed to convert deltaChannel from dmlChannel", zap.String("DmlChannel", v), zap.Error(err))
			return err
		}
		p := funcutil.ToPhysicalChannel(dc)
		vDeltaChannels = append(vDeltaChannels, dc)
		pDeltaChannels = append(pDeltaChannels, p)
		VPDeltaChannels[dc] = p
	}
	log.Info("Starting WatchDeltaChannels ...",
		zap.Int64("collectionID", collectionID),
		zap.Any("channels", VPDeltaChannels),
	)

	coll, err := l.node.metaReplica.getCollectionByID(collectionID)
	if err != nil {
		return err
	}

	channel2FlowGraph, err := l.node.dataSyncService.addFlowGraphsForDeltaChannels(collectionID, vDeltaChannels)
	if err != nil {
		log.Warn("watchDeltaChannel, add flowGraph for deltaChannel failed", zap.Int64("collectionID", collectionID), zap.Strings("vDeltaChannels", vDeltaChannels), zap.Error(err))
		return err
	}
	consumeSubName := funcutil.GenChannelSubName(Params.CommonCfg.QueryNodeSubName, collectionID, Params.QueryNodeCfg.GetNodeID())

	// channels as consumer
	for channel, fg := range channel2FlowGraph {
		pchannel := VPDeltaChannels[channel]
		// use pChannel to consume
		err = fg.consumeFlowGraphFromLatest(pchannel, consumeSubName)
		if err != nil {
			log.Error("msgStream as consumer failed for deltaChannels", zap.Int64("collectionID", collectionID), zap.Strings("vDeltaChannels", vDeltaChannels))
			break
		}
	}

	if err != nil {
		log.Warn("watchDeltaChannel, add flowGraph for deltaChannel failed", zap.Int64("collectionID", collectionID), zap.Strings("vDeltaChannels", vDeltaChannels), zap.Error(err))
		for _, fg := range channel2FlowGraph {
			fg.flowGraph.Close()
		}
		gcChannels := make([]Channel, 0)
		for channel := range channel2FlowGraph {
			gcChannels = append(gcChannels, channel)
		}
		l.node.dataSyncService.removeFlowGraphsByDeltaChannels(gcChannels)
		return err
	}

	log.Info("watchDeltaChannel, add flowGraph for deltaChannel success", zap.Int64("collectionID", collectionID), zap.Strings("vDeltaChannels", vDeltaChannels))

	//set collection replica
	coll.addVDeltaChannels(vDeltaChannels)
	coll.addPDeltaChannels(pDeltaChannels)

	// create tSafe
	for _, channel := range vDeltaChannels {
		l.node.tSafeReplica.addTSafe(channel)
	}

	// add tsafe watch in query shard if exists, we find no way to handle it if query shard not exist
	for _, channel := range vDeltaChannels {
		dmlChannel, err := funcutil.ConvertChannelName(channel, Params.CommonCfg.RootCoordDelta, Params.CommonCfg.RootCoordDml)
		if err != nil {
			log.Error("failed to convert delta channel to dml", zap.String("channel", channel), zap.Error(err))
			panic(err)
		}
		err = l.node.queryShardService.addQueryShard(collectionID, dmlChannel, l.req.GetReplicaID())
		if err != nil {
			log.Error("failed to add shard Service to query shard", zap.String("channel", channel), zap.Error(err))
			panic(err)
		}
	}

	// start flow graphs
	for _, fg := range channel2FlowGraph {
		fg.flowGraph.Start()
	}

	log.Info("WatchDeltaChannels done", zap.Int64("collectionID", collectionID), zap.String("ChannelIDs", fmt.Sprintln(vDeltaChannels)))
	return nil
}

// loadSegmentsTask
func (l *loadSegmentsTask) PreExecute(ctx context.Context) error {
	log.Info("LoadSegmentTask PreExecute start", zap.Int64("msgID", l.req.Base.MsgID))
	var err error
	// init meta
	collectionID := l.req.GetCollectionID()
	l.node.metaReplica.addCollection(collectionID, l.req.GetSchema())
	for _, partitionID := range l.req.GetLoadMeta().GetPartitionIDs() {
		err = l.node.metaReplica.addPartition(collectionID, partitionID)
		if err != nil {
			return err
		}
	}

	// filter segments that are already loaded in this querynode
	var filteredInfos []*queryPb.SegmentLoadInfo
	for _, info := range l.req.Infos {
		has, err := l.node.metaReplica.hasSegment(info.SegmentID, segmentTypeSealed)
		if err != nil {
			return err
		}
		if !has {
			filteredInfos = append(filteredInfos, info)
		} else {
			log.Debug("ignore segment that is already loaded", zap.Int64("collectionID", info.SegmentID), zap.Int64("segmentID", info.SegmentID))
		}
	}
	l.req.Infos = filteredInfos
	log.Info("LoadSegmentTask PreExecute done", zap.Int64("msgID", l.req.Base.MsgID))
	return nil
}

func (l *loadSegmentsTask) Execute(ctx context.Context) error {
	log.Info("LoadSegmentTask Execute start", zap.Int64("msgID", l.req.Base.MsgID))
	err := l.node.loader.LoadSegment(l.req, segmentTypeSealed)
	if err != nil {
		log.Warn("failed to load segment", zap.Int64("collectionID", l.req.CollectionID),
			zap.Int64("replicaID", l.req.ReplicaID), zap.Error(err))
		return err
	}
	vchanName := make([]string, 0)
	for _, deltaPosition := range l.req.DeltaPositions {
		vchanName = append(vchanName, deltaPosition.ChannelName)
	}
	// TODO delta channel need to released 1. if other watchDeltaChannel fail 2. when segment release
	err = l.watchDeltaChannel(vchanName)
	if err != nil {
		// roll back
		for _, segment := range l.req.Infos {
			l.node.metaReplica.removeSegment(segment.SegmentID, segmentTypeSealed)
		}
		log.Warn("failed to watch Delta channel while load segment", zap.Int64("collectionID", l.req.CollectionID),
			zap.Int64("replicaID", l.req.ReplicaID), zap.Error(err))
		return err
	}

	runningGroup, groupCtx := errgroup.WithContext(l.ctx)
	for _, deltaPosition := range l.req.DeltaPositions {
		pos := deltaPosition
		runningGroup.Go(func() error {
			// reload data from dml channel
			return l.node.loader.FromDmlCPLoadDelete(groupCtx, l.req.CollectionID, pos)
		})
	}
	err = runningGroup.Wait()
	if err != nil {
		for _, segment := range l.req.Infos {
			l.node.metaReplica.removeSegment(segment.SegmentID, segmentTypeSealed)
		}
		log.Warn("failed to load delete data while load segment", zap.Int64("collectionID", l.req.CollectionID),
			zap.Int64("replicaID", l.req.ReplicaID), zap.Error(err))
		return err
	}

	log.Info("LoadSegmentTask Execute done", zap.Int64("collectionID", l.req.CollectionID),
		zap.Int64("replicaID", l.req.ReplicaID), zap.Int64("msgID", l.req.Base.MsgID))
	return nil
}

func (r *releaseCollectionTask) Execute(ctx context.Context) error {
	log.Info("Execute release collection task", zap.Any("collectionID", r.req.CollectionID))

	collection, err := r.node.metaReplica.getCollectionByID(r.req.CollectionID)
	if err != nil {
		return err
	}
	// set release time
	log.Info("set release time", zap.Any("collectionID", r.req.CollectionID))
	collection.setReleaseTime(r.req.Base.Timestamp, true)

	// remove all flow graphs of the target collection
	vChannels := collection.getVChannels()
	vDeltaChannels := collection.getVDeltaChannels()
	r.node.dataSyncService.removeFlowGraphsByDMLChannels(vChannels)
	r.node.dataSyncService.removeFlowGraphsByDeltaChannels(vDeltaChannels)

	// remove all tSafes of the target collection
	for _, channel := range vChannels {
		r.node.tSafeReplica.removeTSafe(channel)
	}
	for _, channel := range vDeltaChannels {
		r.node.tSafeReplica.removeTSafe(channel)
	}
	log.Info("Release tSafe in releaseCollectionTask",
		zap.Int64("collectionID", r.req.CollectionID),
		zap.Strings("vChannels", vChannels),
		zap.Strings("vDeltaChannels", vDeltaChannels),
	)

	r.node.metaReplica.removeExcludedSegments(r.req.CollectionID)
	r.node.queryShardService.releaseCollection(r.req.CollectionID)
	r.node.ShardClusterService.releaseCollection(r.req.CollectionID)
	err = r.node.metaReplica.removeCollection(r.req.CollectionID)
	if err != nil {
		return err
	}

	debug.FreeOSMemory()
	log.Info("ReleaseCollection done", zap.Int64("collectionID", r.req.CollectionID))
	return nil
}

// releasePartitionsTask
func (r *releasePartitionsTask) Execute(ctx context.Context) error {
	log.Info("Execute release partition task",
		zap.Int64("collectionID", r.req.GetCollectionID()),
		zap.Int64s("partitionIDs", r.req.GetPartitionIDs()))

	coll, err := r.node.metaReplica.getCollectionByID(r.req.CollectionID)
	if err != nil {
		// skip error if collection not found, do clean up job below
		log.Warn("failed to get collection for release partitions", zap.Int64("collectionID", r.req.GetCollectionID()),
			zap.Int64s("partitionIDs", r.req.GetPartitionIDs()))

	}
	log.Info("start release partition", zap.Int64("collectionID", r.req.GetCollectionID()), zap.Int64s("partitionIDs", r.req.GetPartitionIDs()))

	// shall be false if coll is nil
	releaseAll := r.isAllPartitionsReleased(coll)

	if releaseAll {
		// set release time
		log.Info("set release time", zap.Int64("collectionID", r.req.CollectionID))
		coll.setReleaseTime(r.req.Base.Timestamp, true)

		// remove all flow graphs of the target collection
		vChannels := coll.getVChannels()
		vDeltaChannels := coll.getVDeltaChannels()
		r.node.dataSyncService.removeFlowGraphsByDMLChannels(vChannels)
		r.node.dataSyncService.removeFlowGraphsByDeltaChannels(vDeltaChannels)

		// remove all tSafes of the target collection
		for _, channel := range vChannels {
			r.node.tSafeReplica.removeTSafe(channel)
		}
		for _, channel := range vDeltaChannels {
			r.node.tSafeReplica.removeTSafe(channel)
		}
		log.Info("Release tSafe in releaseCollectionTask",
			zap.Int64("collectionID", r.req.CollectionID),
			zap.Strings("vChannels", vChannels),
			zap.Strings("vDeltaChannels", vDeltaChannels),
		)

		r.node.metaReplica.removeExcludedSegments(r.req.CollectionID)
		r.node.queryShardService.releaseCollection(r.req.CollectionID)
		r.node.ShardClusterService.releaseCollection(r.req.CollectionID)
		err = r.node.metaReplica.removeCollection(r.req.CollectionID)
		if err != nil {
			log.Warn("failed to remove collection", zap.Int64("collectionID", r.req.GetCollectionID()),
				zap.Int64s("partitionIDs", r.req.GetPartitionIDs()), zap.Error(err))
		}
	} else {
		for _, id := range r.req.PartitionIDs {
			// remove partition from streaming and historical
			hasPartition := r.node.metaReplica.hasPartition(id)
			if hasPartition {
				err := r.node.metaReplica.removePartition(id)
				if err != nil {
					// not return, try to release all partitions
					log.Warn(err.Error())
				}
			}
		}
	}

	log.Info("Release partition task done",
		zap.Int64("collectionID", r.req.CollectionID),
		zap.Int64s("partitionIDs", r.req.PartitionIDs))
	return nil
}

func (r *releasePartitionsTask) isAllPartitionsReleased(coll *Collection) bool {
	if coll == nil {
		return false
	}
	if len(r.req.GetPartitionIDs()) < len(coll.partitionIDs) && len(coll.partitionIDs) > 0 {
		return false
	}
	parts := make(typeutil.UniqueSet)
	for _, partID := range r.req.GetPartitionIDs() {
		parts.Insert(partID)
	}

	return parts.Contain(coll.partitionIDs...)
}
