package task

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"go.uber.org/zap"
)

const (
	actionTimeout = 120 * time.Second
)

type actionIndex struct {
	Task int64
	Step int
}

type Executor struct {
	doneCh    chan struct{}
	wg        sync.WaitGroup
	meta      *meta.Meta
	dist      *meta.DistributionManager
	broker    meta.Broker
	targetMgr *meta.TargetManager
	cluster   session.Cluster
	nodeMgr   *session.NodeManager

	// Merge load segment requests
	merger *Merger[segmentIndex, *querypb.LoadSegmentsRequest]

	executingActions sync.Map
}

func NewExecutor(meta *meta.Meta,
	dist *meta.DistributionManager,
	broker meta.Broker,
	targetMgr *meta.TargetManager,
	cluster session.Cluster,
	nodeMgr *session.NodeManager) *Executor {
	return &Executor{
		doneCh:    make(chan struct{}),
		meta:      meta,
		dist:      dist,
		broker:    broker,
		targetMgr: targetMgr,
		cluster:   cluster,
		nodeMgr:   nodeMgr,
		merger:    NewMerger[segmentIndex, *querypb.LoadSegmentsRequest](),

		executingActions: sync.Map{},
	}
}

func (ex *Executor) Start(ctx context.Context) {
	ex.merger.Start(ctx)
	ex.scheduleRequests()
}

func (ex *Executor) Stop() {
	ex.merger.Stop()
	ex.wg.Wait()
}

// Execute executes the given action,
// does nothing and returns false if the action is already committed,
// returns true otherwise.
func (ex *Executor) Execute(task Task, step int) bool {
	index := actionIndex{
		Task: task.ID(),
		Step: step,
	}
	_, exist := ex.executingActions.LoadOrStore(index, struct{}{})
	if exist {
		return false
	}

	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int("step", step),
		zap.Int64("source", task.SourceID()),
	)

	go func() {
		log.Info("execute the action of task")
		switch task.Actions()[step].(type) {
		case *SegmentAction:
			ex.executeSegmentAction(task.(*SegmentTask), step)

		case *ChannelAction:
			ex.executeDmChannelAction(task.(*ChannelTask), step)
		}
	}()

	return true
}

func (ex *Executor) scheduleRequests() {
	ex.wg.Add(1)
	go func() {
		defer ex.wg.Done()
		for mergeTask := range ex.merger.Chan() {
			task := mergeTask.(*LoadSegmentsTask)
			log.Info("get merge task, process it",
				zap.Int64("collectionID", task.req.GetCollectionID()),
				zap.String("shard", task.req.GetInfos()[0].GetInsertChannel()),
				zap.Int64("nodeID", task.req.GetDstNodeID()),
				zap.Int("taskNum", len(task.tasks)),
			)
			go ex.processMergeTask(mergeTask.(*LoadSegmentsTask))
		}
	}()
}

func (ex *Executor) processMergeTask(mergeTask *LoadSegmentsTask) {
	task := mergeTask.tasks[0]
	action := task.Actions()[mergeTask.steps[0]]

	defer func() {
		for i := range mergeTask.tasks {
			mergeTask.tasks[i].SetErr(task.Err())
			ex.removeAction(mergeTask.tasks[i], mergeTask.steps[i])
		}
	}()

	taskIDs := make([]int64, 0, len(mergeTask.tasks))
	segments := make([]int64, 0, len(mergeTask.tasks))
	for _, task := range mergeTask.tasks {
		taskIDs = append(taskIDs, task.ID())
		segments = append(segments, task.SegmentID())
	}
	log := log.With(
		zap.Int64s("taskIDs", taskIDs),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64s("segmentIDs", segments),
		zap.Int64("nodeID", action.Node()),
		zap.Int64("source", task.SourceID()),
	)

	// Get shard leader for the given replica and segment
	channel := mergeTask.req.GetInfos()[0].GetInsertChannel()
	leader, ok := getShardLeader(ex.meta.ReplicaManager, ex.dist, task.CollectionID(), action.Node(), channel)
	if !ok {
		msg := "no shard leader for the segment to execute loading"
		task.SetErr(utils.WrapError(msg, ErrTaskStale))
		log.Warn(msg, zap.String("shard", channel))
		return
	}

	log.Info("load segments...")
	ctx, cancel := context.WithTimeout(task.Context(), actionTimeout)
	status, err := ex.cluster.LoadSegments(ctx, leader, mergeTask.req)
	cancel()
	if err != nil {
		log.Warn("failed to load segment, it may be a false failure", zap.Error(err))
		return
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to load segment", zap.String("reason", status.GetReason()))
		return
	}
	log.Info("load segments done")
}

func (ex *Executor) removeAction(task Task, step int) {
	if task.Err() != nil {
		log.Info("excute action done, remove it",
			zap.Int64("taskID", task.ID()),
			zap.Int("step", step),
			zap.Error(task.Err()))
	}

	index := actionIndex{
		Task: task.ID(),
		Step: step,
	}
	ex.executingActions.Delete(index)
}

func (ex *Executor) executeSegmentAction(task *SegmentTask, step int) {
	switch task.Actions()[step].Type() {
	case ActionTypeGrow:
		ex.loadSegment(task, step)

	case ActionTypeReduce:
		ex.releaseSegment(task, step)
	}
}

// loadSegment commits the request to merger,
// not really executes the request
func (ex *Executor) loadSegment(task *SegmentTask, step int) {
	action := task.Actions()[step].(*SegmentAction)
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("segmentID", task.segmentID),
		zap.Int64("node", action.Node()),
		zap.Int64("source", task.SourceID()),
	)

	shouldRemoveAction := true
	defer func() {
		if shouldRemoveAction {
			ex.removeAction(task, step)
		}
	}()

	ctx, cancel := context.WithTimeout(task.Context(), actionTimeout)
	defer cancel()

	schema, err := ex.broker.GetCollectionSchema(ctx, task.CollectionID())
	if err != nil {
		log.Warn("failed to get schema of collection", zap.Error(err))
		return
	}
	partitions, err := utils.GetPartitions(ex.meta.CollectionManager, ex.broker, task.CollectionID())
	if err != nil {
		log.Warn("failed to get partitions of collection", zap.Error(err))
		return
	}
	loadMeta := packLoadMeta(
		ex.meta.GetLoadType(task.CollectionID()),
		task.CollectionID(),
		partitions...,
	)
	segments, err := ex.broker.GetSegmentInfo(ctx, task.SegmentID())
	if err != nil || len(segments) == 0 {
		log.Warn("failed to get segment info from DataCoord", zap.Error(err))
		return
	}
	segment := segments[0]
	indexes, err := ex.broker.GetIndexInfo(ctx, task.CollectionID(), segment.GetID())
	if err != nil {
		log.Warn("failed to get index of segment", zap.Error(err))
		return
	}
	loadInfo := utils.PackSegmentLoadInfo(segment, indexes)

	// Get shard leader for the given replica and segment
	leader, ok := getShardLeader(ex.meta.ReplicaManager, ex.dist, task.CollectionID(), action.Node(), segment.GetInsertChannel())
	if !ok {
		msg := "no shard leader for the segment to execute loading"
		task.SetErr(utils.WrapError(msg, ErrTaskStale))
		log.Warn(msg, zap.String("shard", segment.GetInsertChannel()))
		return
	}
	log = log.With(zap.Int64("shardLeader", leader))

	deltaPositions, err := getSegmentDeltaPositions(ctx, ex.targetMgr, ex.broker, segment.GetCollectionID(), segment.GetPartitionID(), segment.GetInsertChannel())
	if err != nil {
		log.Warn("failed to get delta positions of segment", zap.Error(err))
		return
	}

	req := packLoadSegmentRequest(task, action, schema, loadMeta, loadInfo, deltaPositions)
	loadTask := NewLoadSegmentsTask(task, step, req)
	ex.merger.Add(loadTask)
	log.Info("load segment task committed")
	shouldRemoveAction = false
}

func (ex *Executor) releaseSegment(task *SegmentTask, step int) {
	defer ex.removeAction(task, step)

	action := task.Actions()[step].(*SegmentAction)
	defer action.isReleaseCommitted.Store(true)

	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("segmentID", task.segmentID),
		zap.Int64("node", action.Node()),
		zap.Int64("source", task.SourceID()),
	)

	ctx, cancel := context.WithTimeout(task.Context(), actionTimeout)
	defer cancel()

	dstNode := action.Node()
	req := packReleaseSegmentRequest(task, action)
	if action.Scope() == querypb.DataScope_Streaming {
		// Any modification to the segment distribution have to set NeedTransfer true,
		// to protect the version, which serves search/query
		req.NeedTransfer = true
	} else {
		var targetSegment *meta.Segment
		segments := ex.dist.SegmentDistManager.GetByNode(action.Node())
		for _, segment := range segments {
			if segment.GetID() == task.SegmentID() {
				targetSegment = segment
				break
			}
		}
		if targetSegment == nil {
			log.Info("segment to release not found in distribution")
			return
		}
		req.Shard = targetSegment.GetInsertChannel()

		if ex.meta.CollectionManager.Exist(task.CollectionID()) {
			leader, ok := getShardLeader(ex.meta.ReplicaManager, ex.dist, task.CollectionID(), action.Node(), req.GetShard())
			if !ok {
				log.Warn("no shard leader for the segment to execute releasing", zap.String("shard", req.GetShard()))
				return
			}
			dstNode = leader
			log = log.With(zap.Int64("shardLeader", leader))
			req.NeedTransfer = true
		}
	}

	log.Info("release segment...")
	status, err := ex.cluster.ReleaseSegments(ctx, dstNode, req)
	if err != nil {
		log.Warn("failed to release segment, it may be a false failure", zap.Error(err))
		return
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to release segment", zap.String("reason", status.GetReason()))
		return
	}
	log.Info("release segment done")
}

func (ex *Executor) executeDmChannelAction(task *ChannelTask, step int) {
	switch task.Actions()[step].Type() {
	case ActionTypeGrow:
		ex.subDmChannel(task, step)

	case ActionTypeReduce:
		ex.unsubDmChannel(task, step)
	}
}

func (ex *Executor) subDmChannel(task *ChannelTask, step int) {
	defer ex.removeAction(task, step)

	action := task.Actions()[step].(*ChannelAction)
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.String("channel", task.Channel()),
		zap.Int64("node", action.Node()),
		zap.Int64("source", task.SourceID()),
	)

	ctx, cancel := context.WithTimeout(task.Context(), actionTimeout)
	defer cancel()

	schema, err := ex.broker.GetCollectionSchema(ctx, task.CollectionID())
	if err != nil {
		log.Warn("failed to get schema of collection")
		return
	}
	partitions, err := utils.GetPartitions(ex.meta.CollectionManager, ex.broker, task.CollectionID())
	if err != nil {
		log.Warn("failed to get partitions of collection")
		return
	}
	loadMeta := packLoadMeta(
		ex.meta.GetLoadType(task.CollectionID()),
		task.CollectionID(),
		partitions...,
	)

	dmChannel := ex.targetMgr.GetDmChannel(action.ChannelName())
	req := packSubDmChannelRequest(task, action, schema, loadMeta, dmChannel)
	err = fillSubDmChannelRequest(ctx, req, ex.broker)
	if err != nil {
		log.Warn("failed to subscribe DmChannel, failed to fill the request with segments",
			zap.Error(err))
		return
	}
	log.Info("subscribe channel...")
	status, err := ex.cluster.WatchDmChannels(ctx, action.Node(), req)
	if err != nil {
		log.Warn("failed to subscribe DmChannel, it may be a false failure", zap.Error(err))
		return
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to subscribe DmChannel", zap.String("reason", status.GetReason()))
		return
	}
	log.Info("subscribe DmChannel done")
}

func (ex *Executor) unsubDmChannel(task *ChannelTask, step int) {
	defer ex.removeAction(task, step)

	action := task.Actions()[step].(*ChannelAction)
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.String("channel", task.Channel()),
		zap.Int64("node", action.Node()),
		zap.Int64("source", task.SourceID()),
	)

	ctx, cancel := context.WithTimeout(task.Context(), actionTimeout)
	defer cancel()

	req := packUnsubDmChannelRequest(task, action)
	status, err := ex.cluster.UnsubDmChannel(ctx, action.Node(), req)
	if err != nil {
		log.Warn("failed to unsubscribe DmChannel, it may be a false failure", zap.Error(err))
		return
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to unsubscribe DmChannel", zap.String("reason", status.GetReason()))
		return
	}
}
