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

package task

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	. "github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/atomic"
	"go.uber.org/zap"
)

const (
	TaskTypeGrow Type = iota + 1
	TaskTypeReduce
	TaskTypeMove
)

var (
	ErrConflictTaskExisted = errors.New("ConflictTaskExisted")

	// The task is canceled or timeout
	ErrTaskCanceled = errors.New("TaskCanceled")

	// The target node is offline,
	// or the target segment is not in TargetManager,
	// or the target channel is not in TargetManager
	ErrTaskStale = errors.New("TaskStale")

	// ErrInsufficientMemory returns insufficient memory error.
	ErrInsufficientMemory = errors.New("InsufficientMemoryToLoad")

	ErrFailedResponse  = errors.New("RpcFailed")
	ErrTaskAlreadyDone = errors.New("TaskAlreadyDone")
)

type Type = int32

type replicaSegmentIndex struct {
	ReplicaID int64
	SegmentID int64
	IsGrowing bool
}

func NewReplicaSegmentIndex(task *SegmentTask) replicaSegmentIndex {
	isGrowing := task.Actions()[0].(*SegmentAction).Scope() == querypb.DataScope_Streaming
	return replicaSegmentIndex{
		ReplicaID: task.ReplicaID(),
		SegmentID: task.SegmentID(),
		IsGrowing: isGrowing,
	}
}

type replicaChannelIndex struct {
	ReplicaID int64
	Channel   string
}

type taskQueue struct {
	// TaskPriority -> TaskID -> Task
	buckets []map[int64]Task
}

func newTaskQueue() *taskQueue {
	buckets := make([]map[int64]Task, len(TaskPriorities))
	for i := range buckets {
		buckets[i] = make(map[int64]Task)
	}
	return &taskQueue{
		buckets: buckets,
	}
}

func (queue *taskQueue) Len() int {
	taskNum := 0
	for _, tasks := range queue.buckets {
		taskNum += len(tasks)
	}

	return taskNum
}

func (queue *taskQueue) Add(task Task) {
	bucket := queue.buckets[task.Priority()]
	bucket[task.ID()] = task
}

func (queue *taskQueue) Remove(task Task) {
	bucket := queue.buckets[task.Priority()]
	delete(bucket, task.ID())
}

// Range iterates all tasks in the queue ordered by priority from high to low
func (queue *taskQueue) Range(fn func(task Task) bool) {
	for priority := len(queue.buckets) - 1; priority >= 0; priority-- {
		for _, task := range queue.buckets[priority] {
			if !fn(task) {
				return
			}
		}
	}
}

type Scheduler interface {
	Start(ctx context.Context)
	Stop()
	AddExecutor(nodeID int64)
	RemoveExecutor(nodeID int64)
	Add(task Task) error
	Dispatch(node int64)
	RemoveByNode(node int64)
	GetNodeSegmentDelta(nodeID int64) int
	GetNodeChannelDelta(nodeID int64) int
}

type taskScheduler struct {
	rwmutex     sync.RWMutex
	ctx         context.Context
	executors   map[int64]*Executor // NodeID -> Executor
	idAllocator func() UniqueID

	distMgr   *meta.DistributionManager
	meta      *meta.Meta
	targetMgr *meta.TargetManager
	broker    meta.Broker
	cluster   session.Cluster
	nodeMgr   *session.NodeManager

	tasks        UniqueSet
	segmentTasks map[replicaSegmentIndex]Task
	channelTasks map[replicaChannelIndex]Task
	processQueue *taskQueue
	waitQueue    *taskQueue
}

func NewScheduler(ctx context.Context,
	meta *meta.Meta,
	distMgr *meta.DistributionManager,
	targetMgr *meta.TargetManager,
	broker meta.Broker,
	cluster session.Cluster,
	nodeMgr *session.NodeManager) *taskScheduler {
	id := int64(0)
	return &taskScheduler{
		ctx:       ctx,
		executors: make(map[int64]*Executor),
		idAllocator: func() UniqueID {
			id++
			return id
		},

		distMgr:   distMgr,
		meta:      meta,
		targetMgr: targetMgr,
		broker:    broker,
		cluster:   cluster,
		nodeMgr:   nodeMgr,

		tasks:        make(UniqueSet),
		segmentTasks: make(map[replicaSegmentIndex]Task),
		channelTasks: make(map[replicaChannelIndex]Task),
		processQueue: newTaskQueue(),
		waitQueue:    newTaskQueue(),
	}
}

func (scheduler *taskScheduler) Start(ctx context.Context) {}

func (scheduler *taskScheduler) Stop() {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	for nodeID, executor := range scheduler.executors {
		executor.Stop()
		delete(scheduler.executors, nodeID)
	}
}

func (scheduler *taskScheduler) AddExecutor(nodeID int64) {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	if _, exist := scheduler.executors[nodeID]; exist {
		return
	}

	executor := NewExecutor(scheduler.meta,
		scheduler.distMgr,
		scheduler.broker,
		scheduler.targetMgr,
		scheduler.cluster,
		scheduler.nodeMgr)

	scheduler.executors[nodeID] = executor
	executor.Start(scheduler.ctx)
	log.Info("add executor for new QueryNode", zap.Int64("nodeID", nodeID))
}

func (scheduler *taskScheduler) RemoveExecutor(nodeID int64) {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	executor, ok := scheduler.executors[nodeID]
	if ok {
		executor.Stop()
		delete(scheduler.executors, nodeID)
		log.Info("remove executor of offline QueryNode", zap.Int64("nodeID", nodeID))
	}
}

func (scheduler *taskScheduler) Add(task Task) error {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	err := scheduler.preAdd(task)
	if err != nil {
		return err
	}

	task.SetID(scheduler.idAllocator())
	scheduler.waitQueue.Add(task)
	scheduler.tasks.Insert(task.ID())
	switch task := task.(type) {
	case *SegmentTask:
		index := NewReplicaSegmentIndex(task)
		scheduler.segmentTasks[index] = task

	case *ChannelTask:
		index := replicaChannelIndex{task.ReplicaID(), task.Channel()}
		scheduler.channelTasks[index] = task
	}

	metrics.QueryCoordTaskNum.WithLabelValues().Set(float64(scheduler.tasks.Len()))
	log.Info("task added", zap.String("task", task.String()))
	return nil
}

// check checks whether the task is valid to add,
// must hold lock
func (scheduler *taskScheduler) preAdd(task Task) error {
	switch task := task.(type) {
	case *SegmentTask:
		index := NewReplicaSegmentIndex(task)
		if old, ok := scheduler.segmentTasks[index]; ok {
			if task.Priority() > old.Priority() {
				log.Info("replace old task, the new one with higher priority",
					zap.Int64("oldID", old.ID()),
					zap.Int32("oldPriority", old.Priority()),
					zap.Int64("newID", task.ID()),
					zap.Int32("newPriority", task.Priority()),
				)
				old.SetStatus(TaskStatusCanceled)
				old.SetErr(utils.WrapError("replaced with the other one with higher priority", ErrTaskCanceled))
				scheduler.remove(old)
				return nil
			}

			return ErrConflictTaskExisted
		}
		if GetTaskType(task) == TaskTypeGrow {
			nodesWithSegment := scheduler.distMgr.LeaderViewManager.GetSealedSegmentDist(task.SegmentID())
			replicaNodeMap := utils.GroupNodesByReplica(scheduler.meta.ReplicaManager, task.CollectionID(), nodesWithSegment)
			if _, ok := replicaNodeMap[task.ReplicaID()]; ok {
				return ErrTaskAlreadyDone
			}
		}

	case *ChannelTask:
		index := replicaChannelIndex{task.ReplicaID(), task.Channel()}
		if old, ok := scheduler.channelTasks[index]; ok {
			if task.Priority() > old.Priority() {
				log.Info("replace old task, the new one with higher priority",
					zap.Int64("oldID", old.ID()),
					zap.Int32("oldPriority", old.Priority()),
					zap.Int64("newID", task.ID()),
					zap.Int32("newPriority", task.Priority()),
				)
				old.SetStatus(TaskStatusCanceled)
				old.SetErr(utils.WrapError("replaced with the other one with higher priority", ErrTaskCanceled))
				scheduler.remove(old)
				return nil
			}

			return ErrConflictTaskExisted
		}

		if GetTaskType(task) == TaskTypeGrow {
			nodesWithChannel := scheduler.distMgr.LeaderViewManager.GetChannelDist(task.Channel())
			replicaNodeMap := utils.GroupNodesByReplica(scheduler.meta.ReplicaManager, task.CollectionID(), nodesWithChannel)
			if _, ok := replicaNodeMap[task.ReplicaID()]; ok {
				return ErrTaskAlreadyDone
			}
		}

	default:
		panic(fmt.Sprintf("preAdd: forget to process task type: %+v", task))
	}

	return nil
}

func (scheduler *taskScheduler) promote(task Task) error {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)
	err := scheduler.prePromote(task)
	if err != nil {
		log.Info("failed to promote task", zap.Error(err))
		return err
	}

	scheduler.processQueue.Add(task)
	task.SetStatus(TaskStatusStarted)
	return nil
}

func (scheduler *taskScheduler) tryPromoteAll() {
	// Promote waiting tasks
	toPromote := make([]Task, 0, scheduler.waitQueue.Len())
	toRemove := make([]Task, 0)
	scheduler.waitQueue.Range(func(task Task) bool {
		err := scheduler.promote(task)

		if err != nil {
			task.SetStatus(TaskStatusCanceled)
			if errors.Is(err, ErrTaskStale) { // Task canceled or stale
				task.SetStatus(TaskStatusStale)
			}

			log.Warn("failed to promote task",
				zap.Int64("taskID", task.ID()),
				zap.Error(err),
			)
			task.SetErr(err)
			toRemove = append(toRemove, task)
		} else {
			toPromote = append(toPromote, task)
		}

		return true
	})

	for _, task := range toPromote {
		scheduler.waitQueue.Remove(task)
	}
	for _, task := range toRemove {
		scheduler.remove(task)
	}

	if len(toPromote) > 0 || len(toRemove) > 0 {
		log.Debug("promoted tasks",
			zap.Int("promotedNum", len(toPromote)),
			zap.Int("toRemoveNum", len(toRemove)))
	}
}

func (scheduler *taskScheduler) prePromote(task Task) error {
	if scheduler.checkCanceled(task) {
		return ErrTaskCanceled
	} else if scheduler.checkStale(task) {
		return ErrTaskStale
	}

	return nil
}

func (scheduler *taskScheduler) Dispatch(node int64) {
	select {
	case <-scheduler.ctx.Done():
		log.Info("scheduler stopped")

	default:
		scheduler.schedule(node)
	}
}

func (scheduler *taskScheduler) GetNodeSegmentDelta(nodeID int64) int {
	scheduler.rwmutex.RLock()
	defer scheduler.rwmutex.RUnlock()

	return calculateNodeDelta(nodeID, scheduler.segmentTasks)
}

func (scheduler *taskScheduler) GetNodeChannelDelta(nodeID int64) int {
	scheduler.rwmutex.RLock()
	defer scheduler.rwmutex.RUnlock()

	return calculateNodeDelta(nodeID, scheduler.channelTasks)
}

func calculateNodeDelta[K comparable, T ~map[K]Task](nodeID int64, tasks T) int {
	delta := 0
	for _, task := range tasks {
		for _, action := range task.Actions() {
			if action.Node() != nodeID {
				continue
			}
			if action.Type() == ActionTypeGrow {
				delta++
			} else if action.Type() == ActionTypeReduce {
				delta--
			}
		}
	}
	return delta
}

func (scheduler *taskScheduler) GetNodeSegmentCntDelta(nodeID int64) int {
	scheduler.rwmutex.RLock()
	defer scheduler.rwmutex.RUnlock()

	delta := 0
	for _, task := range scheduler.segmentTasks {
		for _, action := range task.Actions() {
			if action.Node() != nodeID {
				continue
			}
			segmentAction := action.(*SegmentAction)
			segment := scheduler.targetMgr.GetHistoricalSegment(task.CollectionID(), segmentAction.SegmentID(), meta.NextTarget)
			if action.Type() == ActionTypeGrow {
				delta += int(segment.GetNumOfRows())
			} else {
				delta -= int(segment.GetNumOfRows())
			}
		}
	}
	return delta
}

// schedule selects some tasks to execute, follow these steps for each started selected tasks:
// 1. check whether this task is stale, set status to failed if stale
// 2. step up the task's actions, set status to succeeded if all actions finished
// 3. execute the current action of task
func (scheduler *taskScheduler) schedule(node int64) {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	if scheduler.tasks.Len() == 0 {
		return
	}

	log := log.With(
		zap.Int64("nodeID", node),
	)

	scheduler.tryPromoteAll()

	log.Debug("process tasks related to node",
		zap.Int("processingTaskNum", scheduler.processQueue.Len()),
		zap.Int("waitingTaskNum", scheduler.waitQueue.Len()),
		zap.Int("segmentTaskNum", len(scheduler.segmentTasks)),
		zap.Int("channelTaskNum", len(scheduler.channelTasks)),
	)

	// Process tasks
	toProcess := make([]Task, 0)
	toRemove := make([]Task, 0)
	scheduler.processQueue.Range(func(task Task) bool {
		if scheduler.isRelated(task, node) && scheduler.preProcess(task) {
			toProcess = append(toProcess, task)
		}
		if task.Status() != TaskStatusStarted {
			toRemove = append(toRemove, task)
		}

		return true
	})

	// The scheduler doesn't limit the number of tasks,
	// to commit tasks to executors as soon as possible, to reach higher merge possibility
	failCount := atomic.NewInt32(0)
	funcutil.ProcessFuncParallel(len(toProcess), runtime.GOMAXPROCS(0), func(idx int) error {
		if !scheduler.process(toProcess[idx]) {
			failCount.Inc()
		}
		return nil
	}, "process")

	for _, task := range toRemove {
		scheduler.remove(task)
	}

	log.Info("processed tasks",
		zap.Int("toProcessNum", len(toProcess)),
		zap.Int32("failCount", failCount.Load()),
		zap.Int("toRemoveNum", len(toRemove)),
	)

	log.Debug("process tasks related to node done",
		zap.Int("processingTaskNum", scheduler.processQueue.Len()),
		zap.Int("waitingTaskNum", scheduler.waitQueue.Len()),
		zap.Int("segmentTaskNum", len(scheduler.segmentTasks)),
		zap.Int("channelTaskNum", len(scheduler.channelTasks)),
	)
}

func (scheduler *taskScheduler) isRelated(task Task, node int64) bool {
	for _, action := range task.Actions() {
		if action.Node() == node {
			return true
		}
		if task, ok := task.(*SegmentTask); ok {
			taskType := GetTaskType(task)
			var segment *datapb.SegmentInfo
			if taskType == TaskTypeMove {
				segment = scheduler.targetMgr.GetHistoricalSegment(task.CollectionID(), task.SegmentID(), meta.CurrentTarget)
			} else {
				segment = scheduler.targetMgr.GetHistoricalSegment(task.CollectionID(), task.SegmentID(), meta.NextTarget)
			}
			if segment == nil {
				continue
			}
			replica := scheduler.meta.ReplicaManager.GetByCollectionAndNode(task.CollectionID(), action.Node())
			if replica == nil {
				continue
			}
			leader, ok := scheduler.distMgr.GetShardLeader(replica, segment.GetInsertChannel())
			if !ok {
				continue
			}
			if leader == node {
				return true
			}
		}
	}
	return false
}

// preProcess checks the finished actions of task,
// and converts the task's status,
// return true if the task should be executed,
// false otherwise
func (scheduler *taskScheduler) preProcess(task Task) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int32("type", GetTaskType(task)),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)

	actions, step := task.Actions(), task.Step()
	for step < len(actions) && actions[step].IsFinished(scheduler.distMgr) {
		task.StepUp()
		step++
	}

	if step == len(actions) {
		step--
	}

	executor, ok := scheduler.executors[actions[step].Node()]
	if !ok {
		log.Warn("no executor for QueryNode",
			zap.Int("step", step),
			zap.Int64("nodeID", actions[step].Node()))
		return false
	}

	if task.IsFinished(scheduler.distMgr) {
		if !executor.Exist(task.ID()) {
			task.SetStatus(TaskStatusSucceeded)
		}
		return false
	} else if scheduler.checkCanceled(task) {
		task.SetStatus(TaskStatusCanceled)
		if task.Err() == nil {
			task.SetErr(ErrTaskCanceled)
		}
	} else if scheduler.checkStale(task) {
		task.SetStatus(TaskStatusStale)
		task.SetErr(ErrTaskStale)
	}

	return task.Status() == TaskStatusStarted
}

// process processes the given task,
// return true if the task is started and succeeds to commit the current action
func (scheduler *taskScheduler) process(task Task) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int32("type", GetTaskType(task)),
		zap.Int64("source", task.SourceID()),
	)

	actions, step := task.Actions(), task.Step()
	executor, ok := scheduler.executors[actions[step].Node()]
	if !ok {
		log.Warn("no executor for QueryNode",
			zap.Int("step", step),
			zap.Int64("nodeID", actions[step].Node()))
		return false
	}

	log = log.With(zap.Int("step", step))
	switch task.Status() {
	case TaskStatusStarted:
		return executor.Execute(task, step)

	case TaskStatusSucceeded:
		log.Info("task succeeded")

	case TaskStatusCanceled, TaskStatusStale:
		log.Warn("failed to execute task", zap.Error(task.Err()))

	default:
		panic(fmt.Sprintf("invalid task status: %v", task.Status()))
	}

	return false
}

func (scheduler *taskScheduler) RemoveByNode(node int64) {
	scheduler.rwmutex.Lock()
	defer scheduler.rwmutex.Unlock()

	for _, task := range scheduler.segmentTasks {
		if scheduler.isRelated(task, node) {
			scheduler.remove(task)
		}
	}
	for _, task := range scheduler.channelTasks {
		if scheduler.isRelated(task, node) {
			scheduler.remove(task)
		}
	}
}

func (scheduler *taskScheduler) recordSegmentTaskError(task *SegmentTask) {
	var errCode commonpb.ErrorCode
	if errors.Is(task.Err(), ErrInsufficientMemory) {
		errCode = commonpb.ErrorCode_InsufficientMemoryToLoad
	} else {
		errCode = commonpb.ErrorCode_UnexpectedError
	}
	meta.GlobalFailedLoadCache.Put(task.collectionID, errCode, task.Err())
}

func (scheduler *taskScheduler) remove(task Task) {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int32("taskStatus", task.Status()),
	)
	task.Cancel()
	scheduler.tasks.Remove(task.ID())
	scheduler.waitQueue.Remove(task)
	scheduler.processQueue.Remove(task)

	switch task := task.(type) {
	case *SegmentTask:
		index := NewReplicaSegmentIndex(task)
		delete(scheduler.segmentTasks, index)
		log = log.With(zap.Int64("segmentID", task.SegmentID()))
		if task.Err() != nil {
			log.Warn("task scheduler recordSegmentTaskError", zap.Error(task.err))
			scheduler.recordSegmentTaskError(task)
		}

	case *ChannelTask:
		index := replicaChannelIndex{task.ReplicaID(), task.Channel()}
		delete(scheduler.channelTasks, index)
		log = log.With(zap.String("channel", task.Channel()))
	}

	metrics.QueryCoordTaskNum.WithLabelValues().Set(float64(scheduler.tasks.Len()))
	log.Debug("task removed")
}

func (scheduler *taskScheduler) checkCanceled(task Task) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)

	select {
	case <-task.Context().Done():
		log.Warn("the task is timeout or canceled")
		return true

	default:
		return false
	}
}

func (scheduler *taskScheduler) checkStale(task Task) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)

	switch task := task.(type) {
	case *SegmentTask:
		if scheduler.checkSegmentTaskStale(task) {
			return true
		}

	case *ChannelTask:
		if scheduler.checkChannelTaskStale(task) {
			return true
		}

	default:
		panic(fmt.Sprintf("checkStale: forget to check task type: %+v", task))
	}

	for step, action := range task.Actions() {
		log := log.With(
			zap.Int64("nodeID", action.Node()),
			zap.Int("step", step))

		if scheduler.nodeMgr.Get(action.Node()) == nil {
			log.Warn("the task is stale, the target node is offline")
			return true
		}
	}

	return false
}

func (scheduler *taskScheduler) checkSegmentTaskStale(task *SegmentTask) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)

	for _, action := range task.Actions() {
		switch action.Type() {
		case ActionTypeGrow:
			taskType := GetTaskType(task)
			var segment *datapb.SegmentInfo
			if taskType == TaskTypeMove {
				segment = scheduler.targetMgr.GetHistoricalSegment(task.CollectionID(), task.SegmentID(), meta.CurrentTarget)
			} else {
				segment = scheduler.targetMgr.GetHistoricalSegment(task.CollectionID(), task.SegmentID(), meta.NextTarget)
			}
			if segment == nil {
				log.Warn("task stale due to the segment to load not exists in targets",
					zap.Int64("segment", task.segmentID),
					zap.Int32("taskType", taskType),
				)
				return true
			}

			replica := scheduler.meta.ReplicaManager.GetByCollectionAndNode(task.CollectionID(), action.Node())
			if replica == nil {
				log.Warn("task stale due to replica not found")
				return true
			}
			_, ok := scheduler.distMgr.GetShardLeader(replica, segment.GetInsertChannel())
			if !ok {
				log.Warn("task stale due to leader not found")
				return true
			}

		case ActionTypeReduce:
			// Do nothing here,
			// the task should succeeded if the segment not exists
		}
	}
	return false
}

func (scheduler *taskScheduler) checkChannelTaskStale(task *ChannelTask) bool {
	log := log.With(
		zap.Int64("taskID", task.ID()),
		zap.Int64("collectionID", task.CollectionID()),
		zap.Int64("replicaID", task.ReplicaID()),
		zap.Int64("source", task.SourceID()),
	)

	for _, action := range task.Actions() {
		switch action.Type() {
		case ActionTypeGrow:
			if scheduler.targetMgr.GetDmChannel(task.collectionID, task.Channel(), meta.NextTarget) == nil {
				log.Warn("the task is stale, the channel to subscribe not exists in targets",
					zap.String("channel", task.Channel()))
				return true
			}

		case ActionTypeReduce:
			// Do nothing here,
			// the task should succeeded if the channel not exists
		}
	}
	return false
}
