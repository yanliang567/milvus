package datacoord

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"go.uber.org/zap"
)

// TODO this num should be determined by resources of datanode, for now, we set to a fixed value for simple
const (
	maxParallelCompactionTaskNum      = 100
	compactionTimeout                 = 10 * time.Second
	compactionExpirationCheckInterval = 60 * time.Second
)

type compactionPlanContext interface {
	start()
	stop()
	// execCompactionPlan start to execute plan and return immediately
	execCompactionPlan(plan *datapb.CompactionPlan) error
	// completeCompaction record the result of a compaction
	completeCompaction(result *datapb.CompactionResult) error
	// getCompaction return compaction task. If planId does not exist, return nil.
	getCompaction(planID int64) *compactionTask
	// expireCompaction set the compaction state to expired
	expireCompaction(ts Timestamp) error
	// isFull return true if the task pool is full
	isFull() bool
	// get compaction by signal id and return the number of executing/completed/timeout plans
	getCompactionBySignalID(signalID int64) (executing, completed, timeout int)
}

type compactionTaskState int8

const (
	executing compactionTaskState = iota + 1
	completed
	timeout
)

var (
	errChannelNotWatched = errors.New("channel is not watched")
	errChannelInBuffer   = errors.New("channel is in buffer")
)

type compactionTask struct {
	triggerInfo *compactionSignal
	plan        *datapb.CompactionPlan
	state       compactionTaskState
	dataNodeID  int64
}

func (t *compactionTask) shadowClone(opts ...compactionTaskOpt) *compactionTask {
	task := &compactionTask{
		plan:       t.plan,
		state:      t.state,
		dataNodeID: t.dataNodeID,
	}
	for _, opt := range opts {
		opt(task)
	}
	return task
}

var _ compactionPlanContext = (*compactionPlanHandler)(nil)

type compactionPlanHandler struct {
	plans            map[int64]*compactionTask // planid -> task
	sessions         *SessionManager
	meta             *meta
	chManager        *ChannelManager
	mu               sync.RWMutex
	executingTaskNum int
	allocator        allocator
	quit             chan struct{}
	wg               sync.WaitGroup
	flushCh          chan UniqueID
}

func newCompactionPlanHandler(sessions *SessionManager, cm *ChannelManager, meta *meta,
	allocator allocator, flush chan UniqueID) *compactionPlanHandler {
	return &compactionPlanHandler{
		plans:     make(map[int64]*compactionTask),
		chManager: cm,
		meta:      meta,
		sessions:  sessions,
		allocator: allocator,
		flushCh:   flush,
	}
}

func (c *compactionPlanHandler) start() {
	ticker := time.NewTicker(compactionExpirationCheckInterval)
	c.quit = make(chan struct{})
	c.wg.Add(1)

	go func() {
		defer c.wg.Done()
		for {
			select {
			case <-c.quit:
				ticker.Stop()
				log.Info("compaction handler quit")
				return
			case <-ticker.C:
				cctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				ts, err := c.allocator.allocTimestamp(cctx)
				if err != nil {
					log.Warn("unable to alloc timestamp", zap.Error(err))
					cancel()
					continue
				}
				cancel()
				_ = c.expireCompaction(ts)
			}
		}
	}()
}

func (c *compactionPlanHandler) stop() {
	close(c.quit)
	c.wg.Wait()
}

// execCompactionPlan start to execute plan and return immediately
func (c *compactionPlanHandler) execCompactionPlan(plan *datapb.CompactionPlan) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	nodeID, err := c.chManager.FindWatcher(plan.GetChannel())
	if err != nil {
		return err
	}

	c.setSegmentsCompacting(plan, true)

	// FIXME: check response of compaction call and restore segment state if failed
	c.sessions.Compaction(nodeID, plan)

	task := &compactionTask{
		plan:       plan,
		state:      executing,
		dataNodeID: nodeID,
	}
	c.plans[plan.PlanID] = task
	c.executingTaskNum++
	return nil
}

func (c *compactionPlanHandler) setSegmentsCompacting(plan *datapb.CompactionPlan, compacting bool) {
	for _, segmentBinlogs := range plan.GetSegmentBinlogs() {
		c.meta.SetSegmentCompacting(segmentBinlogs.GetSegmentID(), compacting)
	}
}

// completeCompaction record the result of a compaction
func (c *compactionPlanHandler) completeCompaction(result *datapb.CompactionResult) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	planID := result.PlanID
	if _, ok := c.plans[planID]; !ok {
		return fmt.Errorf("plan %d is not found", planID)
	}

	if c.plans[planID].state != executing {
		return fmt.Errorf("plan %d's state is %v", planID, c.plans[planID].state)
	}

	plan := c.plans[planID].plan
	switch plan.GetType() {
	case datapb.CompactionType_InnerCompaction:
		if err := c.handleInnerCompactionResult(plan, result); err != nil {
			return err
		}
	case datapb.CompactionType_MergeCompaction:
		if err := c.handleMergeCompactionResult(plan, result); err != nil {
			return err
		}
	default:
		return errors.New("unknown compaction type")
	}
	c.plans[planID] = c.plans[planID].shadowClone(setState(completed))
	c.executingTaskNum--
	if c.plans[planID].plan.GetType() == datapb.CompactionType_MergeCompaction {
		c.flushCh <- result.GetSegmentID()
	}
	// TODO: when to clean task list

	return nil
}

func (c *compactionPlanHandler) handleInnerCompactionResult(plan *datapb.CompactionPlan, result *datapb.CompactionResult) error {
	return c.meta.CompleteInnerCompaction(plan.GetSegmentBinlogs()[0], result)
}

func (c *compactionPlanHandler) handleMergeCompactionResult(plan *datapb.CompactionPlan, result *datapb.CompactionResult) error {
	return c.meta.CompleteMergeCompaction(plan.GetSegmentBinlogs(), result)
}

// getCompaction return compaction task. If planId does not exist, return nil.
func (c *compactionPlanHandler) getCompaction(planID int64) *compactionTask {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.plans[planID]
}

// expireCompaction set the compaction state to expired
func (c *compactionPlanHandler) expireCompaction(ts Timestamp) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	tasks := c.getExecutingCompactions()
	for _, task := range tasks {
		if !c.isTimeout(ts, task.plan.GetStartTime(), task.plan.GetTimeoutInSeconds()) {
			continue
		}

		c.setSegmentsCompacting(task.plan, false)

		planID := task.plan.PlanID
		c.plans[planID] = c.plans[planID].shadowClone(setState(timeout))
		c.executingTaskNum--
	}

	return nil
}

func (c *compactionPlanHandler) isTimeout(now Timestamp, start Timestamp, timeout int32) bool {
	starttime, _ := tsoutil.ParseTS(start)
	ts, _ := tsoutil.ParseTS(now)
	return int32(ts.Sub(starttime).Seconds()) >= timeout
}

// isFull return true if the task pool is full
func (c *compactionPlanHandler) isFull() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.executingTaskNum >= maxParallelCompactionTaskNum
}

func (c *compactionPlanHandler) getExecutingCompactions() []*compactionTask {
	tasks := make([]*compactionTask, 0, len(c.plans))
	for _, plan := range c.plans {
		if plan.state == executing {
			tasks = append(tasks, plan)
		}
	}
	return tasks
}

// get compaction by signal id and return the number of executing/completed/timeout plans
func (c *compactionPlanHandler) getCompactionBySignalID(signalID int64) (executingPlans int, completedPlans int, timeoutPlans int) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, t := range c.plans {
		if t.triggerInfo.id != signalID {
			continue
		}
		switch t.state {
		case executing:
			executingPlans++
		case completed:
			completedPlans++
		case timeout:
			timeoutPlans++
		}
	}
	return
}

type compactionTaskOpt func(task *compactionTask)

func setState(state compactionTaskState) compactionTaskOpt {
	return func(task *compactionTask) {
		task.state = state
	}
}
