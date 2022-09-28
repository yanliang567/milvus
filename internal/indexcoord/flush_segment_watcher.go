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

package indexcoord

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/util"
)

type flushedSegmentWatcher struct {
	ctx    context.Context
	cancel context.CancelFunc

	kvClient         kv.MetaKv
	wg               sync.WaitGroup
	scheduleDuration time.Duration

	internalTaskMutex sync.RWMutex
	internalNotify    chan struct{}

	etcdRevision int64
	watchChan    clientv3.WatchChan

	meta    *metaTable
	builder *indexBuilder
	ic      *IndexCoord
	handoff *handoff

	internalTasks map[UniqueID]*internalTask
}

type internalTask struct {
	state       indexTaskState
	segmentInfo *datapb.SegmentInfo
}

func newFlushSegmentWatcher(ctx context.Context, kv kv.MetaKv, meta *metaTable, builder *indexBuilder,
	handoff *handoff, ic *IndexCoord) (*flushedSegmentWatcher, error) {
	ctx, cancel := context.WithCancel(ctx)
	fsw := &flushedSegmentWatcher{
		ctx:               ctx,
		cancel:            cancel,
		kvClient:          kv,
		wg:                sync.WaitGroup{},
		internalTaskMutex: sync.RWMutex{},
		scheduleDuration:  time.Second * 3,
		internalNotify:    make(chan struct{}, 1),
		meta:              meta,
		builder:           builder,
		handoff:           handoff,
		ic:                ic,
	}
	err := fsw.reloadFromKV()
	if err != nil {
		return nil, err
	}
	return fsw, nil
}

func (fsw *flushedSegmentWatcher) reloadFromKV() error {
	log.Ctx(fsw.ctx).Info("flushSegmentWatcher reloadFromKV")
	fsw.internalTasks = make(map[UniqueID]*internalTask)
	_, values, version, err := fsw.kvClient.LoadWithRevision(util.FlushedSegmentPrefix)
	if err != nil {
		log.Ctx(fsw.ctx).Error("flushSegmentWatcher reloadFromKV fail", zap.String("prefix", util.FlushedSegmentPrefix), zap.Error(err))
		return err
	}
	for _, value := range values {
		segID, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			log.Ctx(fsw.ctx).Error("flushSegmentWatcher parse segmentID fail", zap.String("value", value), zap.Error(err))
			return err
		}
		fsw.enqueueInternalTask(segID)
	}
	fsw.etcdRevision = version
	log.Ctx(fsw.ctx).Info("flushSegmentWatcher reloadFromKV success", zap.Int64("etcdRevision", version))
	return nil
}

func (fsw *flushedSegmentWatcher) Start() {
	fsw.wg.Add(1)
	go fsw.internalScheduler()
}

func (fsw *flushedSegmentWatcher) Stop() {
	fsw.cancel()
	fsw.wg.Wait()
}

func (fsw *flushedSegmentWatcher) enqueueInternalTask(segmentID UniqueID) {
	defer fsw.internalNotifyFunc()
	fsw.internalTaskMutex.Lock()
	defer fsw.internalTaskMutex.Unlock()
	if _, ok := fsw.internalTasks[segmentID]; !ok {
		fsw.internalTasks[segmentID] = &internalTask{
			state:       indexTaskPrepare,
			segmentInfo: nil,
		}
	}
}

func (fsw *flushedSegmentWatcher) internalScheduler() {
	log.Ctx(fsw.ctx).Info("IndexCoord flushedSegmentWatcher internalScheduler start...")
	defer fsw.wg.Done()

	ticker := time.NewTicker(fsw.scheduleDuration)
	defer ticker.Stop()

	for {
		select {
		case <-fsw.ctx.Done():
			log.Ctx(fsw.ctx).Warn("IndexCoord flushedSegmentWatcher context done")
			return
		case <-ticker.C:
			fsw.internalRun()
		case <-fsw.internalNotify:
			fsw.internalRun()
		}
	}
}

func (fsw *flushedSegmentWatcher) internalRun() {
	fsw.internalTaskMutex.RLock()
	segmentIDs := make([]UniqueID, 0, len(fsw.internalTasks))
	if len(fsw.internalTasks) > 0 {
		for segID := range fsw.internalTasks {
			segmentIDs = append(segmentIDs, segID)
		}
		sort.Slice(segmentIDs, func(i, j int) bool {
			return segmentIDs[i] < segmentIDs[j]
		})
	}
	fsw.internalTaskMutex.RUnlock()

	for _, segmentID := range segmentIDs {
		fsw.internalProcess(segmentID)
	}
}

func (fsw *flushedSegmentWatcher) internalNotifyFunc() {
	select {
	case fsw.internalNotify <- struct{}{}:
	default:
	}
}

func (fsw *flushedSegmentWatcher) Len() int {
	fsw.internalTaskMutex.RLock()
	defer fsw.internalTaskMutex.RUnlock()

	return len(fsw.internalTasks)
}

func (fsw *flushedSegmentWatcher) updateInternalTaskState(segID UniqueID, state indexTaskState) {
	fsw.internalTaskMutex.Lock()
	defer fsw.internalTaskMutex.Unlock()
	log.Ctx(fsw.ctx).Debug("flushedSegmentWatcher updateInternalTaskState", zap.Int64("segID", segID),
		zap.String("state", state.String()))
	if _, ok := fsw.internalTasks[segID]; ok {
		fsw.internalTasks[segID].state = state
	}
}

func (fsw *flushedSegmentWatcher) deleteInternalTask(segID UniqueID) {
	fsw.internalTaskMutex.Lock()
	defer fsw.internalTaskMutex.Unlock()

	delete(fsw.internalTasks, segID)
	log.Ctx(fsw.ctx).Debug("flushedSegmentWatcher delete the internal task", zap.Int64("segID", segID))
}

func (fsw *flushedSegmentWatcher) getInternalTask(segID UniqueID) *internalTask {
	fsw.internalTaskMutex.RLock()
	defer fsw.internalTaskMutex.RUnlock()

	return &internalTask{
		state:       fsw.internalTasks[segID].state,
		segmentInfo: fsw.internalTasks[segID].segmentInfo,
	}
}

func (fsw *flushedSegmentWatcher) setInternalTaskSegmentInfo(segID UniqueID, segInfo *datapb.SegmentInfo) {
	fsw.internalTaskMutex.Lock()
	defer fsw.internalTaskMutex.Unlock()

	if _, ok := fsw.internalTasks[segID]; ok {
		fsw.internalTasks[segID].segmentInfo = segInfo
	}
	log.Ctx(fsw.ctx).Debug("flushedSegmentWatcher set internal task segment info success", zap.Int64("segID", segID))
}

func (fsw *flushedSegmentWatcher) internalProcess(segID UniqueID) {
	t := fsw.getInternalTask(segID)
	log.Ctx(fsw.ctx).RatedDebug(10, "flushedSegmentWatcher process internal task", zap.Int64("segID", segID),
		zap.String("state", t.state.String()))

	switch t.state {
	case indexTaskPrepare:
		if err := fsw.prepare(segID); err != nil {
			log.Ctx(fsw.ctx).RatedWarn(10, "flushedSegmentWatcher prepare internal task fail", zap.Int64("segID", segID), zap.Error(err))
			return
		}
		fsw.updateInternalTaskState(segID, indexTaskInit)
	case indexTaskInit:
		if err := fsw.constructTask(t); err != nil {
			log.Ctx(fsw.ctx).RatedWarn(10, "flushedSegmentWatcher construct task fail", zap.Int64("segID", segID), zap.Error(err))
			return
		}
		fsw.updateInternalTaskState(segID, indexTaskInProgress)
		fsw.internalNotifyFunc()
	case indexTaskInProgress:
		if fsw.handoff.taskDone(segID) {
			fsw.updateInternalTaskState(segID, indexTaskDone)
			fsw.internalNotifyFunc()
		}
	case indexTaskDone:
		if err := fsw.removeFlushedSegment(t); err != nil {
			log.Ctx(fsw.ctx).RatedWarn(10, "IndexCoord flushSegmentWatcher removeFlushedSegment fail",
				zap.Int64("segID", segID), zap.Error(err))
			return
		}
		fsw.deleteInternalTask(segID)
		fsw.internalNotifyFunc()
	default:
		log.Info("IndexCoord flushedSegmentWatcher internal task get invalid state", zap.Int64("segID", segID),
			zap.String("state", t.state.String()))
	}
}

func (fsw *flushedSegmentWatcher) constructTask(t *internalTask) error {
	fieldIndexes := fsw.meta.GetIndexesForCollection(t.segmentInfo.CollectionID, "")
	if len(fieldIndexes) == 0 {
		log.Ctx(fsw.ctx).Debug("segment no need to build index", zap.Int64("segmentID", t.segmentInfo.ID),
			zap.Int64("num of rows", t.segmentInfo.NumOfRows), zap.Int("collection indexes num", len(fieldIndexes)))
		// no need to build index
		return nil
	}

	for _, index := range fieldIndexes {
		segIdx := &model.SegmentIndex{
			SegmentID:    t.segmentInfo.ID,
			CollectionID: t.segmentInfo.CollectionID,
			PartitionID:  t.segmentInfo.PartitionID,
			NumRows:      t.segmentInfo.NumOfRows,
			IndexID:      index.IndexID,
			CreateTime:   t.segmentInfo.StartPosition.Timestamp,
		}

		//create index task for metaTable
		// send to indexBuilder
		have, buildID, err := fsw.ic.createIndexForSegment(segIdx)
		if err != nil {
			log.Ctx(fsw.ctx).Warn("IndexCoord create index for segment fail", zap.Int64("segID", t.segmentInfo.ID),
				zap.Int64("indexID", index.IndexID), zap.Error(err))
			return err
		}
		if !have {
			fsw.builder.enqueue(buildID)
		}
	}
	fsw.handoff.enqueue(t.segmentInfo.ID)
	log.Ctx(fsw.ctx).Debug("flushedSegmentWatcher construct task success", zap.Int64("segID", t.segmentInfo.ID),
		zap.Int("tasks num", len(fieldIndexes)))
	return nil
}

func (fsw *flushedSegmentWatcher) removeFlushedSegment(t *internalTask) error {
	deletedKeys := fmt.Sprintf("%s/%d/%d/%d", util.FlushedSegmentPrefix, t.segmentInfo.CollectionID, t.segmentInfo.PartitionID, t.segmentInfo.ID)
	err := fsw.kvClient.RemoveWithPrefix(deletedKeys)
	if err != nil {
		log.Ctx(fsw.ctx).Warn("IndexCoord remove flushed segment fail", zap.Int64("collID", t.segmentInfo.CollectionID),
			zap.Int64("partID", t.segmentInfo.PartitionID), zap.Int64("segID", t.segmentInfo.ID), zap.Error(err))
		return err
	}
	log.Ctx(fsw.ctx).Info("IndexCoord remove flushed segment success", zap.Int64("collID", t.segmentInfo.CollectionID),
		zap.Int64("partID", t.segmentInfo.PartitionID), zap.Int64("segID", t.segmentInfo.ID))
	return nil
}

func (fsw *flushedSegmentWatcher) prepare(segID UniqueID) error {
	defer fsw.internalNotifyFunc()
	log.Debug("prepare flushed segment task", zap.Int64("segID", segID))
	t := fsw.getInternalTask(segID)
	if t.segmentInfo != nil {
		return nil
	}
	info, err := fsw.ic.pullSegmentInfo(fsw.ctx, segID)
	if err != nil {
		log.Error("flushedSegmentWatcher get segment info fail", zap.Int64("segID", segID),
			zap.Error(err))
		if errors.Is(err, ErrSegmentNotFound) {
			fsw.deleteInternalTask(segID)
			return err
		}
		return err
	}
	fsw.setInternalTaskSegmentInfo(segID, info)
	return nil
}
