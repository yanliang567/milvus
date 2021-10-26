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

package querycoord

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func setup() {
	Params.Init()
}

func refreshParams() {
	rand.Seed(time.Now().UnixNano())
	suffix := "-test-query-Coord" + strconv.FormatInt(rand.Int63(), 10)
	Params.StatsChannelName = Params.StatsChannelName + suffix
	Params.TimeTickChannelName = Params.TimeTickChannelName + suffix
	Params.MetaRootPath = Params.MetaRootPath + suffix
}

func TestMain(m *testing.M) {
	setup()
	//refreshChannelNames()
	exitCode := m.Run()
	os.Exit(exitCode)
}

func NewQueryCoordTest(ctx context.Context, factory msgstream.Factory) (*QueryCoord, error) {
	queryCoord, err := NewQueryCoord(ctx, factory)
	if err != nil {
		return nil, err
	}
	queryCoord.newNodeFn = newQueryNodeTest
	return queryCoord, nil
}

func startQueryCoord(ctx context.Context) (*QueryCoord, error) {
	factory := msgstream.NewPmsFactory()

	coord, err := NewQueryCoordTest(ctx, factory)
	if err != nil {
		return nil, err
	}

	rootCoord := newRootCoordMock()
	rootCoord.createCollection(defaultCollectionID)
	rootCoord.createPartition(defaultCollectionID, defaultPartitionID)

	dataCoord, err := newDataCoordMock(ctx)
	if err != nil {
		return nil, err
	}

	coord.SetRootCoord(rootCoord)
	coord.SetDataCoord(dataCoord)

	err = coord.Register()
	if err != nil {
		return nil, err
	}
	err = coord.Init()
	if err != nil {
		return nil, err
	}
	err = coord.Start()
	if err != nil {
		return nil, err
	}
	return coord, nil
}

func createDefaultPartition(ctx context.Context, queryCoord *QueryCoord) error {
	_, err := queryCoord.rootCoordClient.CreatePartition(ctx, nil)
	return err
}

func startUnHealthyQueryCoord(ctx context.Context) (*QueryCoord, error) {
	factory := msgstream.NewPmsFactory()

	coord, err := NewQueryCoordTest(ctx, factory)
	if err != nil {
		return nil, err
	}

	rootCoord := newRootCoordMock()
	rootCoord.createCollection(defaultCollectionID)
	rootCoord.createPartition(defaultCollectionID, defaultPartitionID)

	dataCoord, err := newDataCoordMock(ctx)
	if err != nil {
		return nil, err
	}

	coord.SetRootCoord(rootCoord)
	coord.SetDataCoord(dataCoord)

	err = coord.Register()
	if err != nil {
		return nil, err
	}
	err = coord.Init()
	if err != nil {
		return nil, err
	}

	return coord, nil
}

func TestWatchNodeLoop(t *testing.T) {
	baseCtx := context.Background()

	t.Run("Test OfflineNodes", func(t *testing.T) {
		refreshParams()
		kv, err := etcdkv.NewEtcdKV(Params.EtcdEndpoints, Params.MetaRootPath)
		assert.Nil(t, err)

		kvs := make(map[string]string)
		session := &sessionutil.Session{
			ServerID: 100,
			Address:  "localhost",
		}
		sessionBlob, err := json.Marshal(session)
		assert.Nil(t, err)
		sessionKey := fmt.Sprintf("%s/%d", queryNodeInfoPrefix, 100)
		kvs[sessionKey] = string(sessionBlob)

		collectionInfo := &querypb.CollectionInfo{
			CollectionID: defaultCollectionID,
		}
		collectionBlobs, err := proto.Marshal(collectionInfo)
		assert.Nil(t, err)
		nodeKey := fmt.Sprintf("%s/%d", queryNodeMetaPrefix, 100)
		kvs[nodeKey] = string(collectionBlobs)

		err = kv.MultiSave(kvs)
		assert.Nil(t, err)

		queryCoord, err := startQueryCoord(baseCtx)
		assert.Nil(t, err)

		for {
			_, err = queryCoord.cluster.offlineNodes()
			if err == nil {
				break
			}
		}

		queryCoord.Stop()
		err = removeAllSession()
		assert.Nil(t, err)
	})

	t.Run("Test RegisterNewNode", func(t *testing.T) {
		refreshParams()
		queryCoord, err := startQueryCoord(baseCtx)
		assert.Nil(t, err)

		queryNode1, err := startQueryNodeServer(baseCtx)
		assert.Nil(t, err)

		nodeID := queryNode1.queryNodeID
		for {
			_, err = queryCoord.cluster.getNodeByID(nodeID)
			if err == nil {
				break
			}
		}

		queryCoord.Stop()
		queryNode1.stop()
		err = removeAllSession()
		assert.Nil(t, err)
	})

	t.Run("Test RemoveNode", func(t *testing.T) {
		refreshParams()
		queryNode1, err := startQueryNodeServer(baseCtx)
		assert.Nil(t, err)

		queryCoord, err := startQueryCoord(baseCtx)
		assert.Nil(t, err)

		nodeID := queryNode1.queryNodeID
		queryNode1.stop()
		err = removeNodeSession(nodeID)
		assert.Nil(t, err)
		for {
			_, err = queryCoord.cluster.getNodeByID(nodeID)
			if err != nil {
				break
			}
		}
		queryCoord.Stop()
		err = removeAllSession()
		assert.Nil(t, err)
	})
}

func TestHandoffSegmentLoop(t *testing.T) {
	refreshParams()
	baseCtx := context.Background()

	queryCoord, err := startQueryCoord(baseCtx)
	assert.Nil(t, err)

	queryNode1, err := startQueryNodeServer(baseCtx)
	assert.Nil(t, err)
	waitQueryNodeOnline(queryCoord.cluster, queryNode1.queryNodeID)

	t.Run("Test watchHandoffLoop", func(t *testing.T) {
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    defaultSegmentID,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID,
			SegmentState: querypb.SegmentState_sealed,
		}

		key := fmt.Sprintf("%s/%d/%d/%d", handoffSegmentPrefix, defaultCollectionID, defaultPartitionID, defaultSegmentID)
		value, err := proto.Marshal(segmentInfo)
		assert.Nil(t, err)
		err = queryCoord.kvClient.Save(key, string(value))
		assert.Nil(t, err)
		for {
			_, err := queryCoord.kvClient.Load(key)
			if err != nil {
				break
			}
		}
	})

	loadPartitionTask := genLoadPartitionTask(baseCtx, queryCoord)
	err = queryCoord.scheduler.Enqueue(loadPartitionTask)
	assert.Nil(t, err)
	waitTaskFinalState(loadPartitionTask, taskExpired)

	t.Run("Test partitionNotLoaded", func(t *testing.T) {
		baseTask := newBaseTask(baseCtx, querypb.TriggerCondition_handoff)
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    defaultSegmentID,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID + 1,
			SegmentState: querypb.SegmentState_sealed,
		}
		handoffReq := &querypb.HandoffSegmentsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_HandoffSegments,
			},
			SegmentInfos: []*querypb.SegmentInfo{segmentInfo},
		}
		handoffTask := &handoffTask{
			baseTask:               baseTask,
			HandoffSegmentsRequest: handoffReq,
			dataCoord:              queryCoord.dataCoordClient,
			cluster:                queryCoord.cluster,
			meta:                   queryCoord.meta,
		}
		err = queryCoord.scheduler.Enqueue(handoffTask)
		assert.Nil(t, err)

		waitTaskFinalState(handoffTask, taskExpired)
	})

	loadCollectionTask := genLoadCollectionTask(baseCtx, queryCoord)
	err = queryCoord.scheduler.Enqueue(loadCollectionTask)
	assert.Nil(t, err)
	waitTaskFinalState(loadCollectionTask, taskExpired)

	t.Run("Test handoffGrowingSegment", func(t *testing.T) {
		infos := queryCoord.meta.showSegmentInfos(defaultCollectionID, nil)
		assert.NotEqual(t, 0, len(infos))
		segmentID := defaultSegmentID + 4
		baseTask := newBaseTask(baseCtx, querypb.TriggerCondition_handoff)

		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    segmentID,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID + 2,
			SegmentState: querypb.SegmentState_sealed,
		}
		handoffReq := &querypb.HandoffSegmentsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_HandoffSegments,
			},
			SegmentInfos: []*querypb.SegmentInfo{segmentInfo},
		}
		handoffTask := &handoffTask{
			baseTask:               baseTask,
			HandoffSegmentsRequest: handoffReq,
			dataCoord:              queryCoord.dataCoordClient,
			cluster:                queryCoord.cluster,
			meta:                   queryCoord.meta,
		}
		err = queryCoord.scheduler.Enqueue(handoffTask)
		assert.Nil(t, err)

		waitTaskFinalState(handoffTask, taskExpired)
	})

	t.Run("Test binlogNotExist", func(t *testing.T) {
		baseTask := newBaseTask(baseCtx, querypb.TriggerCondition_handoff)
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    defaultSegmentID + 100,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID,
			SegmentState: querypb.SegmentState_sealed,
		}
		handoffReq := &querypb.HandoffSegmentsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_HandoffSegments,
			},
			SegmentInfos: []*querypb.SegmentInfo{segmentInfo},
		}
		handoffTask := &handoffTask{
			baseTask:               baseTask,
			HandoffSegmentsRequest: handoffReq,
			dataCoord:              queryCoord.dataCoordClient,
			cluster:                queryCoord.cluster,
			meta:                   queryCoord.meta,
		}
		err = queryCoord.scheduler.Enqueue(handoffTask)
		assert.Nil(t, err)

		waitTaskFinalState(handoffTask, taskFailed)
	})

	t.Run("Test sealedSegmentExist", func(t *testing.T) {
		baseTask := newBaseTask(baseCtx, querypb.TriggerCondition_handoff)
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    defaultSegmentID,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID,
			SegmentState: querypb.SegmentState_sealed,
		}
		handoffReq := &querypb.HandoffSegmentsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_HandoffSegments,
			},
			SegmentInfos: []*querypb.SegmentInfo{segmentInfo},
		}
		handoffTask := &handoffTask{
			baseTask:               baseTask,
			HandoffSegmentsRequest: handoffReq,
			dataCoord:              queryCoord.dataCoordClient,
			cluster:                queryCoord.cluster,
			meta:                   queryCoord.meta,
		}
		err = queryCoord.scheduler.Enqueue(handoffTask)
		assert.Nil(t, err)

		waitTaskFinalState(handoffTask, taskFailed)
	})

	releasePartitionTask := genReleasePartitionTask(baseCtx, queryCoord)
	err = queryCoord.scheduler.Enqueue(releasePartitionTask)
	assert.Nil(t, err)
	waitTaskFinalState(releasePartitionTask, taskExpired)

	t.Run("Test handoffReleasedPartition", func(t *testing.T) {
		baseTask := newBaseTask(baseCtx, querypb.TriggerCondition_handoff)
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    defaultSegmentID,
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID,
			SegmentState: querypb.SegmentState_sealed,
		}
		handoffReq := &querypb.HandoffSegmentsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_HandoffSegments,
			},
			SegmentInfos: []*querypb.SegmentInfo{segmentInfo},
		}
		handoffTask := &handoffTask{
			baseTask:               baseTask,
			HandoffSegmentsRequest: handoffReq,
			dataCoord:              queryCoord.dataCoordClient,
			cluster:                queryCoord.cluster,
			meta:                   queryCoord.meta,
		}
		err = queryCoord.scheduler.Enqueue(handoffTask)
		assert.Nil(t, err)

		waitTaskFinalState(handoffTask, taskExpired)
	})

	queryCoord.Stop()
	err = removeAllSession()
	assert.Nil(t, err)
}
