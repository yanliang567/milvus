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
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func TestImpl_GetComponentStates(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	rsp, err := node.GetComponentStates(ctx)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	rsp, err = node.GetComponentStates(ctx)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
}

func TestImpl_GetTimeTickChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	rsp, err := node.GetTimeTickChannel(ctx)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)
}

func TestImpl_GetStatisticsChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	rsp, err := node.GetStatisticsChannel(ctx)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)
}

func TestImpl_AddQueryChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	t.Run("test addQueryChannel", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})

	t.Run("test addQueryChannel has queryCollection", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		err = node.queryService.addQueryCollection(defaultCollectionID)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})

	t.Run("test node is abnormal", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		node.UpdateStateCode(internalpb.StateCode_Abnormal)
		status, err := node.AddQueryChannel(ctx, nil)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	t.Run("test nil query service", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			CollectionID: defaultCollectionID,
		}

		node.queryService = nil
		status, err := node.AddQueryChannel(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	t.Run("test add query collection failed", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		err = node.streaming.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	t.Run("test init global sealed segments", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
			GlobalSealedSegments: []*queryPb.SegmentInfo{{
				SegmentID:    defaultSegmentID,
				CollectionID: defaultCollectionID,
				PartitionID:  defaultPartitionID,
			}},
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})

	t.Run("test init global sealed segments failed", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
			GlobalSealedSegments: []*queryPb.SegmentInfo{{
				SegmentID:    defaultSegmentID,
				CollectionID: 1000,
				PartitionID:  defaultPartitionID,
			}},
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	t.Run("test seek error", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		position := &internalpb.MsgPosition{
			ChannelName: genQueryChannel(),
			MsgID:       []byte{1, 2, 3},
			MsgGroup:    defaultSubName,
			Timestamp:   0,
		}

		req := &queryPb.AddQueryChannelRequest{
			Base:             genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
			NodeID:           0,
			CollectionID:     defaultCollectionID,
			RequestChannelID: genQueryChannel(),
			ResultChannelID:  genQueryResultChannel(),
			SeekPosition:     position,
		}

		status, err := node.AddQueryChannel(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})
}

func TestImpl_RemoveQueryChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	status, err := node.RemoveQueryChannel(ctx, nil)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
}

func TestImpl_WatchDmChannels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	schema := genSimpleSegCoreSchema()

	req := &queryPb.WatchDmChannelsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_WatchQueryChannels,
			MsgID:   rand.Int63(),
		},
		NodeID:       0,
		CollectionID: defaultCollectionID,
		PartitionID:  defaultPartitionID,
		Schema:       schema,
	}

	status, err := node.WatchDmChannels(ctx, req)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	status, err = node.WatchDmChannels(ctx, req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
}

func TestImpl_LoadSegments(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	schema := genSimpleSegCoreSchema()

	req := &queryPb.LoadSegmentsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_WatchQueryChannels,
			MsgID:   rand.Int63(),
		},
		DstNodeID:     0,
		Schema:        schema,
		LoadCondition: queryPb.TriggerCondition_grpcRequest,
	}

	status, err := node.LoadSegments(ctx, req)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	status, err = node.LoadSegments(ctx, req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
}

func TestImpl_ReleaseCollection(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	req := &queryPb.ReleaseCollectionRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_WatchQueryChannels,
			MsgID:   rand.Int63(),
		},
		NodeID:       0,
		CollectionID: defaultCollectionID,
	}

	status, err := node.ReleaseCollection(ctx, req)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	status, err = node.ReleaseCollection(ctx, req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
}

func TestImpl_ReleasePartitions(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	req := &queryPb.ReleasePartitionsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_WatchQueryChannels,
			MsgID:   rand.Int63(),
		},
		NodeID:       0,
		CollectionID: defaultCollectionID,
		PartitionIDs: []UniqueID{defaultPartitionID},
	}

	status, err := node.ReleasePartitions(ctx, req)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	status, err = node.ReleasePartitions(ctx, req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
}

func TestImpl_GetSegmentInfo(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Run("test GetSegmentInfo", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		node.UpdateStateCode(internalpb.StateCode_Abnormal)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test no collection in historical", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		err = node.historical.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test no collection in streaming", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		err = node.streaming.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test different segment type", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		seg, err := node.streaming.replica.getSegmentByID(defaultSegmentID)
		assert.NoError(t, err)

		seg.setType(segmentTypeInvalid)
		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		seg.setType(segmentTypeSealed)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		seg.setType(segmentTypeGrowing)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		seg.setType(segmentTypeIndexing)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		seg.setType(-100)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)
	})

	t.Run("test GetSegmentInfo with indexed segment", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		seg, err := node.historical.replica.getSegmentByID(defaultSegmentID)
		assert.NoError(t, err)

		err = seg.setIndexInfo(simpleVecField.id, &indexInfo{
			indexName: "query-node-test",
			indexID:   UniqueID(0),
			buildID:   UniqueID(0),
		})
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, rsp.Status.ErrorCode)

		node.UpdateStateCode(internalpb.StateCode_Abnormal)
		rsp, err = node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test GetSegmentInfo without streaming partition", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		node.streaming.replica.(*collectionReplica).partitions = make(map[UniqueID]*Partition)
		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test GetSegmentInfo without streaming segment", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		node.streaming.replica.(*collectionReplica).segments = make(map[UniqueID]*Segment)
		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test GetSegmentInfo without historical partition", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		node.historical.replica.(*collectionReplica).partitions = make(map[UniqueID]*Partition)
		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})

	t.Run("test GetSegmentInfo without historical segment", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.GetSegmentInfoRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			SegmentIDs:   []UniqueID{defaultSegmentID},
			CollectionID: defaultCollectionID,
		}

		node.historical.replica.(*collectionReplica).segments = make(map[UniqueID]*Segment)
		rsp, err := node.GetSegmentInfo(ctx, req)
		assert.Error(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, rsp.Status.ErrorCode)
	})
}

func TestImpl_isHealthy(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	isHealthy := node.isHealthy()
	assert.True(t, isHealthy)
}

func TestImpl_GetMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Run("test GetMetrics", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.MetaRootPath, Params.EtcdEndpoints)

		metricReq := make(map[string]string)
		metricReq[metricsinfo.MetricTypeKey] = "system_info"
		mReq, err := json.Marshal(metricReq)
		assert.NoError(t, err)

		req := &milvuspb.GetMetricsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
			Request: string(mReq),
		}

		_, err = node.GetMetrics(ctx, req)
		assert.NoError(t, err)
	})

	t.Run("test ParseMetricType failed", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &milvuspb.GetMetricsRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_WatchQueryChannels,
				MsgID:   rand.Int63(),
			},
		}

		_, err = node.GetMetrics(ctx, req)
		assert.NoError(t, err)

		node.UpdateStateCode(internalpb.StateCode_Abnormal)
		_, err = node.GetMetrics(ctx, req)
		assert.NoError(t, err)
	})
}

func TestImpl_ReleaseSegments(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Run("test valid", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.ReleaseSegmentsRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_ReleaseSegments),
			CollectionID: defaultCollectionID,
			PartitionIDs: []UniqueID{defaultPartitionID},
			SegmentIDs:   []UniqueID{defaultSegmentID},
		}

		_, err = node.ReleaseSegments(ctx, req)
		assert.NoError(t, err)
	})

	t.Run("test invalid query node", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.ReleaseSegmentsRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_ReleaseSegments),
			CollectionID: defaultCollectionID,
			PartitionIDs: []UniqueID{defaultPartitionID},
			SegmentIDs:   []UniqueID{defaultSegmentID},
		}

		node.UpdateStateCode(internalpb.StateCode_Abnormal)
		_, err = node.ReleaseSegments(ctx, req)
		assert.Error(t, err)
	})

	t.Run("test segment not exists", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		req := &queryPb.ReleaseSegmentsRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_ReleaseSegments),
			CollectionID: defaultCollectionID,
			PartitionIDs: []UniqueID{defaultPartitionID},
			SegmentIDs:   []UniqueID{defaultSegmentID},
		}

		err = node.historical.replica.removeSegment(defaultSegmentID)
		assert.NoError(t, err)

		err = node.streaming.replica.removeSegment(defaultSegmentID)
		assert.NoError(t, err)

		status, err := node.ReleaseSegments(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})
}
