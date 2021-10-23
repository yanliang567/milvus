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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func TestTask_watchDmChannelsTask(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	genWatchDMChannelsRequest := func() *querypb.WatchDmChannelsRequest {
		schema := genSimpleSegCoreSchema()
		req := &querypb.WatchDmChannelsRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_WatchDmChannels),
			CollectionID: defaultCollectionID,
			PartitionID:  defaultPartitionID,
			Schema:       schema,
		}
		return req
	}

	t.Run("test timestamp", func(t *testing.T) {
		task := watchDmChannelsTask{
			req: genWatchDMChannelsRequest(),
		}
		timestamp := Timestamp(1000)
		task.req.Base.Timestamp = timestamp
		resT := task.Timestamp()
		assert.Equal(t, timestamp, resT)
		task.req.Base = nil
		resT = task.Timestamp()
		assert.Equal(t, Timestamp(0), resT)
	})

	t.Run("test OnEnqueue", func(t *testing.T) {
		task := watchDmChannelsTask{
			req: genWatchDMChannelsRequest(),
		}
		err := task.OnEnqueue()
		assert.NoError(t, err)
		task.req.Base = nil
		err = task.OnEnqueue()
		assert.NoError(t, err)
	})

	t.Run("test execute loadCollection", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := watchDmChannelsTask{
			req:  genWatchDMChannelsRequest(),
			node: node,
		}
		task.req.Infos = []*datapb.VchannelInfo{
			{
				CollectionID: defaultCollectionID,
				ChannelName:  defaultVChannel,
			},
		}
		task.req.PartitionID = 0
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	t.Run("test execute loadPartition", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := watchDmChannelsTask{
			req:  genWatchDMChannelsRequest(),
			node: node,
		}
		task.req.Infos = []*datapb.VchannelInfo{
			{
				CollectionID: defaultCollectionID,
				ChannelName:  defaultVChannel,
			},
		}
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	t.Run("test execute loadPartition without init collection and partition", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := watchDmChannelsTask{
			req:  genWatchDMChannelsRequest(),
			node: node,
		}
		task.req.Infos = []*datapb.VchannelInfo{
			{
				CollectionID: defaultCollectionID,
				ChannelName:  defaultVChannel,
			},
		}
		task.req.CollectionID++
		task.req.PartitionID++
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	//t.Run("test execute seek error", func(t *testing.T) {
	//
	//	node, err := genSimpleQueryNode(ctx)
	//	assert.NoError(t, err)
	//
	//	task := watchDmChannelsTask{
	//		req:  genWatchDMChannelsRequest(),
	//		node: node,
	//	}
	//	task.req.Infos = []*datapb.VchannelInfo{
	//		{
	//			CollectionID: defaultCollectionID,
	//			ChannelName:  defaultVChannel,
	//			SeekPosition: &msgstream.MsgPosition{
	//				ChannelName: defaultVChannel,
	//				MsgID:       []byte{1, 2, 3},
	//				MsgGroup:    defaultSubName,
	//				Timestamp:   0,
	//			},
	//		},
	//	}
	//	err = task.Execute(ctx)
	//	assert.Error(t, err)
	//})
}

func TestTask_loadSegmentsTask(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	genLoadEmptySegmentsRequest := func() *querypb.LoadSegmentsRequest {
		schema := genSimpleSegCoreSchema()
		req := &querypb.LoadSegmentsRequest{
			Base:          genCommonMsgBase(commonpb.MsgType_LoadSegments),
			Schema:        schema,
			LoadCondition: querypb.TriggerCondition_grpcRequest,
		}
		return req
	}

	t.Run("test timestamp", func(t *testing.T) {
		task := loadSegmentsTask{
			req: genLoadEmptySegmentsRequest(),
		}
		timestamp := Timestamp(1000)
		task.req.Base.Timestamp = timestamp
		resT := task.Timestamp()
		assert.Equal(t, timestamp, resT)
		task.req.Base = nil
		resT = task.Timestamp()
		assert.Equal(t, Timestamp(0), resT)
	})

	t.Run("test OnEnqueue", func(t *testing.T) {
		task := loadSegmentsTask{
			req: genLoadEmptySegmentsRequest(),
		}
		err := task.OnEnqueue()
		assert.NoError(t, err)
		task.req.Base = nil
		err = task.OnEnqueue()
		assert.NoError(t, err)
	})

	t.Run("test execute grpc", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		schema := genSimpleInsertDataSchema()

		fieldBinlog, err := saveSimpleBinLog(ctx)
		assert.NoError(t, err)

		req := &querypb.LoadSegmentsRequest{
			Base:          genCommonMsgBase(commonpb.MsgType_LoadSegments),
			Schema:        schema,
			LoadCondition: querypb.TriggerCondition_grpcRequest,
			Infos: []*querypb.SegmentLoadInfo{
				{
					SegmentID:    defaultSegmentID,
					PartitionID:  defaultPartitionID,
					CollectionID: defaultCollectionID,
					BinlogPaths:  fieldBinlog,
				},
			},
		}

		task := loadSegmentsTask{
			req:  req,
			node: node,
		}
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	t.Run("test execute grpc error", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := loadSegmentsTask{
			req:  genLoadEmptySegmentsRequest(),
			node: node,
		}
		task.req.Infos = []*querypb.SegmentLoadInfo{
			{
				SegmentID:    defaultSegmentID + 1,
				PartitionID:  defaultPartitionID + 1,
				CollectionID: defaultCollectionID + 1,
			},
		}
		err = task.Execute(ctx)
		assert.Error(t, err)
	})

	t.Run("test execute node down", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := loadSegmentsTask{
			req:  genLoadEmptySegmentsRequest(),
			node: node,
		}
		task.req.Infos = []*querypb.SegmentLoadInfo{
			{
				SegmentID:    defaultSegmentID + 1,
				PartitionID:  defaultPartitionID + 1,
				CollectionID: defaultCollectionID + 1,
			},
		}
		task.req.LoadCondition = querypb.TriggerCondition_nodeDown
		err = task.Execute(ctx)
		assert.Error(t, err)
	})

	t.Run("test OOM", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		totalRAM := Params.CacheSize * 1024 * 1024 * 1024

		col, err := node.historical.replica.getCollectionByID(defaultCollectionID)
		assert.NoError(t, err)

		sizePerRecord, err := typeutil.EstimateSizePerRecord(col.schema)
		assert.NoError(t, err)

		task := loadSegmentsTask{
			req:  genLoadEmptySegmentsRequest(),
			node: node,
		}
		task.req.Infos = []*querypb.SegmentLoadInfo{
			{
				SegmentID:    defaultSegmentID,
				PartitionID:  defaultPartitionID,
				CollectionID: defaultCollectionID,
				NumOfRows:    totalRAM / int64(sizePerRecord),
			},
		}
		task.req.LoadCondition = querypb.TriggerCondition_handoff
		err = task.Execute(ctx)
		assert.Error(t, err)
	})
}

func TestTask_releaseCollectionTask(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	genReleaseCollectionRequest := func() *querypb.ReleaseCollectionRequest {
		req := &querypb.ReleaseCollectionRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_LoadSegments),
			CollectionID: defaultCollectionID,
		}
		return req
	}

	t.Run("test timestamp", func(t *testing.T) {
		task := releaseCollectionTask{
			req: genReleaseCollectionRequest(),
		}
		timestamp := Timestamp(1000)
		task.req.Base.Timestamp = timestamp
		resT := task.Timestamp()
		assert.Equal(t, timestamp, resT)
		task.req.Base = nil
		resT = task.Timestamp()
		assert.Equal(t, Timestamp(0), resT)
	})

	t.Run("test OnEnqueue", func(t *testing.T) {
		task := releaseCollectionTask{
			req: genReleaseCollectionRequest(),
		}
		err := task.OnEnqueue()
		assert.NoError(t, err)
		task.req.Base = nil
		err = task.OnEnqueue()
		assert.NoError(t, err)
	})

	t.Run("test execute", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := releaseCollectionTask{
			req:  genReleaseCollectionRequest(),
			node: node,
		}
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	t.Run("test execute no collection in streaming", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		err = node.streaming.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		task := releaseCollectionTask{
			req:  genReleaseCollectionRequest(),
			node: node,
		}
		err = task.Execute(ctx)
		assert.Error(t, err)
	})
}

func TestTask_releasePartitionTask(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	genReleasePartitionsRequest := func() *querypb.ReleasePartitionsRequest {
		req := &querypb.ReleasePartitionsRequest{
			Base:         genCommonMsgBase(commonpb.MsgType_LoadSegments),
			CollectionID: defaultCollectionID,
			PartitionIDs: []UniqueID{defaultPartitionID},
		}
		return req
	}

	t.Run("test timestamp", func(t *testing.T) {
		task := releasePartitionsTask{
			req: genReleasePartitionsRequest(),
		}
		timestamp := Timestamp(1000)
		task.req.Base.Timestamp = timestamp
		resT := task.Timestamp()
		assert.Equal(t, timestamp, resT)
		task.req.Base = nil
		resT = task.Timestamp()
		assert.Equal(t, Timestamp(0), resT)
	})

	t.Run("test OnEnqueue", func(t *testing.T) {
		task := releasePartitionsTask{
			req: genReleasePartitionsRequest(),
		}
		err := task.OnEnqueue()
		assert.NoError(t, err)
		task.req.Base = nil
		err = task.OnEnqueue()
		assert.NoError(t, err)
	})

	t.Run("test execute", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := releasePartitionsTask{
			req:  genReleasePartitionsRequest(),
			node: node,
		}
		task.node.streaming.dataSyncService.addPartitionFlowGraph(defaultCollectionID,
			defaultPartitionID,
			[]Channel{defaultVChannel})
		err = task.Execute(ctx)
		assert.NoError(t, err)
	})

	t.Run("test execute no collection in historical", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := releasePartitionsTask{
			req:  genReleasePartitionsRequest(),
			node: node,
		}
		err = node.historical.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		err = task.Execute(ctx)
		assert.Error(t, err)
	})

	t.Run("test execute no collection in streaming", func(t *testing.T) {
		node, err := genSimpleQueryNode(ctx)
		assert.NoError(t, err)

		task := releasePartitionsTask{
			req:  genReleasePartitionsRequest(),
			node: node,
		}
		err = node.streaming.replica.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		err = task.Execute(ctx)
		assert.Error(t, err)
	})
}
