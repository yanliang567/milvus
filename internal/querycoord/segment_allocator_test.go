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
	"testing"

	"github.com/stretchr/testify/assert"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func TestShuffleSegmentsToQueryNode(t *testing.T) {
	refreshParams()
	baseCtx, cancel := context.WithCancel(context.Background())
	kv, err := etcdkv.NewEtcdKV(Params.EtcdEndpoints, Params.MetaRootPath)
	assert.Nil(t, err)
	clusterSession := sessionutil.NewSession(context.Background(), Params.MetaRootPath, Params.EtcdEndpoints)
	clusterSession.Init(typeutil.QueryCoordRole, Params.Address, true)
	meta, err := newMeta(baseCtx, kv, nil, nil)
	assert.Nil(t, err)
	cluster := &queryNodeCluster{
		ctx:         baseCtx,
		cancel:      cancel,
		client:      kv,
		clusterMeta: meta,
		nodes:       make(map[int64]Node),
		newNodeFn:   newQueryNodeTest,
		session:     clusterSession,
	}

	schema := genCollectionSchema(defaultCollectionID, false)
	firstReq := &querypb.LoadSegmentsRequest{
		CollectionID: defaultCollectionID,
		Schema:       schema,
		Infos: []*querypb.SegmentLoadInfo{
			{
				SegmentID:    defaultSegmentID,
				PartitionID:  defaultPartitionID,
				CollectionID: defaultCollectionID,
				NumOfRows:    defaultNumRowPerSegment,
			},
		},
	}
	secondReq := &querypb.LoadSegmentsRequest{
		CollectionID: defaultCollectionID,
		Schema:       schema,
		Infos: []*querypb.SegmentLoadInfo{
			{
				SegmentID:    defaultSegmentID + 1,
				PartitionID:  defaultPartitionID,
				CollectionID: defaultCollectionID,
				NumOfRows:    defaultNumRowPerSegment,
			},
		},
	}
	reqs := []*querypb.LoadSegmentsRequest{firstReq, secondReq}

	t.Run("Test shuffleSegmentsWithoutQueryNode", func(t *testing.T) {
		err = shuffleSegmentsToQueryNode(baseCtx, reqs, cluster, false, nil)
		assert.NotNil(t, err)
	})

	node1, err := startQueryNodeServer(baseCtx)
	assert.Nil(t, err)
	node1Session := node1.session
	node1ID := node1.queryNodeID
	cluster.registerNode(baseCtx, node1Session, node1ID, disConnect)
	waitQueryNodeOnline(cluster, node1ID)

	t.Run("Test shuffleSegmentsToQueryNode", func(t *testing.T) {
		err = shuffleSegmentsToQueryNode(baseCtx, reqs, cluster, false, nil)
		assert.Nil(t, err)

		assert.Equal(t, node1ID, firstReq.DstNodeID)
		assert.Equal(t, node1ID, secondReq.DstNodeID)
	})

	node2, err := startQueryNodeServer(baseCtx)
	assert.Nil(t, err)
	node2Session := node2.session
	node2ID := node2.queryNodeID
	cluster.registerNode(baseCtx, node2Session, node2ID, disConnect)
	waitQueryNodeOnline(cluster, node2ID)
	cluster.stopNode(node1ID)

	t.Run("Test shuffleSegmentsToQueryNodeV2", func(t *testing.T) {
		err = shuffleSegmentsToQueryNodeV2(baseCtx, reqs, cluster, false, nil)
		assert.Nil(t, err)

		assert.Equal(t, node2ID, firstReq.DstNodeID)
		assert.Equal(t, node2ID, secondReq.DstNodeID)
	})

	err = removeAllSession()
	assert.Nil(t, err)
}
