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

package grpcindexcoord

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/indexcoord"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/etcd"
)

func TestIndexCoordinateServer(t *testing.T) {
	ctx := context.Background()
	factory := dependency.NewDefaultFactory(true)
	server, err := NewServer(ctx, factory)
	assert.Nil(t, err)
	assert.NotNil(t, server)
	Params.Init()
	etcd, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	assert.NotNil(t, etcd)
	indexCoordClient := &indexcoord.Mock{}
	indexCoordClient.SetEtcdClient(etcd)
	err = server.SetClient(indexCoordClient)
	assert.Nil(t, err)

	dcm := &indexcoord.DataCoordMock{}
	server.dataCoord = dcm
	err = server.Run()
	assert.Nil(t, err)

	t.Run("GetComponentStates", func(t *testing.T) {
		req := &internalpb.GetComponentStatesRequest{}
		states, err := server.GetComponentStates(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, internalpb.StateCode_Healthy, states.State.StateCode)
	})

	t.Run("GetTimeTickChannel", func(t *testing.T) {
		req := &internalpb.GetTimeTickChannelRequest{}
		resp, err := server.GetTimeTickChannel(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetStatisticsChannel", func(t *testing.T) {
		req := &internalpb.GetStatisticsChannelRequest{}
		resp, err := server.GetStatisticsChannel(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("BuildIndex", func(t *testing.T) {
		req := &indexpb.BuildIndexRequest{
			IndexBuildID: 0,
			IndexID:      0,
			DataPaths:    []string{},
		}
		resp, err := server.BuildIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetIndexStates", func(t *testing.T) {
		req := &indexpb.GetIndexStatesRequest{
			IndexBuildIDs: []UniqueID{0},
		}
		resp, err := server.GetIndexStates(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, len(req.IndexBuildIDs), len(resp.States))
		assert.Equal(t, commonpb.IndexState_Finished, resp.States[0].State)
	})

	t.Run("DropIndex", func(t *testing.T) {
		req := &indexpb.DropIndexRequest{
			IndexID: 0,
		}
		resp, err := server.DropIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	})

	t.Run("RemoveIndex", func(t *testing.T) {
		req := &indexpb.RemoveIndexRequest{
			BuildIDs: []UniqueID{0},
		}
		resp, err := server.RemoveIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	})

	t.Run("GetIndexFilePaths", func(t *testing.T) {
		req := &indexpb.GetIndexFilePathsRequest{
			IndexBuildIDs: []UniqueID{0, 1},
		}
		resp, err := server.GetIndexFilePaths(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, len(req.IndexBuildIDs), len(resp.FilePaths))
	})

	t.Run("GetMetrics", func(t *testing.T) {
		req := &milvuspb.GetMetricsRequest{
			Request: "",
		}
		resp, err := server.GetMetrics(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, "IndexCoord", resp.ComponentName)
	})

	err = server.Stop()
	assert.Nil(t, err)
}
