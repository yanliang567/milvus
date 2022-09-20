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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func TestGetSystemInfoMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	defer etcdCli.Close()
	node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.EtcdCfg.MetaRootPath, etcdCli)

	req := &milvuspb.GetMetricsRequest{
		Base: genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
	}
	resp, err := getSystemInfoMetrics(ctx, req, node)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())

	// test getQuotaMetricsError
	rateCol.Deregister(metricsinfo.NQPerSecond)
	resp, err = getSystemInfoMetrics(ctx, req, node)
	assert.NoError(t, err)
	assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	rateCol.Register(metricsinfo.NQPerSecond)
}

func TestGetComponentConfigurationsFailed(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	node, err := genSimpleQueryNode(ctx)
	assert.NoError(t, err)

	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	defer etcdCli.Close()
	node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.EtcdCfg.MetaRootPath, etcdCli)

	req := &internalpb.ShowConfigurationsRequest{
		Base:    genCommonMsgBase(commonpb.MsgType_WatchQueryChannels),
		Pattern: "Cache",
	}

	resq := getComponentConfigurations(ctx, req)
	assert.Equal(t, resq.Status.ErrorCode, commonpb.ErrorCode_Success)
}
