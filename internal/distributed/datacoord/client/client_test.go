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

package grpcdatacoordclient

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/internal/proxy"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/mock"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
)

func Test_NewClient(t *testing.T) {
	proxy.Params.InitOnce()

	ctx := context.Background()
	etcdCli, err := etcd.GetEtcdClient(&proxy.Params.EtcdCfg)
	assert.Nil(t, err)
	client, err := NewClient(ctx, proxy.Params.EtcdCfg.MetaRootPath, etcdCli)
	assert.Nil(t, err)
	assert.NotNil(t, client)

	err = client.Init()
	assert.Nil(t, err)

	err = client.Start()
	assert.Nil(t, err)

	err = client.Register()
	assert.Nil(t, err)

	checkFunc := func(retNotNil bool) {
		retCheck := func(notNil bool, ret interface{}, err error) {
			if notNil {
				assert.NotNil(t, ret)
				assert.Nil(t, err)
			} else {
				assert.Nil(t, ret)
				assert.NotNil(t, err)
			}
		}

		r1, err := client.GetComponentStates(ctx)
		retCheck(retNotNil, r1, err)

		r2, err := client.GetTimeTickChannel(ctx)
		retCheck(retNotNil, r2, err)

		r3, err := client.GetStatisticsChannel(ctx)
		retCheck(retNotNil, r3, err)

		r4, err := client.Flush(ctx, nil)
		retCheck(retNotNil, r4, err)

		r5, err := client.AssignSegmentID(ctx, nil)
		retCheck(retNotNil, r5, err)

		r6, err := client.GetSegmentInfo(ctx, nil)
		retCheck(retNotNil, r6, err)

		r7, err := client.GetSegmentStates(ctx, nil)
		retCheck(retNotNil, r7, err)

		r8, err := client.GetInsertBinlogPaths(ctx, nil)
		retCheck(retNotNil, r8, err)

		r9, err := client.GetCollectionStatistics(ctx, nil)
		retCheck(retNotNil, r9, err)

		r10, err := client.GetPartitionStatistics(ctx, nil)
		retCheck(retNotNil, r10, err)

		r11, err := client.GetSegmentInfoChannel(ctx)
		retCheck(retNotNil, r11, err)

		// r12, err := client.SaveBinlogPaths(ctx, nil)
		// retCheck(retNotNil, r12, err)

		r13, err := client.GetRecoveryInfo(ctx, nil)
		retCheck(retNotNil, r13, err)

		r14, err := client.GetFlushedSegments(ctx, nil)
		retCheck(retNotNil, r14, err)

		r15, err := client.GetMetrics(ctx, nil)
		retCheck(retNotNil, r15, err)

		r17, err := client.GetCompactionState(ctx, nil)
		retCheck(retNotNil, r17, err)

		r18, err := client.ManualCompaction(ctx, nil)
		retCheck(retNotNil, r18, err)

		r19, err := client.GetCompactionStateWithPlans(ctx, nil)
		retCheck(retNotNil, r19, err)

		r20, err := client.WatchChannels(ctx, nil)
		retCheck(retNotNil, r20, err)

		r21, err := client.DropVirtualChannel(ctx, nil)
		retCheck(retNotNil, r21, err)

		r22, err := client.SetSegmentState(ctx, nil)
		retCheck(retNotNil, r22, err)

		r23, err := client.Import(ctx, nil)
		retCheck(retNotNil, r23, err)

		r24, err := client.UpdateSegmentStatistics(ctx, nil)
		retCheck(retNotNil, r24, err)

		r25, err := client.AcquireSegmentLock(ctx, nil)
		retCheck(retNotNil, r25, err)

		r26, err := client.ReleaseSegmentLock(ctx, nil)
		retCheck(retNotNil, r26, err)

		r27, err := client.SaveImportSegment(ctx, nil)
		retCheck(retNotNil, r27, err)

		r29, err := client.UnsetIsImportingState(ctx, nil)
		retCheck(retNotNil, r29, err)

		r30, err := client.MarkSegmentsDropped(ctx, nil)
		retCheck(retNotNil, r30, err)

		r31, err := client.ShowConfigurations(ctx, nil)
		retCheck(retNotNil, r31, err)
	}

	client.grpcClient = &mock.GRPCClientBase{
		GetGrpcClientErr: errors.New("dummy"),
	}

	newFunc1 := func(cc *grpc.ClientConn) interface{} {
		return &mock.GrpcDataCoordClient{Err: nil}
	}
	client.grpcClient.SetNewGrpcClientFunc(newFunc1)

	checkFunc(false)

	// special case since this method didn't use recall()
	ret, err := client.SaveBinlogPaths(ctx, nil)
	assert.Nil(t, ret)
	assert.NotNil(t, err)

	client.grpcClient = &mock.GRPCClientBase{
		GetGrpcClientErr: nil,
	}
	newFunc2 := func(cc *grpc.ClientConn) interface{} {
		return &mock.GrpcDataCoordClient{Err: errors.New("dummy")}
	}
	client.grpcClient.SetNewGrpcClientFunc(newFunc2)
	checkFunc(false)

	// special case since this method didn't use recall()
	ret, err = client.SaveBinlogPaths(ctx, nil)
	assert.Nil(t, ret)
	assert.NotNil(t, err)

	client.grpcClient = &mock.GRPCClientBase{
		GetGrpcClientErr: nil,
	}
	newFunc3 := func(cc *grpc.ClientConn) interface{} {
		return &mock.GrpcDataCoordClient{Err: nil}
	}
	client.grpcClient.SetNewGrpcClientFunc(newFunc3)
	checkFunc(true)

	// special case since this method didn't use recall()
	ret, err = client.SaveBinlogPaths(ctx, nil)
	assert.NotNil(t, ret)
	assert.Nil(t, err)

	err = client.Stop()
	assert.Nil(t, err)
}
