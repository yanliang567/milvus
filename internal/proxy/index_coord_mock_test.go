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

package proxy

import (
	"context"
	"errors"
	"sync/atomic"

	"github.com/milvus-io/milvus/internal/types"

	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/uniquegenerator"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type IndexCoordMock struct {
	nodeID  typeutil.UniqueID
	address string

	state atomic.Value // internal.StateCode

	getMetricsFunc getMetricsFuncType

	statisticsChannel string
	timeTickChannel   string

	minioBucketName string
}

func (coord *IndexCoordMock) updateState(state internalpb.StateCode) {
	coord.state.Store(state)
}

func (coord *IndexCoordMock) getState() internalpb.StateCode {
	return coord.state.Load().(internalpb.StateCode)
}

func (coord *IndexCoordMock) healthy() bool {
	return coord.getState() == internalpb.StateCode_Healthy
}

func (coord *IndexCoordMock) Init() error {
	coord.updateState(internalpb.StateCode_Initializing)
	return nil
}

func (coord *IndexCoordMock) Start() error {
	defer coord.updateState(internalpb.StateCode_Healthy)

	return nil
}

func (coord *IndexCoordMock) Stop() error {
	defer coord.updateState(internalpb.StateCode_Abnormal)

	return nil
}

func (coord *IndexCoordMock) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return &internalpb.ComponentStates{
		State: &internalpb.ComponentInfo{
			NodeID:    coord.nodeID,
			Role:      typeutil.IndexCoordRole,
			StateCode: coord.getState(),
			ExtraInfo: nil,
		},
		SubcomponentStates: nil,
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
	}, nil
}

func (coord *IndexCoordMock) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: coord.statisticsChannel,
	}, nil
}

func (coord *IndexCoordMock) Register() error {
	return nil
}

func (coord *IndexCoordMock) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: coord.timeTickChannel,
	}, nil
}

func (coord *IndexCoordMock) BuildIndex(ctx context.Context, req *indexpb.BuildIndexRequest) (*indexpb.BuildIndexResponse, error) {
	return &indexpb.BuildIndexResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		IndexBuildID: 0,
	}, nil
}

func (coord *IndexCoordMock) DropIndex(ctx context.Context, req *indexpb.DropIndexRequest) (*commonpb.Status, error) {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
		Reason:    "",
	}, nil
}

func (coord *IndexCoordMock) RemoveIndex(ctx context.Context, req *indexpb.RemoveIndexRequest) (*commonpb.Status, error) {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
		Reason:    "",
	}, nil
}

func (coord *IndexCoordMock) GetIndexStates(ctx context.Context, req *indexpb.GetIndexStatesRequest) (*indexpb.GetIndexStatesResponse, error) {
	return &indexpb.GetIndexStatesResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		States: nil,
	}, nil
}

func (coord *IndexCoordMock) GetIndexFilePaths(ctx context.Context, req *indexpb.GetIndexFilePathsRequest) (*indexpb.GetIndexFilePathsResponse, error) {
	return &indexpb.GetIndexFilePathsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		FilePaths: nil,
	}, nil
}

func (coord *IndexCoordMock) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	if !coord.healthy() {
		return &milvuspb.GetMetricsResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    "unhealthy",
			},
		}, nil
	}

	if coord.getMetricsFunc != nil {
		return coord.getMetricsFunc(ctx, req)
	}

	return &milvuspb.GetMetricsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    "not implemented",
		},
		Response:      "",
		ComponentName: "",
	}, nil
}

func NewIndexCoordMock() *IndexCoordMock {
	return &IndexCoordMock{
		nodeID:            typeutil.UniqueID(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
		address:           funcutil.GenRandomStr(), // TODO(dragondriver): random address
		statisticsChannel: funcutil.GenRandomStr(),
		timeTickChannel:   funcutil.GenRandomStr(),
		minioBucketName:   funcutil.GenRandomStr(),
	}
}

type GetIndexStatesFunc func(ctx context.Context, request *indexpb.GetIndexStatesRequest) (*indexpb.GetIndexStatesResponse, error)

type mockIndexCoord struct {
	types.IndexCoord
	GetIndexStatesFunc
}

func (m *mockIndexCoord) GetIndexStates(ctx context.Context, request *indexpb.GetIndexStatesRequest) (*indexpb.GetIndexStatesResponse, error) {
	if m.GetIndexStatesFunc != nil {
		return m.GetIndexStatesFunc(ctx, request)
	}
	return nil, errors.New("mock")
}

func newMockIndexCoord() *mockIndexCoord {
	return &mockIndexCoord{}
}
