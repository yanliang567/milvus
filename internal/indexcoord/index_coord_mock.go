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
	"math/rand"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// Mock is an alternative to IndexCoord, it will return specific results based on specific parameters.
type Mock struct {
	types.IndexCoord

	CallInit                 func() error
	CallStart                func() error
	CallStop                 func() error
	CallGetComponentStates   func(ctx context.Context) (*internalpb.ComponentStates, error)
	CallGetStatisticsChannel func(ctx context.Context) (*milvuspb.StringResponse, error)
	CallRegister             func() error

	CallSetEtcdClient   func(etcdClient *clientv3.Client)
	CallSetDataCoord    func(dataCoord types.DataCoord) error
	CallSetRootCoord    func(rootCoord types.RootCoord) error
	CallUpdateStateCode func(stateCode internalpb.StateCode)

	CallCreateIndex           func(ctx context.Context, req *indexpb.CreateIndexRequest) (*commonpb.Status, error)
	CallGetIndexState         func(ctx context.Context, req *indexpb.GetIndexStateRequest) (*indexpb.GetIndexStateResponse, error)
	CallGetSegmentIndexState  func(ctx context.Context, req *indexpb.GetSegmentIndexStateRequest) (*indexpb.GetSegmentIndexStateResponse, error)
	CallGetIndexInfos         func(ctx context.Context, req *indexpb.GetIndexInfoRequest) (*indexpb.GetIndexInfoResponse, error)
	CallDescribeIndex         func(ctx context.Context, req *indexpb.DescribeIndexRequest) (*indexpb.DescribeIndexResponse, error)
	CallGetIndexBuildProgress func(ctx context.Context, req *indexpb.GetIndexBuildProgressRequest) (*indexpb.GetIndexBuildProgressResponse, error)
	CallDropIndex             func(ctx context.Context, req *indexpb.DropIndexRequest) (*commonpb.Status, error)
	CallShowConfigurations    func(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error)
	CallGetMetrics            func(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error)
}

// Init initializes the Mock of IndexCoord. When param `Failure` is true, it will return an error.
func (m *Mock) Init() error {
	return m.CallInit()
}

// Start starts the Mock of IndexCoord. When param `Failure` is true, it will return an error.
func (m *Mock) Start() error {
	return m.CallStart()
}

// Stop stops the Mock of IndexCoord. When param `Failure` is true, it will return an error.
func (m *Mock) Stop() error {
	return m.CallStop()
}

// Register registers an IndexCoord role in ETCD, if Param `Failure` is true, it will return an error.
func (m *Mock) Register() error {
	return m.CallRegister()
}

func (m *Mock) SetEtcdClient(client *clientv3.Client) {
	m.CallSetEtcdClient(client)
}

func (m *Mock) SetDataCoord(dataCoord types.DataCoord) error {
	return m.CallSetDataCoord(dataCoord)
}

func (m *Mock) SetRootCoord(rootCoord types.RootCoord) error {
	return m.CallSetRootCoord(rootCoord)
}

func (m *Mock) UpdateStateCode(stateCode internalpb.StateCode) {
	m.CallUpdateStateCode(stateCode)
}

// GetComponentStates gets the component states of the mocked IndexCoord.
func (m *Mock) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return m.CallGetComponentStates(ctx)
}

// GetStatisticsChannel gets the statistics channel of the mocked IndexCoord, if Param `Failure` is true, it will return an error.
func (m *Mock) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return m.CallGetStatisticsChannel(ctx)
}

func (m *Mock) CreateIndex(ctx context.Context, req *indexpb.CreateIndexRequest) (*commonpb.Status, error) {
	return m.CallCreateIndex(ctx, req)
}

func (m *Mock) GetIndexState(ctx context.Context, req *indexpb.GetIndexStateRequest) (*indexpb.GetIndexStateResponse, error) {
	return m.CallGetIndexState(ctx, req)
}

func (m *Mock) GetSegmentIndexState(ctx context.Context, req *indexpb.GetSegmentIndexStateRequest) (*indexpb.GetSegmentIndexStateResponse, error) {
	return m.CallGetSegmentIndexState(ctx, req)
}

func (m *Mock) GetIndexInfos(ctx context.Context, req *indexpb.GetIndexInfoRequest) (*indexpb.GetIndexInfoResponse, error) {
	return m.CallGetIndexInfos(ctx, req)
}

func (m *Mock) DescribeIndex(ctx context.Context, req *indexpb.DescribeIndexRequest) (*indexpb.DescribeIndexResponse, error) {
	return m.CallDescribeIndex(ctx, req)
}

func (m *Mock) GetIndexBuildProgress(ctx context.Context, req *indexpb.GetIndexBuildProgressRequest) (*indexpb.GetIndexBuildProgressResponse, error) {
	return m.CallGetIndexBuildProgress(ctx, req)
}

func (m *Mock) DropIndex(ctx context.Context, req *indexpb.DropIndexRequest) (*commonpb.Status, error) {
	return m.CallDropIndex(ctx, req)
}

func (m *Mock) ShowConfigurations(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error) {
	return m.CallShowConfigurations(ctx, req)
}

func (m *Mock) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	return m.CallGetMetrics(ctx, req)
}

func NewIndexCoordMock() *Mock {
	return &Mock{
		CallInit: func() error {
			return nil
		},
		CallStart: func() error {
			return nil
		},
		CallRegister: func() error {
			return nil
		},
		CallStop: func() error {
			return nil
		},
		CallSetEtcdClient: func(etcdClient *clientv3.Client) {
		},
		CallSetDataCoord: func(dataCoord types.DataCoord) error {
			return nil
		},
		CallSetRootCoord: func(rootCoord types.RootCoord) error {
			return nil
		},
		CallGetComponentStates: func(ctx context.Context) (*internalpb.ComponentStates, error) {
			return &internalpb.ComponentStates{
				State: &internalpb.ComponentInfo{
					NodeID:    1,
					Role:      typeutil.IndexCoordRole,
					StateCode: internalpb.StateCode_Healthy,
				},
				SubcomponentStates: nil,
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallGetStatisticsChannel: func(ctx context.Context) (*milvuspb.StringResponse, error) {
			return &milvuspb.StringResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallCreateIndex: func(ctx context.Context, req *indexpb.CreateIndexRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}, nil
		},
		CallGetIndexState: func(ctx context.Context, req *indexpb.GetIndexStateRequest) (*indexpb.GetIndexStateResponse, error) {
			return &indexpb.GetIndexStateResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				State: commonpb.IndexState_Finished,
			}, nil
		},
		CallGetSegmentIndexState: func(ctx context.Context, req *indexpb.GetSegmentIndexStateRequest) (*indexpb.GetSegmentIndexStateResponse, error) {
			segmentStates := make([]*indexpb.SegmentIndexState, 0)
			for _, segID := range req.SegmentIDs {
				segmentStates = append(segmentStates, &indexpb.SegmentIndexState{
					SegmentID: segID,
					State:     commonpb.IndexState_Finished,
				})
			}
			return &indexpb.GetSegmentIndexStateResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				States: segmentStates,
			}, nil
		},
		CallGetIndexInfos: func(ctx context.Context, req *indexpb.GetIndexInfoRequest) (*indexpb.GetIndexInfoResponse, error) {
			segmentInfos := make(map[int64]*indexpb.SegmentInfo)
			filePaths := make([]*indexpb.IndexFilePathInfo, 0)
			for _, segID := range req.SegmentIDs {
				filePaths = append(filePaths, &indexpb.IndexFilePathInfo{
					SegmentID:      segID,
					IndexName:      "default",
					IndexFilePaths: []string{"file1", "file2"},
				})
				segmentInfos[segID] = &indexpb.SegmentInfo{
					CollectionID: req.CollectionID,
					SegmentID:    segID,
					EnableIndex:  true,
					IndexInfos:   filePaths,
				}
			}
			return &indexpb.GetIndexInfoResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				SegmentInfo: segmentInfos,
			}, nil
		},
		CallDescribeIndex: func(ctx context.Context, req *indexpb.DescribeIndexRequest) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{
						CollectionID: 1,
						FieldID:      0,
						IndexName:    "default",
						IndexID:      0,
						TypeParams:   nil,
						IndexParams:  nil,
					},
				},
			}, nil
		},
		CallGetIndexBuildProgress: func(ctx context.Context, req *indexpb.GetIndexBuildProgressRequest) (*indexpb.GetIndexBuildProgressResponse, error) {
			return &indexpb.GetIndexBuildProgressResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexedRows: 10240,
				TotalRows:   10240,
			}, nil
		},
		CallDropIndex: func(ctx context.Context, req *indexpb.DropIndexRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}, nil
		},
		CallShowConfigurations: func(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error) {
			return &internalpb.ShowConfigurationsResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallGetMetrics: func(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
			return &milvuspb.GetMetricsResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				ComponentName: typeutil.IndexCoordRole,
			}, nil
		},
	}
}

type RootCoordMock struct {
	types.RootCoord

	CallInit               func() error
	CallStart              func() error
	CallGetComponentStates func(ctx context.Context) (*internalpb.ComponentStates, error)

	CallAllocID func(ctx context.Context, req *rootcoordpb.AllocIDRequest) (*rootcoordpb.AllocIDResponse, error)
}

func (rcm *RootCoordMock) Init() error {
	return rcm.CallInit()
}

func (rcm *RootCoordMock) Start() error {
	return rcm.CallStart()
}

func (rcm *RootCoordMock) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return rcm.CallGetComponentStates(ctx)
}

func (rcm *RootCoordMock) AllocID(ctx context.Context, req *rootcoordpb.AllocIDRequest) (*rootcoordpb.AllocIDResponse, error) {
	return rcm.CallAllocID(ctx, req)
}

func NewRootCoordMock() *RootCoordMock {
	return &RootCoordMock{
		CallInit: func() error {
			return nil
		},
		CallStart: func() error {
			return nil
		},
		CallGetComponentStates: func(ctx context.Context) (*internalpb.ComponentStates, error) {
			return &internalpb.ComponentStates{
				State: &internalpb.ComponentInfo{
					NodeID:    1,
					Role:      typeutil.IndexCoordRole,
					StateCode: internalpb.StateCode_Healthy,
				},
				SubcomponentStates: nil,
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallAllocID: func(ctx context.Context, req *rootcoordpb.AllocIDRequest) (*rootcoordpb.AllocIDResponse, error) {
			return &rootcoordpb.AllocIDResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				ID:    rand.Int63(),
				Count: req.Count,
			}, nil
		},
	}
}

type DataCoordMock struct {
	types.DataCoord

	CallInit               func() error
	CallStart              func() error
	CallGetComponentStates func(ctx context.Context) (*internalpb.ComponentStates, error)

	CallGetSegmentInfo     func(ctx context.Context, req *datapb.GetSegmentInfoRequest) (*datapb.GetSegmentInfoResponse, error)
	CallGetFlushedSegment  func(ctx context.Context, req *datapb.GetFlushedSegmentsRequest) (*datapb.GetFlushedSegmentsResponse, error)
	CallAcquireSegmentLock func(ctx context.Context, req *datapb.AcquireSegmentLockRequest) (*commonpb.Status, error)
	CallReleaseSegmentLock func(ctx context.Context, req *datapb.ReleaseSegmentLockRequest) (*commonpb.Status, error)
}

func (dcm *DataCoordMock) Init() error {
	return dcm.CallInit()
}

func (dcm *DataCoordMock) Start() error {
	return dcm.CallStart()
}

func (dcm *DataCoordMock) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return dcm.CallGetComponentStates(ctx)
}

func (dcm *DataCoordMock) GetSegmentInfo(ctx context.Context, req *datapb.GetSegmentInfoRequest) (*datapb.GetSegmentInfoResponse, error) {
	return dcm.CallGetSegmentInfo(ctx, req)
}
func (dcm *DataCoordMock) AcquireSegmentLock(ctx context.Context, req *datapb.AcquireSegmentLockRequest) (*commonpb.Status, error) {
	return dcm.CallAcquireSegmentLock(ctx, req)
}

func (dcm *DataCoordMock) ReleaseSegmentLock(ctx context.Context, req *datapb.ReleaseSegmentLockRequest) (*commonpb.Status, error) {
	return dcm.CallReleaseSegmentLock(ctx, req)
}

func (dcm *DataCoordMock) GetFlushedSegments(ctx context.Context, req *datapb.GetFlushedSegmentsRequest) (*datapb.GetFlushedSegmentsResponse, error) {
	return dcm.CallGetFlushedSegment(ctx, req)
}

func NewDataCoordMock() *DataCoordMock {
	return &DataCoordMock{
		CallInit: func() error {
			return nil
		},
		CallStart: func() error {
			return nil
		},
		CallGetComponentStates: func(ctx context.Context) (*internalpb.ComponentStates, error) {
			return &internalpb.ComponentStates{
				State: &internalpb.ComponentInfo{
					NodeID:    1,
					Role:      typeutil.IndexCoordRole,
					StateCode: internalpb.StateCode_Healthy,
				},
				SubcomponentStates: nil,
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallGetSegmentInfo: func(ctx context.Context, req *datapb.GetSegmentInfoRequest) (*datapb.GetSegmentInfoResponse, error) {
			segInfos := make([]*datapb.SegmentInfo, 0)
			for _, segID := range req.SegmentIDs {
				segInfos = append(segInfos, &datapb.SegmentInfo{
					ID:             segID,
					CollectionID:   100,
					PartitionID:    200,
					InsertChannel:  "",
					NumOfRows:      1026,
					State:          commonpb.SegmentState_Flushed,
					MaxRowNum:      0,
					LastExpireTime: 0,
					StartPosition:  nil,
					DmlPosition:    nil,
					Binlogs: []*datapb.FieldBinlog{
						{
							Binlogs: []*datapb.Binlog{
								{
									LogPath: "file1",
								},
								{
									LogPath: "file2",
								},
							},
						},
					},
					Statslogs:           nil,
					Deltalogs:           nil,
					CreatedByCompaction: false,
					CompactionFrom:      nil,
					DroppedAt:           0,
				})
			}
			return &datapb.GetSegmentInfoResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				Infos: segInfos,
			}, nil
		},
		CallGetFlushedSegment: func(ctx context.Context, req *datapb.GetFlushedSegmentsRequest) (*datapb.GetFlushedSegmentsResponse, error) {
			return &datapb.GetFlushedSegmentsResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, nil
		},
		CallAcquireSegmentLock: func(ctx context.Context, req *datapb.AcquireSegmentLockRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}, nil
		},
		CallReleaseSegmentLock: func(ctx context.Context, req *datapb.ReleaseSegmentLockRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}, nil
		},
	}
}

type mockETCDKV struct {
	kv.MetaKv

	save                        func(string, string) error
	remove                      func(string) error
	multiSave                   func(map[string]string) error
	watchWithRevision           func(string, int64) clientv3.WatchChan
	loadWithRevisionAndVersions func(string) ([]string, []string, []int64, int64, error)
	compareVersionAndSwap       func(key string, version int64, target string, opts ...clientv3.OpOption) (bool, error)
	loadWithPrefix2             func(key string) ([]string, []string, []int64, error)
	loadWithPrefix              func(key string) ([]string, []string, error)
	loadWithRevision            func(key string) ([]string, []string, int64, error)
	removeWithPrefix            func(key string) error
}

func NewMockEtcdKV() *mockETCDKV {
	return &mockETCDKV{
		save: func(s string, s2 string) error {
			return nil
		},
		remove: func(s string) error {
			return nil
		},
		multiSave: func(m map[string]string) error {
			return nil
		},
		loadWithRevisionAndVersions: func(s string) ([]string, []string, []int64, int64, error) {
			return []string{}, []string{}, []int64{}, 0, nil
		},
		compareVersionAndSwap: func(key string, version int64, target string, opts ...clientv3.OpOption) (bool, error) {
			return true, nil
		},
		loadWithPrefix2: func(key string) ([]string, []string, []int64, error) {
			return []string{}, []string{}, []int64{}, nil
		},
		loadWithRevision: func(key string) ([]string, []string, int64, error) {
			return []string{}, []string{}, 0, nil
		},
		removeWithPrefix: func(key string) error {
			return nil
		},
	}
}

func (mk *mockETCDKV) Save(key string, value string) error {
	return mk.save(key, value)
}

func (mk *mockETCDKV) Remove(key string) error {
	return mk.remove(key)
}

func (mk *mockETCDKV) MultiSave(kvs map[string]string) error {
	return mk.multiSave(kvs)
}

func (mk *mockETCDKV) LoadWithRevisionAndVersions(prefix string) ([]string, []string, []int64, int64, error) {
	return mk.loadWithRevisionAndVersions(prefix)
}

func (mk *mockETCDKV) CompareVersionAndSwap(key string, version int64, target string, opts ...clientv3.OpOption) (bool, error) {
	return mk.compareVersionAndSwap(key, version, target, opts...)
}

func (mk *mockETCDKV) LoadWithPrefix(key string) ([]string, []string, error) {
	return mk.loadWithPrefix(key)
}

func (mk *mockETCDKV) LoadWithPrefix2(key string) ([]string, []string, []int64, error) {
	return mk.loadWithPrefix2(key)
}

func (mk *mockETCDKV) WatchWithRevision(key string, revision int64) clientv3.WatchChan {
	return mk.watchWithRevision(key, revision)
}

func (mk *mockETCDKV) LoadWithRevision(key string) ([]string, []string, int64, error) {
	return mk.loadWithRevision(key)
}

func (mk *mockETCDKV) RemoveWithPrefix(key string) error {
	return mk.removeWithPrefix(key)
}

type chunkManagerMock struct {
	storage.ChunkManager

	removeWithPrefix func(string) error
	listWithPrefix   func(string, bool) ([]string, []time.Time, error)
	remove           func(string) error
}

func (cmm *chunkManagerMock) RootPath() string {
	return ""
}

func (cmm *chunkManagerMock) RemoveWithPrefix(prefix string) error {
	return cmm.removeWithPrefix(prefix)
}

func (cmm *chunkManagerMock) ListWithPrefix(prefix string, recursive bool) ([]string, []time.Time, error) {
	return cmm.listWithPrefix(prefix, recursive)
}

func (cmm *chunkManagerMock) Remove(key string) error {
	return cmm.remove(key)
}
