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

package rootcoord

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/stretchr/testify/assert"
)

type customKV struct {
	kv.MockMetaKV
}

func TestImportManager_NewImportManager(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	Params.RootCoordCfg.ImportTaskExpiration = 50
	Params.RootCoordCfg.ImportTaskRetention = 200
	checkPendingTasksInterval = 100
	cleanUpLoopInterval = 100
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))

	mockCallImportServiceErr := false
	callImportServiceFn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if mockCallImportServiceErr {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, errors.New("mock err")
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	var wg sync.WaitGroup
	wg.Add(1)
	t.Run("working task expired", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		assert.NotNil(t, mgr)
		_, err := mgr.loadFromTaskStore(true)
		assert.NoError(t, err)
		var wgLoop sync.WaitGroup
		wgLoop.Add(2)
		mgr.sendOutTasks(ctx)
		assert.Equal(t, 1, len(mgr.workingTasks))
		mgr.cleanupLoop(&wgLoop)
		assert.Equal(t, 0, len(mgr.workingTasks))
		mgr.sendOutTasksLoop(&wgLoop)
		wgLoop.Wait()
	})

	wg.Add(1)
	t.Run("context done", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		assert.NotNil(t, mgr)
		mgr.init(context.TODO())
		var wgLoop sync.WaitGroup
		wgLoop.Add(2)
		mgr.cleanupLoop(&wgLoop)
		mgr.sendOutTasksLoop(&wgLoop)
		wgLoop.Wait()
	})

	wg.Add(1)
	t.Run("importManager init fail because of loadFromTaskStore fail", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		mockKv.LoadWithPrefixMockErr = true
		defer func() {
			mockKv.LoadWithPrefixMockErr = false
		}()
		assert.NotNil(t, mgr)
		assert.Panics(t, func() {
			mgr.init(context.TODO())
		})
	})

	wg.Add(1)
	t.Run("sendOutTasks fail", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		mockKv.SaveMockErr = true
		defer func() {
			mockKv.SaveMockErr = false
		}()
		assert.NotNil(t, mgr)
		mgr.init(context.TODO())
	})

	wg.Add(1)
	t.Run("sendOutTasks fail", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		assert.NotNil(t, mgr)
		mgr.init(context.TODO())
		func() {
			mockKv.SaveMockErr = true
			defer func() {
				mockKv.SaveMockErr = false
			}()
			mgr.sendOutTasks(context.TODO())
		}()

		func() {
			mockCallImportServiceErr = true
			defer func() {
				mockKv.SaveMockErr = false
			}()
			mgr.sendOutTasks(context.TODO())
		}()
	})

	wg.Add(1)
	t.Run("pending task expired", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		assert.NotNil(t, mgr)
		mgr.pendingTasks = append(mgr.pendingTasks, &datapb.ImportTaskInfo{
			Id: 300,
			State: &datapb.ImportTaskState{
				StateCode: commonpb.ImportState_ImportPending,
			},
			CreateTs: time.Now().Unix() + 1,
		})
		mgr.pendingTasks = append(mgr.pendingTasks, &datapb.ImportTaskInfo{
			Id: 400,
			State: &datapb.ImportTaskState{
				StateCode: commonpb.ImportState_ImportPending,
			},
			CreateTs: time.Now().Unix() - 100,
		})
		_, err := mgr.loadFromTaskStore(true)
		assert.NoError(t, err)
		var wgLoop sync.WaitGroup
		wgLoop.Add(2)
		assert.Equal(t, 2, len(mgr.pendingTasks))
		mgr.cleanupLoop(&wgLoop)
		assert.Equal(t, 1, len(mgr.pendingTasks))
		mgr.sendOutTasksLoop(&wgLoop)
		wgLoop.Wait()
	})

	wg.Add(1)
	t.Run("check init", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
		assert.NotNil(t, mgr)
		mgr.init(ctx)
		var wgLoop sync.WaitGroup
		wgLoop.Add(2)
		mgr.cleanupLoop(&wgLoop)
		mgr.sendOutTasksLoop(&wgLoop)
		time.Sleep(100 * time.Millisecond)
		wgLoop.Wait()
	})

	wg.Wait()
}

func TestImportManager_TestEtcdCleanUp(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	Params.RootCoordCfg.ImportTaskExpiration = 50
	Params.RootCoordCfg.ImportTaskRetention = 200
	checkPendingTasksInterval = 100
	cleanUpLoopInterval = 100
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 500,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 500,
	}
	ti3 := &datapb.ImportTaskInfo{
		Id: 300,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo3, err := proto.Marshal(ti3)
	assert.NoError(t, err)
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))
	mockKv.Save(BuildImportTaskKey(300), string(taskInfo3))

	mockCallImportServiceErr := false
	callImportServiceFn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if mockCallImportServiceErr {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, errors.New("mock err")
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil, nil, nil, nil)
	assert.NotNil(t, mgr)
	_, err = mgr.loadFromTaskStore(true)
	assert.NoError(t, err)
	var wgLoop sync.WaitGroup
	wgLoop.Add(2)
	keys, _, _ := mockKv.LoadWithPrefix("")
	// All 3 tasks are stored in Etcd.
	assert.Equal(t, 3, len(keys))
	mgr.cleanupLoop(&wgLoop)
	keys, _, _ = mockKv.LoadWithPrefix("")
	// task 1 and task 2 have passed retention period.
	assert.Equal(t, 1, len(keys))
	mgr.sendOutTasksLoop(&wgLoop)
}

func TestImportManager_TestFlipTaskStateLoop(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	Params.RootCoordCfg.ImportTaskExpiration = 50
	Params.RootCoordCfg.ImportTaskRetention = 200
	checkPendingTasksInterval = 100
	cleanUpLoopInterval = 100
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
			Segments:  []int64{201, 202, 203},
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))

	mockCallImportServiceErr := false
	callImportServiceFn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if mockCallImportServiceErr {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, errors.New("mock err")
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	callDescribeIndex := func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
		return &indexpb.DescribeIndexResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
			IndexInfos: []*indexpb.IndexInfo{
				{},
			},
		}, nil
	}
	callGetSegmentIndexState := func(ctx context.Context, collID UniqueID, indexName string,
		segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
		return []*indexpb.SegmentIndexState{
			{
				SegmentID: 200,
				State:     commonpb.IndexState_Finished,
			},
			{
				SegmentID: 201,
				State:     commonpb.IndexState_Finished,
			},
			{
				SegmentID: 202,
				State:     commonpb.IndexState_Finished,
			},
		}, nil
	}
	callUnsetIsImportingState := func(context.Context, *datapb.UnsetIsImportingStateRequest) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	flipTaskStateInterval = 50
	var wg sync.WaitGroup
	wg.Add(1)
	t.Run("normal case", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped,
			nil, callDescribeIndex, callGetSegmentIndexState, callUnsetIsImportingState)
		assert.NotNil(t, mgr)
		var wgLoop sync.WaitGroup
		wgLoop.Add(1)
		mgr.flipTaskStateLoop(&wgLoop)
		wgLoop.Wait()
		time.Sleep(100 * time.Millisecond)
	})

	wg.Add(1)
	t.Run("describe index fail", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		callDescribeIndex = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_UnexpectedError,
				},
			}, nil
		}
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped,
			nil, callDescribeIndex, callGetSegmentIndexState, callUnsetIsImportingState)
		assert.NotNil(t, mgr)
		var wgLoop sync.WaitGroup
		wgLoop.Add(1)
		mgr.flipTaskStateLoop(&wgLoop)
		wgLoop.Wait()
		time.Sleep(100 * time.Millisecond)
	})

	wg.Add(1)
	t.Run("describe index with index not exist", func(t *testing.T) {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		callDescribeIndex = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_IndexNotExist,
				},
			}, nil
		}
		mgr := newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped,
			nil, callDescribeIndex, callGetSegmentIndexState, callUnsetIsImportingState)
		assert.NotNil(t, mgr)
		var wgLoop sync.WaitGroup
		wgLoop.Add(1)
		mgr.flipTaskStateLoop(&wgLoop)
		wgLoop.Wait()
		time.Sleep(100 * time.Millisecond)
	})
	wg.Wait()
}

func TestImportManager_ImportJob(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	colID := int64(100)
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	mgr := newImportManager(context.TODO(), mockKv, idAlloc, nil, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp := mgr.importJob(context.TODO(), nil, colID, 0)
	assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)

	rowReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       true,
		Files:          []string{"f1", "f2", "f3"},
	}

	resp = mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)

	colReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       false,
		Files:          []string{"f1", "f2"},
		Options: []*commonpb.KeyValuePair{
			{
				Key:   Bucket,
				Value: "mybucket",
			},
		},
	}

	fn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
			},
		}, nil
	}

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp = mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.Equal(t, len(rowReq.Files), len(mgr.pendingTasks))
	assert.Equal(t, 0, len(mgr.workingTasks))

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp = mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 1, len(mgr.pendingTasks))
	assert.Equal(t, 0, len(mgr.workingTasks))

	fn = func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp = mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, len(rowReq.Files), len(mgr.workingTasks))

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp = mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, 1, len(mgr.workingTasks))

	count := 0
	fn = func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if count >= 2 {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_UnexpectedError,
				},
			}, nil
		}
		count++
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	resp = mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.Equal(t, len(rowReq.Files)-2, len(mgr.pendingTasks))
	assert.Equal(t, 2, len(mgr.workingTasks))

	for i := 0; i <= 32; i++ {
		rowReq.Files = append(rowReq.Files, strconv.Itoa(i))
	}
	resp = mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
}

func TestImportManager_AllDataNodesBusy(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	colID := int64(100)
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	rowReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       true,
		Files:          []string{"f1", "f2", "f3"},
	}
	colReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       false,
		Files:          []string{"f1", "f2"},
		Options: []*commonpb.KeyValuePair{
			{
				Key:   Bucket,
				Value: "mybucket",
			},
		},
	}

	dnList := []int64{1, 2, 3}
	count := 0
	fn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if count < len(dnList) {
			count++
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				DatanodeId: dnList[count-1],
			}, nil
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
			},
		}, nil
	}

	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	mgr := newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, len(rowReq.Files), len(mgr.workingTasks))

	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), rowReq, colID, 0)
	assert.Equal(t, len(rowReq.Files), len(mgr.pendingTasks))
	assert.Equal(t, 0, len(mgr.workingTasks))

	// Reset count.
	count = 0
	mgr = newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, 1, len(mgr.workingTasks))

	mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, 2, len(mgr.workingTasks))

	mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 0, len(mgr.pendingTasks))
	assert.Equal(t, 3, len(mgr.workingTasks))

	mgr.importJob(context.TODO(), colReq, colID, 0)
	assert.Equal(t, 1, len(mgr.pendingTasks))
	assert.Equal(t, 3, len(mgr.workingTasks))
}

func TestImportManager_TaskState(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	colID := int64(100)
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	fn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	rowReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       true,
		Files:          []string{"f1", "f2", "f3"},
	}

	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	mgr := newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), rowReq, colID, 0)

	info := &rootcoordpb.ImportResult{
		TaskId: 10000,
	}
	_, err := mgr.updateTaskInfo(info)
	assert.NotNil(t, err)

	info = &rootcoordpb.ImportResult{
		TaskId:   2,
		RowCount: 1000,
		State:    commonpb.ImportState_ImportPersisted,
		Infos: []*commonpb.KeyValuePair{
			{
				Key:   "key1",
				Value: "value1",
			},
			{
				Key:   "failed_reason",
				Value: "some_reason",
			},
		},
	}
	ti, err := mgr.updateTaskInfo(info)
	assert.NoError(t, err)
	assert.Equal(t, int64(2), ti.GetId())
	assert.Equal(t, int64(100), ti.GetCollectionId())
	assert.Equal(t, int64(0), ti.GetPartitionId())
	assert.Equal(t, true, ti.GetRowBased())
	assert.Equal(t, []string{"f2"}, ti.GetFiles())
	assert.Equal(t, commonpb.ImportState_ImportPersisted, ti.GetState().GetStateCode())
	assert.Equal(t, int64(1000), ti.GetState().GetRowCount())

	resp := mgr.getTaskState(10000)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)

	resp = mgr.getTaskState(2)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, commonpb.ImportState_ImportPersisted, resp.State)

	resp = mgr.getTaskState(1)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, commonpb.ImportState_ImportStarted, resp.State)

	info = &rootcoordpb.ImportResult{
		TaskId:   1,
		RowCount: 1000,
		State:    commonpb.ImportState_ImportFailed,
		Infos: []*commonpb.KeyValuePair{
			{
				Key:   "key1",
				Value: "value1",
			},
			{
				Key:   "failed_reason",
				Value: "some_reason",
			},
		},
	}
	newTaskInfo, err := mgr.updateTaskInfo(info)
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ImportState_ImportFailed, newTaskInfo.GetState().GetStateCode())

	newTaskInfo, err = mgr.updateTaskInfo(info)
	assert.Error(t, err)
	assert.Nil(t, newTaskInfo)
}

func TestImportManager_AllocFail(t *testing.T) {
	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		return 0, 0, errors.New("injected failure")
	}
	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	colID := int64(100)
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	fn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	rowReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       true,
		Files:          []string{"f1", "f2", "f3"},
	}

	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	mgr := newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), rowReq, colID, 0)
}

func TestImportManager_ListAllTasks(t *testing.T) {
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)

	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}

	Params.RootCoordCfg.ImportTaskSubPath = "test_import_task"
	colID := int64(100)
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}

	// reject some tasks so there are 3 tasks left in pending list
	fn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
			},
		}, nil
	}

	rowReq := &milvuspb.ImportRequest{
		CollectionName: "c1",
		PartitionName:  "p1",
		RowBased:       true,
		Files:          []string{"f1", "f2", "f3"},
	}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	mgr := newImportManager(context.TODO(), mockKv, idAlloc, fn, callMarkSegmentsDropped, nil, nil, nil, nil)
	mgr.importJob(context.TODO(), rowReq, colID, 0)

	tasks := mgr.listAllTasks("", 100)
	assert.Equal(t, len(rowReq.Files), len(tasks))

	resp := mgr.getTaskState(1)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, commonpb.ImportState_ImportPending, resp.State)
	assert.Equal(t, int64(1), resp.Id)

	// accept tasks to working list
	mgr.callImportService = func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}

	mgr.importJob(context.TODO(), rowReq, colID, 0)
	tasks = mgr.listAllTasks("", 100)
	assert.Equal(t, len(rowReq.Files)*2, len(tasks))
	tasks = mgr.listAllTasks("", 1)
	assert.Equal(t, 1, len(tasks))
	tasks = mgr.listAllTasks("bad-collection-name", 1)
	assert.Equal(t, 0, len(tasks))

	// the id of tasks must be 1,2,3,4,5,6(sequence not guaranteed)
	ids := make(map[int64]struct{})
	for i := 0; i < len(tasks); i++ {
		ids[int64(i)+1] = struct{}{}
	}
	for i := 0; i < len(tasks); i++ {
		delete(ids, tasks[i].Id)
	}
	assert.Equal(t, 0, len(ids))
}

func TestImportManager_setCollectionPartitionName(t *testing.T) {
	mgr := &importManager{
		getCollectionName: func(collID, partitionID typeutil.UniqueID) (string, string, error) {
			if collID == 1 && partitionID == 2 {
				return "c1", "p1", nil
			}
			return "", "", errors.New("Error")
		},
	}

	info := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportStarted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	err := mgr.setCollectionPartitionName(1, 2, info)
	assert.Nil(t, err)
	assert.Equal(t, "c1", info.GetCollectionName())
	assert.Equal(t, "p1", info.GetPartitionName())

	err = mgr.setCollectionPartitionName(0, 0, info)
	assert.Error(t, err)
}

func TestImportManager_rearrangeTasks(t *testing.T) {
	tasks := make([]*milvuspb.GetImportStateResponse, 0)
	tasks = append(tasks, &milvuspb.GetImportStateResponse{
		Id: 100,
	})
	tasks = append(tasks, &milvuspb.GetImportStateResponse{
		Id: 1,
	})
	tasks = append(tasks, &milvuspb.GetImportStateResponse{
		Id: 50,
	})
	rearrangeTasks(tasks)
	assert.Equal(t, 3, len(tasks))
	assert.Equal(t, int64(1), tasks[0].GetId())
	assert.Equal(t, int64(50), tasks[1].GetId())
	assert.Equal(t, int64(100), tasks[2].GetId())
}
