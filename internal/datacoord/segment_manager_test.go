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

package datacoord

import (
	"context"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/schemapb"
	memkv "github.com/milvus-io/milvus/internal/kv/mem"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/stretchr/testify/assert"
)

func TestManagerOptions(t *testing.T) {
	//	ctx := context.Background()
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)
	segmentManager := newSegmentManager(meta, mockAllocator, nil)

	t.Run("test with alloc helper", func(t *testing.T) {
		opt := withAllocHelper(allocHelper{})
		opt.apply(segmentManager)

		assert.True(t, segmentManager.helper.afterCreateSegment == nil)
	})

	t.Run("test withCalUpperLimitPolicy", func(t *testing.T) {
		opt := withCalUpperLimitPolicy(defaultCalUpperLimitPolicy())
		assert.NotNil(t, opt)

		//manual set nil``
		segmentManager.estimatePolicy = nil
		opt.apply(segmentManager)
		assert.True(t, segmentManager.estimatePolicy != nil)
	})

	t.Run("test withAllocPolicy", func(t *testing.T) {
		opt := withAllocPolicy(defaultAllocatePolicy())
		assert.NotNil(t, opt)
		// manual set nil
		segmentManager.allocPolicy = nil
		opt.apply(segmentManager)
		assert.True(t, segmentManager.allocPolicy != nil)
	})

	t.Run("test withSegmentSealPolicy", func(t *testing.T) {
		opt := withSegmentSealPolices(defaultSegmentSealPolicy()...)
		assert.NotNil(t, opt)
		// manual set nil
		segmentManager.segmentSealPolicies = []segmentSealPolicy{}
		opt.apply(segmentManager)
		assert.True(t, len(segmentManager.segmentSealPolicies) > 0)
	})

	t.Run("test withChannelSealPolicies", func(t *testing.T) {
		opt := withChannelSealPolices(getChannelOpenSegCapacityPolicy(1000))
		assert.NotNil(t, opt)
		// manaul set nil
		segmentManager.channelSealPolicies = []channelSealPolicy{}
		opt.apply(segmentManager)
		assert.True(t, len(segmentManager.channelSealPolicies) > 0)
	})
	t.Run("test withFlushPolicy", func(t *testing.T) {
		opt := withFlushPolicy(defaultFlushPolicy())
		assert.NotNil(t, opt)
		// manual set nil
		segmentManager.flushPolicy = nil
		opt.apply(segmentManager)
		assert.True(t, segmentManager.flushPolicy != nil)
	})
}

func TestAllocSegment(t *testing.T) {
	ctx := context.Background()
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)
	segmentManager := newSegmentManager(meta, mockAllocator, nil)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(ctx)
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})

	t.Run("normal allocation", func(t *testing.T) {
		allocations, err := segmentManager.AllocSegment(ctx, collID, 100, "c1", 100)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))
		assert.EqualValues(t, 100, allocations[0].NumOfRows)
		assert.NotEqualValues(t, 0, allocations[0].SegmentID)
		assert.NotEqualValues(t, 0, allocations[0].ExpireTime)
	})

	t.Run("allocation fails 1", func(t *testing.T) {
		failsAllocator := &FailsAllocator{
			allocTsSucceed: true,
		}
		segmentManager := newSegmentManager(meta, failsAllocator, nil)
		_, err := segmentManager.AllocSegment(ctx, collID, 100, "c2", 100)
		assert.NotNil(t, err)
	})

	t.Run("allocation fails 2", func(t *testing.T) {
		failsAllocator := &FailsAllocator{
			allocIDSucceed: true,
		}
		segmentManager := newSegmentManager(meta, failsAllocator, nil)
		_, err := segmentManager.AllocSegment(ctx, collID, 100, "c1", 100)
		assert.NotNil(t, err)
	})
}

func TestAllocSegmentForImport(t *testing.T) {
	ctx := context.Background()
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)
	ms := newMockRootCoordService()
	segmentManager := newSegmentManager(meta, mockAllocator, ms)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(ctx)
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})

	t.Run("normal allocation", func(t *testing.T) {
		allocation, err := segmentManager.allocSegmentForImport(ctx, collID, 100, "c1", 100, 0)
		assert.Nil(t, err)
		assert.NotNil(t, allocation)
		assert.EqualValues(t, 100, allocation.NumOfRows)
		assert.NotEqualValues(t, 0, allocation.SegmentID)
		assert.NotEqualValues(t, 0, allocation.ExpireTime)
	})

	t.Run("allocation fails 1", func(t *testing.T) {
		failsAllocator := &FailsAllocator{
			allocTsSucceed: true,
		}
		segmentManager := newSegmentManager(meta, failsAllocator, ms)
		_, err := segmentManager.allocSegmentForImport(ctx, collID, 100, "c1", 100, 0)
		assert.NotNil(t, err)
	})

	t.Run("allocation fails 2", func(t *testing.T) {
		failsAllocator := &FailsAllocator{
			allocIDSucceed: true,
		}
		segmentManager := newSegmentManager(meta, failsAllocator, ms)
		_, err := segmentManager.allocSegmentForImport(ctx, collID, 100, "c1", 100, 0)
		assert.NotNil(t, err)
	})

	t.Run("nil RootCoord", func(t *testing.T) {
		segmentManager := newSegmentManager(meta, mockAllocator, nil)
		_, err := segmentManager.allocSegmentForImport(ctx, collID, 100, "c1", 100, 0)
		assert.NotNil(t, err)
	})
}

func TestLoadSegmentsFromMeta(t *testing.T) {
	ctx := context.Background()
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(ctx)
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})

	sealedSegment := &datapb.SegmentInfo{
		ID:             1,
		CollectionID:   collID,
		PartitionID:    0,
		InsertChannel:  "",
		State:          commonpb.SegmentState_Sealed,
		MaxRowNum:      100,
		LastExpireTime: 1000,
	}
	growingSegment := &datapb.SegmentInfo{
		ID:             2,
		CollectionID:   collID,
		PartitionID:    0,
		InsertChannel:  "",
		State:          commonpb.SegmentState_Growing,
		MaxRowNum:      100,
		LastExpireTime: 1000,
	}
	flushedSegment := &datapb.SegmentInfo{
		ID:             3,
		CollectionID:   collID,
		PartitionID:    0,
		InsertChannel:  "",
		State:          commonpb.SegmentState_Flushed,
		MaxRowNum:      100,
		LastExpireTime: 1000,
	}
	err = meta.AddSegment(NewSegmentInfo(sealedSegment))
	assert.Nil(t, err)
	err = meta.AddSegment(NewSegmentInfo(growingSegment))
	assert.Nil(t, err)
	err = meta.AddSegment(NewSegmentInfo(flushedSegment))
	assert.Nil(t, err)

	segmentManager := newSegmentManager(meta, mockAllocator, nil)
	segments := segmentManager.segments
	assert.EqualValues(t, 2, len(segments))
}

func TestSaveSegmentsToMeta(t *testing.T) {
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(context.Background())
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
	segmentManager := newSegmentManager(meta, mockAllocator, nil)
	allocations, err := segmentManager.AllocSegment(context.Background(), collID, 0, "c1", 1000)
	assert.Nil(t, err)
	assert.EqualValues(t, 1, len(allocations))
	_, err = segmentManager.SealAllSegments(context.Background(), collID, nil)
	assert.Nil(t, err)
	segment := meta.GetSegment(allocations[0].SegmentID)
	assert.NotNil(t, segment)
	assert.EqualValues(t, segment.LastExpireTime, allocations[0].ExpireTime)
	assert.EqualValues(t, commonpb.SegmentState_Sealed, segment.State)
}

func TestSaveSegmentsToMetaWithSpecificSegments(t *testing.T) {
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(context.Background())
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
	segmentManager := newSegmentManager(meta, mockAllocator, nil)
	allocations, err := segmentManager.AllocSegment(context.Background(), collID, 0, "c1", 1000)
	assert.Nil(t, err)
	assert.EqualValues(t, 1, len(allocations))
	_, err = segmentManager.SealAllSegments(context.Background(), collID, []int64{allocations[0].SegmentID})
	assert.Nil(t, err)
	segment := meta.GetSegment(allocations[0].SegmentID)
	assert.NotNil(t, segment)
	assert.EqualValues(t, segment.LastExpireTime, allocations[0].ExpireTime)
	assert.EqualValues(t, commonpb.SegmentState_Sealed, segment.State)
}

func TestDropSegment(t *testing.T) {
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(context.Background())
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
	segmentManager := newSegmentManager(meta, mockAllocator, nil)
	allocations, err := segmentManager.AllocSegment(context.Background(), collID, 0, "c1", 1000)
	assert.Nil(t, err)
	assert.EqualValues(t, 1, len(allocations))
	segID := allocations[0].SegmentID
	segment := meta.GetSegment(segID)
	assert.NotNil(t, segment)

	segmentManager.DropSegment(context.Background(), segID)
	segment = meta.GetSegment(segID)
	assert.NotNil(t, segment)
}

func TestAllocRowsLargerThanOneSegment(t *testing.T) {
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(context.Background())
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})

	var mockPolicy = func(schema *schemapb.CollectionSchema) (int, error) {
		return 1, nil
	}
	segmentManager := newSegmentManager(meta, mockAllocator, nil, withCalUpperLimitPolicy(mockPolicy))
	allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
	assert.Nil(t, err)
	assert.EqualValues(t, 2, len(allocations))
	assert.EqualValues(t, 1, allocations[0].NumOfRows)
	assert.EqualValues(t, 1, allocations[1].NumOfRows)
}

func TestExpireAllocation(t *testing.T) {
	Params.Init()
	mockAllocator := newMockAllocator()
	meta, err := newMemoryMeta(mockAllocator)
	assert.Nil(t, err)

	schema := newTestSchema()
	collID, err := mockAllocator.allocID(context.Background())
	assert.Nil(t, err)
	meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})

	var mockPolicy = func(schema *schemapb.CollectionSchema) (int, error) {
		return 10000000, nil
	}
	segmentManager := newSegmentManager(meta, mockAllocator, nil, withCalUpperLimitPolicy(mockPolicy))
	// alloc 100 times and expire
	var maxts Timestamp
	var id int64 = -1
	for i := 0; i < 100; i++ {
		allocs, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "ch1", 100)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocs))
		if id == -1 {
			id = allocs[0].SegmentID
		} else {
			assert.EqualValues(t, id, allocs[0].SegmentID)
		}
		if allocs[0].ExpireTime > maxts {
			maxts = allocs[0].ExpireTime
		}
	}

	segment := meta.GetSegment(id)
	assert.NotNil(t, segment)
	assert.EqualValues(t, 100, len(segment.allocations))
	err = segmentManager.ExpireAllocations("ch1", maxts)
	assert.Nil(t, err)
	segment = meta.GetSegment(id)
	assert.NotNil(t, segment)
	assert.EqualValues(t, 0, len(segment.allocations))
}

func TestGetFlushableSegments(t *testing.T) {
	t.Run("get flushable segments between small interval", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		meta, err := newMemoryMeta(mockAllocator)
		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil)
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		ids, err := segmentManager.SealAllSegments(context.TODO(), collID, nil)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(ids))
		assert.EqualValues(t, allocations[0].SegmentID, ids[0])

		meta.SetCurrentRows(allocations[0].SegmentID, 1)
		ids, err = segmentManager.GetFlushableSegments(context.TODO(), "c1", allocations[0].ExpireTime)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(ids))
		assert.EqualValues(t, allocations[0].SegmentID, ids[0])

		meta.SetLastFlushTime(allocations[0].SegmentID, time.Now())
		ids, err = segmentManager.GetFlushableSegments(context.TODO(), "c1", allocations[0].ExpireTime)
		assert.Nil(t, err)
		assert.Empty(t, ids)

		meta.SetLastFlushTime(allocations[0].SegmentID, time.Now().Local().Add(-flushInterval))
		ids, err = segmentManager.GetFlushableSegments(context.TODO(), "c1", allocations[0].ExpireTime)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(ids))
		assert.EqualValues(t, allocations[0].SegmentID, ids[0])

		meta.SetCurrentRows(allocations[0].SegmentID, 0)
		ids, err = segmentManager.GetFlushableSegments(context.TODO(), "c1", allocations[0].ExpireTime)
		assert.Nil(t, err)
		assert.Empty(t, ids)
		assert.Nil(t, meta.GetSegment(allocations[0].SegmentID))
	})
}

func TestTryToSealSegment(t *testing.T) {
	t.Run("normal seal with segment policies", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		meta, err := newMemoryMeta(mockAllocator)
		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil, withSegmentSealPolices(sealByLifetimePolicy(math.MinInt64))) //always seal
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		ts, err := segmentManager.allocator.allocTimestamp(context.Background())
		assert.Nil(t, err)
		err = segmentManager.tryToSealSegment(ts, "c1")
		assert.Nil(t, err)

		for _, seg := range segmentManager.meta.segments.segments {
			assert.Equal(t, commonpb.SegmentState_Sealed, seg.GetState())
		}
	})

	t.Run("normal seal with channel seal policies", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		meta, err := newMemoryMeta(mockAllocator)
		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil, withChannelSealPolices(getChannelOpenSegCapacityPolicy(-1))) //always seal
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		ts, err := segmentManager.allocator.allocTimestamp(context.Background())
		assert.Nil(t, err)
		err = segmentManager.tryToSealSegment(ts, "c1")
		assert.Nil(t, err)

		for _, seg := range segmentManager.meta.segments.segments {
			assert.Equal(t, commonpb.SegmentState_Sealed, seg.GetState())
		}
	})

	t.Run("normal seal with both segment & channel seal policy", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		meta, err := newMemoryMeta(mockAllocator)
		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil,
			withSegmentSealPolices(sealByLifetimePolicy(math.MinInt64)),
			withChannelSealPolices(getChannelOpenSegCapacityPolicy(-1))) //always seal
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		ts, err := segmentManager.allocator.allocTimestamp(context.Background())
		assert.Nil(t, err)
		err = segmentManager.tryToSealSegment(ts, "c1")
		assert.Nil(t, err)

		for _, seg := range segmentManager.meta.segments.segments {
			assert.Equal(t, commonpb.SegmentState_Sealed, seg.GetState())
		}
	})

	t.Run("seal with segment policy with kv fails", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		memoryKV := memkv.NewMemoryKV()
		fkv := &saveFailKV{TxnKV: memoryKV}
		meta, err := newMeta(context.TODO(), memoryKV)

		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil, withSegmentSealPolices(sealByLifetimePolicy(math.MinInt64))) //always seal
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		segmentManager.meta.catalog = &datacoord.Catalog{Txn: fkv}

		ts, err := segmentManager.allocator.allocTimestamp(context.Background())
		assert.Nil(t, err)
		err = segmentManager.tryToSealSegment(ts, "c1")
		assert.NotNil(t, err)
	})

	t.Run("seal with channel policy with kv fails", func(t *testing.T) {
		Params.Init()
		mockAllocator := newMockAllocator()
		memoryKV := memkv.NewMemoryKV()
		fkv := &saveFailKV{TxnKV: memoryKV}
		meta, err := newMeta(context.TODO(), memoryKV)

		assert.Nil(t, err)

		schema := newTestSchema()
		collID, err := mockAllocator.allocID(context.Background())
		assert.Nil(t, err)
		meta.AddCollection(&datapb.CollectionInfo{ID: collID, Schema: schema})
		segmentManager := newSegmentManager(meta, mockAllocator, nil, withChannelSealPolices(getChannelOpenSegCapacityPolicy(-1))) //always seal
		allocations, err := segmentManager.AllocSegment(context.TODO(), collID, 0, "c1", 2)
		assert.Nil(t, err)
		assert.EqualValues(t, 1, len(allocations))

		segmentManager.meta.catalog = &datacoord.Catalog{Txn: fkv}

		ts, err := segmentManager.allocator.allocTimestamp(context.Background())
		assert.Nil(t, err)
		err = segmentManager.tryToSealSegment(ts, "c1")
		assert.NotNil(t, err)
	})
}

func TestAllocationPool(t *testing.T) {
	t.Run("normal get&put", func(t *testing.T) {
		allocPool = sync.Pool{
			New: func() interface{} {
				return &Allocation{}
			},
		}

		allo := getAllocation(100)
		assert.EqualValues(t, 100, allo.NumOfRows)
		assert.EqualValues(t, 0, allo.ExpireTime)
		assert.EqualValues(t, 0, allo.SegmentID)

		putAllocation(allo)
	})

	t.Run("put nil", func(t *testing.T) {
		var allo *Allocation
		allocPool = sync.Pool{
			New: func() interface{} {
				return &Allocation{}
			},
		}
		putAllocation(allo)
		allo = getAllocation(100)
		assert.EqualValues(t, 100, allo.NumOfRows)
		assert.EqualValues(t, 0, allo.ExpireTime)
		assert.EqualValues(t, 0, allo.SegmentID)
	})

	t.Run("put something else", func(t *testing.T) {
		allocPool = sync.Pool{
			New: func() interface{} {
				return &Allocation{}
			},
		}
		allocPool.Put(&struct{}{})
		allo := getAllocation(100)
		assert.EqualValues(t, 100, allo.NumOfRows)
		assert.EqualValues(t, 0, allo.ExpireTime)
		assert.EqualValues(t, 0, allo.SegmentID)

	})
}

func TestSegmentManager_DropSegmentsOfChannel(t *testing.T) {
	type fields struct {
		meta     *meta
		segments []UniqueID
	}
	type args struct {
		channel string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   []UniqueID
	}{
		{
			"test drop segments",
			fields{
				meta: &meta{
					segments: &SegmentsInfo{
						segments: map[int64]*SegmentInfo{
							1: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:            1,
									InsertChannel: "ch1",
									State:         commonpb.SegmentState_Flushed,
								},
							},
							2: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:            2,
									InsertChannel: "ch2",
									State:         commonpb.SegmentState_Flushed,
								},
							},
						},
					},
				},
				segments: []UniqueID{1, 2},
			},
			args{
				"ch1",
			},
			[]UniqueID{2},
		},
		{
			"test drop segments with dropped segment",
			fields{
				meta: &meta{
					segments: &SegmentsInfo{
						segments: map[int64]*SegmentInfo{
							1: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:            1,
									InsertChannel: "ch1",
									State:         commonpb.SegmentState_Dropped,
								},
							},
							2: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:            2,
									InsertChannel: "ch2",
									State:         commonpb.SegmentState_Growing,
								},
							},
						},
					},
				},
				segments: []UniqueID{1, 2, 3},
			},
			args{
				"ch1",
			},
			[]UniqueID{2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &SegmentManager{
				meta:     tt.fields.meta,
				segments: tt.fields.segments,
			}
			s.DropSegmentsOfChannel(context.TODO(), tt.args.channel)
			assert.ElementsMatch(t, tt.want, s.segments)
		})
	}
}
