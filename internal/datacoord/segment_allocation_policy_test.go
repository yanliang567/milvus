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
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/tsoutil"
)

func TestUpperLimitCalBySchema(t *testing.T) {
	type testCase struct {
		schema    *schemapb.CollectionSchema
		expected  int
		expectErr bool
	}
	testCases := []testCase{
		{
			schema:    nil,
			expected:  -1,
			expectErr: true,
		},
		{
			schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{},
			},
			expected:  -1,
			expectErr: true,
		},
		{
			schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						DataType: schemapb.DataType_FloatVector,
						TypeParams: []*commonpb.KeyValuePair{
							{Key: common.DimKey, Value: "bad_dim"},
						},
					},
				},
			},
			expected:  -1,
			expectErr: true,
		},
		{
			schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						DataType: schemapb.DataType_Int64,
					},
					{
						DataType: schemapb.DataType_Int32,
					},
					{
						DataType: schemapb.DataType_FloatVector,
						TypeParams: []*commonpb.KeyValuePair{
							{Key: common.DimKey, Value: "128"},
						},
					},
				},
			},
			expected:  int(Params.DataCoordCfg.SegmentMaxSize.GetAsFloat() * 1024 * 1024 / float64(524)),
			expectErr: false,
		},
	}
	for _, c := range testCases {
		result, err := calBySchemaPolicy(c.schema)
		if c.expectErr {
			assert.Error(t, err)
		} else {
			assert.Equal(t, c.expected, result)
		}
	}
}

func TestGetChannelOpenSegCapacityPolicy(t *testing.T) {
	p := getChannelOpenSegCapacityPolicy(3)
	type testCase struct {
		channel   string
		segments  []*SegmentInfo
		ts        Timestamp
		validator func([]*SegmentInfo) bool
	}
	testCases := []testCase{
		{
			segments: []*SegmentInfo{},
			ts:       tsoutil.ComposeTS(time.Now().Unix()/int64(time.Millisecond), rand.Int63n(1000)),
			validator: func(result []*SegmentInfo) bool {
				return len(result) == 0
			},
		},
		{
			segments: []*SegmentInfo{
				{
					SegmentInfo: &datapb.SegmentInfo{},
				},
				{
					SegmentInfo: &datapb.SegmentInfo{},
				},
				{
					SegmentInfo: &datapb.SegmentInfo{},
				},
				{
					SegmentInfo: &datapb.SegmentInfo{},
				},
			},
			ts: tsoutil.ComposeTS(time.Now().Unix()/int64(time.Millisecond), rand.Int63n(1000)),
			validator: func(result []*SegmentInfo) bool {
				return len(result) == 1
			},
		},
	}
	for _, c := range testCases {
		result, _ := p(c.channel, c.segments, c.ts)
		if c.validator != nil {
			assert.True(t, c.validator(result))
		}
	}
}

func TestCalBySegmentSizePolicy(t *testing.T) {
	t.Run("nil schema", func(t *testing.T) {
		rows, err := calBySegmentSizePolicy(nil, 1024)

		assert.Error(t, err)
		assert.Equal(t, -1, rows)
	})

	t.Run("get dim failed", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{
			Name:        "coll1",
			Description: "",
			Fields: []*schemapb.FieldSchema{
				{FieldID: fieldID, Name: "field0", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: fieldID + 1, Name: "field1", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "fake"}}},
			},
			EnableDynamicField: false,
			Properties:         nil,
		}

		rows, err := calBySegmentSizePolicy(schema, 1024)
		assert.Error(t, err)
		assert.Equal(t, -1, rows)
	})

	t.Run("sizePerRecord is zero", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{Fields: nil}
		rows, err := calBySegmentSizePolicy(schema, 1024)

		assert.Error(t, err)
		assert.Equal(t, -1, rows)
	})

	t.Run("normal case", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{
			Name:        "coll1",
			Description: "",
			Fields: []*schemapb.FieldSchema{
				{FieldID: fieldID, Name: "field0", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: fieldID + 1, Name: "field1", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "8"}}},
			},
			EnableDynamicField: false,
			Properties:         nil,
		}

		rows, err := calBySegmentSizePolicy(schema, 1200)
		assert.NoError(t, err)
		// 1200/(4*8+8)
		assert.Equal(t, 30, rows)
	})
}

func TestSortSegmentsByLastExpires(t *testing.T) {
	segs := make([]*SegmentInfo, 0, 10)
	for i := 0; i < 10; i++ {
		segs = append(segs,
			&SegmentInfo{
				SegmentInfo: &datapb.SegmentInfo{
					LastExpireTime: uint64(rand.Int31n(100000)),
				},
			})
	}
	sortSegmentsByLastExpires(segs)
	for i := 1; i < 10; i++ {
		assert.True(t, segs[i-1].LastExpireTime <= segs[i].LastExpireTime)
	}
}

func TestSealSegmentPolicy(t *testing.T) {
	paramtable.Init()
	t.Run("test seal segment by lifetime", func(t *testing.T) {
		paramtable.Get().Save(paramtable.Get().DataCoordCfg.SegmentMaxLifetime.Key, "2")
		defer paramtable.Get().Reset(paramtable.Get().DataCoordCfg.SegmentMaxLifetime.Key)

		lifetime := 2 * time.Second
		now := time.Now()
		curTS := now.UnixNano() / int64(time.Millisecond)
		nosealTs := (now.Add(lifetime / 2)).UnixNano() / int64(time.Millisecond)
		sealTs := (now.Add(lifetime)).UnixNano() / int64(time.Millisecond)

		p := sealL1SegmentByLifetime()

		segment := &SegmentInfo{
			SegmentInfo: &datapb.SegmentInfo{
				ID:            1,
				StartPosition: &msgpb.MsgPosition{Timestamp: tsoutil.ComposeTS(curTS, 0)},
			},
		}

		shouldSeal, _ := p.ShouldSeal(segment, tsoutil.ComposeTS(nosealTs, 0))
		assert.False(t, shouldSeal)

		shouldSeal, _ = p.ShouldSeal(segment, tsoutil.ComposeTS(sealTs, 0))
		assert.True(t, shouldSeal)
	})
}

func Test_sealLongTimeIdlePolicy(t *testing.T) {
	idleTimeTolerance := 2 * time.Second
	minSizeToSealIdleSegment := 16.0
	maxSizeOfSegment := 512.0
	policy := sealL1SegmentByIdleTime(idleTimeTolerance, minSizeToSealIdleSegment, maxSizeOfSegment)
	seg1 := &SegmentInfo{lastWrittenTime: time.Now().Add(idleTimeTolerance * 5)}
	shouldSeal, _ := policy.ShouldSeal(seg1, 100)
	assert.False(t, shouldSeal)
	seg2 := &SegmentInfo{lastWrittenTime: getZeroTime(), SegmentInfo: &datapb.SegmentInfo{MaxRowNum: 10000, NumOfRows: 1}}
	shouldSeal, _ = policy.ShouldSeal(seg2, 100)
	assert.False(t, shouldSeal)
	seg3 := &SegmentInfo{lastWrittenTime: getZeroTime(), SegmentInfo: &datapb.SegmentInfo{MaxRowNum: 10000, NumOfRows: 1000}}
	shouldSeal, _ = policy.ShouldSeal(seg3, 100)
	assert.True(t, shouldSeal)
}

func Test_sealByTotalGrowingSegmentsSize(t *testing.T) {
	paramtable.Get().Save(paramtable.Get().DataCoordCfg.GrowingSegmentsMemSizeInMB.Key, "100")
	defer paramtable.Get().Reset(paramtable.Get().DataCoordCfg.GrowingSegmentsMemSizeInMB.Key)

	seg0 := &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID:      0,
		State:   commonpb.SegmentState_Growing,
		Binlogs: []*datapb.FieldBinlog{{Binlogs: []*datapb.Binlog{{MemorySize: 30 * MB}}}},
	}}
	seg1 := &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID:      1,
		State:   commonpb.SegmentState_Growing,
		Binlogs: []*datapb.FieldBinlog{{Binlogs: []*datapb.Binlog{{MemorySize: 40 * MB}}}},
	}}
	seg2 := &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID:      2,
		State:   commonpb.SegmentState_Growing,
		Binlogs: []*datapb.FieldBinlog{{Binlogs: []*datapb.Binlog{{MemorySize: 50 * MB}}}},
	}}
	seg3 := &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID:    3,
		State: commonpb.SegmentState_Sealed,
	}}

	fn := sealByTotalGrowingSegmentsSize()
	// size not reach threshold
	res, _ := fn("ch-0", []*SegmentInfo{seg0}, 0)
	assert.Equal(t, 0, len(res))
	// size reached the threshold
	res, _ = fn("ch-0", []*SegmentInfo{seg0, seg1, seg2, seg3}, 0)
	assert.Equal(t, 1, len(res))
	assert.Equal(t, seg2.GetID(), res[0].GetID())
}

func Test_sealByBlockingL0(t *testing.T) {
	paramtable.Init()
	pt := paramtable.Get()
	type testCase struct {
		tag             string
		channel         string
		sizeLimit       int64
		entryNumLimit   int64
		l0Segments      []*SegmentInfo
		growingSegments []*SegmentInfo
		expected        []int64
	}

	l0_1 := &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:            1001,
			InsertChannel: "channel_1",
			Deltalogs: []*datapb.FieldBinlog{
				{
					Binlogs: []*datapb.Binlog{
						{EntriesNum: 50, MemorySize: 1 * 1024 * 1024},
					},
				},
			},
			Level:         datapb.SegmentLevel_L0,
			StartPosition: &msgpb.MsgPosition{Timestamp: 10},
			DmlPosition:   &msgpb.MsgPosition{Timestamp: 20},
		},
	}
	l0_2 := &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:            1002,
			InsertChannel: "channel_1",
			Deltalogs: []*datapb.FieldBinlog{
				{
					Binlogs: []*datapb.Binlog{
						{EntriesNum: 60, MemorySize: 2 * 1024 * 1024},
					},
				},
			},
			Level:         datapb.SegmentLevel_L0,
			StartPosition: &msgpb.MsgPosition{Timestamp: 30},
			DmlPosition:   &msgpb.MsgPosition{Timestamp: 40},
		},
	}
	growing_1 := &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:            2001,
			InsertChannel: "channel_1",
			StartPosition: &msgpb.MsgPosition{Timestamp: 10},
		},
	}
	growing_2 := &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:            2002,
			InsertChannel: "channel_1",
			StartPosition: &msgpb.MsgPosition{Timestamp: 35},
		},
	}
	growing_3 := &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:            2003,
			InsertChannel: "channel_1",
		},
	}

	testCases := []*testCase{
		{
			tag:             "seal_by_entrynum",
			channel:         "channel_1",
			sizeLimit:       -1,
			entryNumLimit:   100,
			l0Segments:      []*SegmentInfo{l0_1, l0_2},           // ts: [10,20] [30, 40], entryNum: 50, 60
			growingSegments: []*SegmentInfo{growing_1, growing_2}, // ts: [10, 35]
			expected:        []int64{2001},
		},
		{
			tag:             "seal_by_size",
			channel:         "channel_1",
			sizeLimit:       1, // 1MB
			entryNumLimit:   -1,
			l0Segments:      []*SegmentInfo{l0_1, l0_2},           // ts: [10,20] [30, 40], entryNum: 1MB, 2MB
			growingSegments: []*SegmentInfo{growing_1, growing_2}, // ts: [10, 35]
			expected:        []int64{2001, 2002},
		},
		{
			tag:             "empty_input",
			channel:         "channel_1",
			growingSegments: []*SegmentInfo{growing_1, growing_2},
			sizeLimit:       1,
			entryNumLimit:   50,
			expected:        []int64{},
		},
		{
			tag:             "all_disabled",
			channel:         "channel_1",
			l0Segments:      []*SegmentInfo{l0_1, l0_2},
			growingSegments: []*SegmentInfo{growing_1, growing_2},
			sizeLimit:       -1,
			entryNumLimit:   -1,
			expected:        []int64{},
		},
		{
			tag:             "growing_segment_with_nil_start_position",
			channel:         "channel_1",
			l0Segments:      []*SegmentInfo{l0_1, l0_2},
			growingSegments: []*SegmentInfo{growing_3},
			expected:        []int64{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.tag, func(t *testing.T) {
			pt.Save(pt.DataCoordCfg.BlockingL0SizeInMB.Key, strconv.FormatInt(tc.sizeLimit, 10))
			defer pt.Reset(pt.DataCoordCfg.BlockingL0SizeInMB.Key)
			pt.Save(pt.DataCoordCfg.BlockingL0EntryNum.Key, strconv.FormatInt(tc.entryNumLimit, 10))
			defer pt.Reset(pt.DataCoordCfg.BlockingL0EntryNum.Key)

			segments := NewSegmentsInfo()
			for _, l0segment := range tc.l0Segments {
				segments.SetSegment(l0segment.GetID(), l0segment)
			}

			meta := &meta{
				segments: segments,
			}

			result, _ := sealByBlockingL0(meta)(tc.channel, tc.growingSegments, 0)
			sealedIDs := lo.Map(result, func(segment *SegmentInfo, _ int) int64 {
				return segment.GetID()
			})
			assert.ElementsMatch(t, tc.expected, sealedIDs)
		})
	}
}
