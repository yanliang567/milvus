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

package datanode

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/bits-and-blooms/bloom/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/storage"
)

func TestNewReplica(t *testing.T) {
	rc := &RootCoordFactory{}
	replica, err := newReplica(context.Background(), rc, 0)
	assert.Nil(t, err)
	assert.NotNil(t, replica)
}

type mockMinioKV struct {
	kv.BaseKV
}

func (kv *mockMinioKV) MultiLoad(keys []string) ([]string, error) {
	stats := &storage.Int64Stats{
		FieldID: common.RowIDField,
		Min:     0,
		Max:     10,
		BF:      bloom.NewWithEstimates(bloomFilterSize, maxBloomFalsePositive),
	}
	buffer, _ := json.Marshal(stats)
	return []string{string(buffer)}, nil
}

type mockPkfilterMergeError struct {
	kv.BaseKV
}

func (kv *mockPkfilterMergeError) MultiLoad(keys []string) ([]string, error) {
	stats := &storage.Int64Stats{
		FieldID: common.RowIDField,
		Min:     0,
		Max:     10,
		BF:      bloom.NewWithEstimates(1, 0.0001),
	}
	buffer, _ := json.Marshal(stats)
	return []string{string(buffer)}, nil
}

type mockMinioKVError struct {
	kv.BaseKV
}

func (kv *mockMinioKVError) MultiLoad(keys []string) ([]string, error) {
	return nil, fmt.Errorf("mock error")
}

type mockMinioKVStatsError struct {
	kv.BaseKV
}

func (kv *mockMinioKVStatsError) MultiLoad(keys []string) ([]string, error) {
	return []string{"3123123,error,test"}, nil
}

func getSimpleFieldBinlog() *datapb.FieldBinlog {
	return &datapb.FieldBinlog{
		FieldID: 106,
		Binlogs: []string{"test"},
	}
}

func TestSegmentReplica_getCollectionAndPartitionID(te *testing.T) {
	tests := []struct {
		segInNew     UniqueID
		segInNormal  UniqueID
		segInFlushed UniqueID

		inCollID    UniqueID
		inParID     UniqueID
		description string
	}{
		{100, 0, 0, 1, 10, "Segment 100 in NewSegments"},
		{0, 200, 0, 2, 20, "Segment 200 in NormalSegments"},
		{0, 0, 300, 3, 30, "Segment 300 in FlushedSegments"},
		{0, 0, 0, 4, 40, "No Segment in replica"},
	}

	for _, test := range tests {
		te.Run(test.description, func(t *testing.T) {
			if test.segInNew != 0 {
				sr := &SegmentReplica{
					newSegments: map[UniqueID]*Segment{
						test.segInNew: {
							collectionID: test.inCollID,
							partitionID:  test.inParID,
							segmentID:    test.segInNew,
						}},
				}

				collID, parID, err := sr.getCollectionAndPartitionID(test.segInNew)
				assert.NoError(t, err)
				assert.Equal(t, test.inCollID, collID)
				assert.Equal(t, test.inParID, parID)
			} else if test.segInNormal != 0 {
				sr := &SegmentReplica{
					normalSegments: map[UniqueID]*Segment{
						test.segInNormal: {
							collectionID: test.inCollID,
							partitionID:  test.inParID,
							segmentID:    test.segInNormal,
						}},
				}

				collID, parID, err := sr.getCollectionAndPartitionID(test.segInNormal)
				assert.NoError(t, err)
				assert.Equal(t, test.inCollID, collID)
				assert.Equal(t, test.inParID, parID)
			} else if test.segInFlushed != 0 {
				sr := &SegmentReplica{
					flushedSegments: map[UniqueID]*Segment{
						test.segInFlushed: {
							collectionID: test.inCollID,
							partitionID:  test.inParID,
							segmentID:    test.segInFlushed,
						}},
				}

				collID, parID, err := sr.getCollectionAndPartitionID(test.segInFlushed)
				assert.NoError(t, err)
				assert.Equal(t, test.inCollID, collID)
				assert.Equal(t, test.inParID, parID)
			} else {
				sr := &SegmentReplica{}
				collID, parID, err := sr.getCollectionAndPartitionID(1000)
				assert.Error(t, err)
				assert.Zero(t, collID)
				assert.Zero(t, parID)
			}
		})
	}
}

func TestSegmentReplica(t *testing.T) {
	rc := &RootCoordFactory{}
	collID := UniqueID(1)

	t.Run("Test coll mot match", func(t *testing.T) {
		replica, err := newReplica(context.Background(), rc, collID)
		assert.Nil(t, err)

		err = replica.addNewSegment(1, collID+1, 0, "", nil, nil)
		assert.NotNil(t, err)
	})

	t.Run("Test segmentFlushed", func(t *testing.T) {
		testReplica := &SegmentReplica{
			newSegments:     make(map[UniqueID]*Segment),
			normalSegments:  make(map[UniqueID]*Segment),
			flushedSegments: make(map[UniqueID]*Segment),
		}

		type Test struct {
			inisNew     bool
			inisFlushed bool
			inSegID     UniqueID

			expectedisNew     bool
			expectedisFlushed bool
			expectedSegID     UniqueID
		}

		tests := []Test{
			// new segment
			{true, false, 1, false, true, 1},
			{true, false, 2, false, true, 2},
			{true, false, 3, false, true, 3},
			// normal segment
			{false, false, 10, false, true, 10},
			{false, false, 20, false, true, 20},
			{false, false, 30, false, true, 30},
			// flushed segment
			{false, true, 100, false, true, 100},
			{false, true, 200, false, true, 200},
			{false, true, 300, false, true, 300},
		}

		newSeg := func(sr *SegmentReplica, isNew, isFlushed bool, id UniqueID) {
			ns := &Segment{segmentID: id}
			ns.isNew.Store(isNew)
			ns.isFlushed.Store(isFlushed)

			if isNew && !isFlushed {
				sr.newSegments[id] = ns
				return
			}

			if !isNew && !isFlushed {
				sr.normalSegments[id] = ns
				return
			}

			if !isNew && isFlushed {
				sr.flushedSegments[id] = ns
				return
			}
		}

		for _, te := range tests {
			// prepare case
			newSeg(testReplica, te.inisNew, te.inisFlushed, te.inSegID)

			testReplica.segmentFlushed(te.inSegID)

			flushedSeg := testReplica.flushedSegments[te.inSegID]
			assert.Equal(t, te.expectedSegID, flushedSeg.segmentID)
			assert.Equal(t, te.expectedisNew, flushedSeg.isNew.Load().(bool))
			assert.Equal(t, te.expectedisFlushed, flushedSeg.isFlushed.Load().(bool))
		}

	})
}

func TestSegmentReplica_InterfaceMethod(t *testing.T) {
	rc := &RootCoordFactory{}

	t.Run("Test refreshFlushedSegmentPKRange", func(t *testing.T) {
		replica, err := newReplica(context.TODO(), rc, 1)
		require.NoError(t, err)

		require.False(t, replica.hasSegment(100, true))
		replica.refreshFlushedSegmentPKRange(100, []int64{10})

		replica.addFlushedSegmentWithPKs(100, 1, 10, "a", 1, []int64{9})
		require.True(t, replica.hasSegment(100, true))
		replica.refreshFlushedSegmentPKRange(100, []int64{10})

	})

	t.Run("Test addFlushedSegmentWithPKs", func(t *testing.T) {
		tests := []struct {
			isvalid bool

			incollID      UniqueID
			replicaCollID UniqueID
			description   string
		}{
			{true, 1, 1, "valid input collection with replica collection"},
			{false, 1, 2, "invalid input collection with replica collection"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				replica, err := newReplica(context.TODO(), rc, test.replicaCollID)
				require.NoError(t, err)
				if test.isvalid {
					replica.addFlushedSegmentWithPKs(100, test.incollID, 10, "a", 1, []int64{9})

					assert.True(t, replica.hasSegment(100, true))
					assert.False(t, replica.hasSegment(100, false))
				} else {
					replica.addFlushedSegmentWithPKs(100, test.incollID, 10, "a", 1, []int64{9})
					assert.False(t, replica.hasSegment(100, true))
					assert.False(t, replica.hasSegment(100, false))
				}
			})
		}
	})

	t.Run("Test_addNewSegment", func(t *testing.T) {
		tests := []struct {
			isValidCase   bool
			replicaCollID UniqueID
			inCollID      UniqueID
			inSegID       UniqueID

			instartPos *internalpb.MsgPosition

			expectdIsNew      bool
			expectedIsFlushed bool

			description string
		}{
			{isValidCase: false, replicaCollID: 1, inCollID: 2, inSegID: 300, description: "input CollID 2 mismatch with Replica collID"},
			{true, 1, 1, 200, new(internalpb.MsgPosition), true, false, "nill address for startPos"},
			{true, 1, 1, 200, &internalpb.MsgPosition{}, true, false, "empty struct for startPos"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr, err := newReplica(context.Background(), rc, test.replicaCollID)
				assert.Nil(t, err)
				require.False(t, sr.hasSegment(test.inSegID, true))
				err = sr.addNewSegment(test.inSegID,
					test.inCollID, 1, "", test.instartPos, &internalpb.MsgPosition{})
				if test.isValidCase {
					assert.NoError(t, err)
					assert.True(t, sr.hasSegment(test.inSegID, true))
					assert.Equal(t, test.expectdIsNew, sr.newSegments[test.inSegID].isNew.Load().(bool))
					assert.Equal(t, test.expectedIsFlushed, sr.newSegments[test.inSegID].isFlushed.Load().(bool))
				} else {
					assert.Error(t, err)
					assert.False(t, sr.hasSegment(test.inSegID, true))
				}
			})
		}
	})

	t.Run("Test_addNormalSegment", func(t *testing.T) {
		tests := []struct {
			isValidCase   bool
			replicaCollID UniqueID
			inCollID      UniqueID
			inSegID       UniqueID

			expectdIsNew      bool
			expectedIsFlushed bool

			description string
		}{
			{isValidCase: false, replicaCollID: 1, inCollID: 2, inSegID: 300, description: "input CollID 2 mismatch with Replica collID"},
			{true, 1, 1, 200, false, false, "normal case"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr, err := newReplica(context.Background(), rc, test.replicaCollID)
				sr.minIOKV = &mockMinioKV{}
				assert.Nil(t, err)
				require.False(t, sr.hasSegment(test.inSegID, true))
				err = sr.addNormalSegment(test.inSegID, test.inCollID, 1, "", 0, []*datapb.FieldBinlog{getSimpleFieldBinlog()}, &segmentCheckPoint{})
				if test.isValidCase {
					assert.NoError(t, err)
					assert.True(t, sr.hasSegment(test.inSegID, true))
					assert.Equal(t, test.expectdIsNew, sr.normalSegments[test.inSegID].isNew.Load().(bool))
					assert.Equal(t, test.expectedIsFlushed, sr.normalSegments[test.inSegID].isFlushed.Load().(bool))
				} else {
					assert.Error(t, err)
					assert.False(t, sr.hasSegment(test.inSegID, true))
				}
			})
		}
	})

	t.Run("Test_listSegmentsCheckPoints", func(t *testing.T) {
		tests := []struct {
			newSegID UniqueID
			newSegCP *segmentCheckPoint

			normalSegID UniqueID
			normalSegCP *segmentCheckPoint

			flushedSegID UniqueID
			flushedSegCP *segmentCheckPoint

			description string
		}{
			{newSegID: 100, newSegCP: new(segmentCheckPoint),
				description: "Only contain new Seg 100"},
			{normalSegID: 200, normalSegCP: new(segmentCheckPoint),
				description: "Only contain normal Seg 200"},
			{flushedSegID: 300, flushedSegCP: new(segmentCheckPoint),
				description: "Only contain flushed Seg 300"},
			{100, new(segmentCheckPoint), 200, new(segmentCheckPoint), 0, new(segmentCheckPoint),
				"New seg 100 and normal seg 200"},
			{100, new(segmentCheckPoint), 0, new(segmentCheckPoint), 300, new(segmentCheckPoint),
				"New seg 100 and flushed seg 300"},
			{0, new(segmentCheckPoint), 200, new(segmentCheckPoint), 300, new(segmentCheckPoint),
				"Normal seg 200 and flushed seg 300"},
			{100, new(segmentCheckPoint), 200, new(segmentCheckPoint), 300, new(segmentCheckPoint),
				"New seg 100, normal seg 200 and flushed seg 300"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr := SegmentReplica{
					newSegments:     make(map[UniqueID]*Segment),
					normalSegments:  make(map[UniqueID]*Segment),
					flushedSegments: make(map[UniqueID]*Segment),
				}

				expectdCount := 0
				if test.newSegID != 0 {
					sr.newSegments[test.newSegID] = &Segment{checkPoint: *test.newSegCP}
					expectdCount++
				}
				if test.normalSegID != 0 {
					sr.normalSegments[test.normalSegID] = &Segment{checkPoint: *test.normalSegCP}
					expectdCount++
				}
				if test.flushedSegID != 0 {
					sr.flushedSegments[test.flushedSegID] = &Segment{checkPoint: *test.flushedSegCP}
				}

				scp := sr.listSegmentsCheckPoints()
				assert.Equal(t, expectdCount, len(scp))
			})
		}
	})

	t.Run("Test_updateSegmentEndPosition", func(t *testing.T) {
		tests := []struct {
			newSegID     UniqueID
			normalSegID  UniqueID
			flushedSegID UniqueID

			inSegID     UniqueID
			description string
		}{
			{newSegID: 100, inSegID: 100,
				description: "input seg 100 in newSegments"},
			{newSegID: 100, inSegID: 101,
				description: "input seg 101 not in newSegments"},
			{normalSegID: 200, inSegID: 200,
				description: "input seg 200 in normalSegments"},
			{normalSegID: 200, inSegID: 201,
				description: "input seg 201 not in normalSegments"},
			{flushedSegID: 300, inSegID: 300,
				description: "input seg 300 in flushedSegments"},
			{flushedSegID: 300, inSegID: 301,
				description: "input seg 301 not in flushedSegments"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr := SegmentReplica{
					newSegments:     make(map[UniqueID]*Segment),
					normalSegments:  make(map[UniqueID]*Segment),
					flushedSegments: make(map[UniqueID]*Segment),
				}

				if test.newSegID != 0 {
					sr.newSegments[test.newSegID] = &Segment{}
				}
				if test.normalSegID != 0 {
					sr.normalSegments[test.normalSegID] = &Segment{}
				}
				if test.flushedSegID != 0 {
					sr.flushedSegments[test.flushedSegID] = &Segment{}
				}
				sr.updateSegmentEndPosition(test.inSegID, new(internalpb.MsgPosition))
				sr.removeSegment(0)

			})
		}
	})

	t.Run("Test_updateStatistics", func(t *testing.T) {
		tests := []struct {
			isvalidCase bool

			newSegID     UniqueID
			normalSegID  UniqueID
			flushedSegID UniqueID

			inSegID     UniqueID
			inNumRows   int64
			description string
		}{
			{isvalidCase: true, newSegID: 100, inSegID: 100, inNumRows: 100,
				description: "input seg 100 in newSegments with numRows 100"},
			{isvalidCase: false, newSegID: 100, inSegID: 101, inNumRows: 100,
				description: "input seg 101 not in newSegments with numRows 100"},
			{isvalidCase: true, normalSegID: 200, inSegID: 200, inNumRows: 200,
				description: "input seg 200 in normalSegments with numRows 200"},
			{isvalidCase: false, normalSegID: 200, inSegID: 201, inNumRows: 200,
				description: "input seg 201 not in normalSegments with numRows 200"},
			{isvalidCase: true, flushedSegID: 300, inSegID: 300, inNumRows: 300,
				description: "input seg 300 in flushedSegments"},
			{isvalidCase: false, flushedSegID: 300, inSegID: 301, inNumRows: 300,
				description: "input seg 301 not in flushedSegments"},
		}
		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr := SegmentReplica{
					newSegments:     make(map[UniqueID]*Segment),
					normalSegments:  make(map[UniqueID]*Segment),
					flushedSegments: make(map[UniqueID]*Segment),
				}

				if test.newSegID != 0 {
					sr.newSegments[test.newSegID] = &Segment{}
				}
				if test.normalSegID != 0 {
					sr.normalSegments[test.normalSegID] = &Segment{}
				}
				if test.flushedSegID != 0 { // not update flushed num rows
					sr.flushedSegments[test.flushedSegID] = &Segment{
						numRows: test.inNumRows,
					}
				}

				sr.updateStatistics(test.inSegID, test.inNumRows)
				if test.isvalidCase {

					updates, err := sr.getSegmentStatisticsUpdates(test.inSegID)
					assert.NoError(t, err)
					assert.Equal(t, test.inNumRows, updates.GetNumRows())
					assert.Equal(t, test.inSegID, updates.GetSegmentID())

					sr.updateSegmentCheckPoint(10000)
				} else {
					updates, err := sr.getSegmentStatisticsUpdates(test.inSegID)
					assert.Error(t, err)
					assert.Nil(t, updates)
				}
			})
		}
	})

	t.Run("Test_getCollectionSchema", func(t *testing.T) {
		tests := []struct {
			isValid       bool
			replicaCollID UniqueID
			inputCollID   UniqueID

			metaServiceErr bool
			description    string
		}{
			{true, 1, 1, false, "Normal case"},
			{false, 1, 2, false, "Input collID 2 mismatch with replicaCollID 1"},
			{false, 1, 1, true, "RPC call fails"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				sr, err := newReplica(context.Background(), rc, test.replicaCollID)
				assert.Nil(t, err)

				if test.metaServiceErr {
					rc.setCollectionID(-1)
				} else {
					rc.setCollectionID(1)
				}

				s, err := sr.getCollectionSchema(test.inputCollID, Timestamp(0))
				if test.isValid {
					assert.NoError(t, err)
					assert.NotNil(t, s)
				} else {
					assert.Error(t, err)
					assert.Nil(t, s)
				}
			})
		}

	})

	t.Run("Test_addSegmentMinIOLoadError", func(t *testing.T) {
		sr, err := newReplica(context.Background(), rc, 1)
		assert.Nil(t, err)
		sr.minIOKV = &mockMinioKVError{}

		cpPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(10)}
		cp := &segmentCheckPoint{int64(10), *cpPos}
		err = sr.addNormalSegment(1, 1, 2, "insert-01", int64(10), []*datapb.FieldBinlog{getSimpleFieldBinlog()}, cp)
		assert.NotNil(t, err)
		err = sr.addFlushedSegment(1, 1, 2, "insert-01", int64(0), []*datapb.FieldBinlog{getSimpleFieldBinlog()})
		assert.NotNil(t, err)
	})

	t.Run("Test_addSegmentStatsError", func(t *testing.T) {
		sr, err := newReplica(context.Background(), rc, 1)
		assert.Nil(t, err)
		sr.minIOKV = &mockMinioKVStatsError{}

		cpPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(10)}
		cp := &segmentCheckPoint{int64(10), *cpPos}
		err = sr.addNormalSegment(1, 1, 2, "insert-01", int64(10), []*datapb.FieldBinlog{getSimpleFieldBinlog()}, cp)
		assert.NotNil(t, err)
		err = sr.addFlushedSegment(1, 1, 2, "insert-01", int64(0), []*datapb.FieldBinlog{getSimpleFieldBinlog()})
		assert.NotNil(t, err)
	})

	t.Run("Test_addSegmentPkfilterError", func(t *testing.T) {
		sr, err := newReplica(context.Background(), rc, 1)
		assert.Nil(t, err)
		sr.minIOKV = &mockPkfilterMergeError{}

		cpPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(10)}
		cp := &segmentCheckPoint{int64(10), *cpPos}
		err = sr.addNormalSegment(1, 1, 2, "insert-01", int64(10), []*datapb.FieldBinlog{getSimpleFieldBinlog()}, cp)
		assert.NotNil(t, err)
		err = sr.addFlushedSegment(1, 1, 2, "insert-01", int64(0), []*datapb.FieldBinlog{getSimpleFieldBinlog()})
		assert.NotNil(t, err)
	})

}
func TestInnerFunctionSegment(t *testing.T) {
	rc := &RootCoordFactory{}
	collID := UniqueID(1)
	replica, err := newReplica(context.Background(), rc, collID)
	assert.Nil(t, err)
	replica.minIOKV = &mockMinioKV{}
	assert.False(t, replica.hasSegment(0, true))
	assert.False(t, replica.hasSegment(0, false))

	startPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(100)}
	endPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(200)}
	err = replica.addNewSegment(0, 1, 2, "insert-01", startPos, endPos)
	assert.NoError(t, err)
	assert.True(t, replica.hasSegment(0, true))
	assert.Equal(t, 1, len(replica.newSegments))

	seg, ok := replica.newSegments[UniqueID(0)]
	assert.True(t, ok)
	require.NotNil(t, seg)
	assert.Equal(t, UniqueID(0), seg.segmentID)
	assert.Equal(t, UniqueID(1), seg.collectionID)
	assert.Equal(t, UniqueID(2), seg.partitionID)
	assert.Equal(t, "insert-01", seg.channelName)
	assert.Equal(t, Timestamp(100), seg.startPos.Timestamp)
	assert.Equal(t, Timestamp(200), seg.endPos.Timestamp)
	assert.Equal(t, startPos.ChannelName, seg.checkPoint.pos.ChannelName)
	assert.Equal(t, startPos.Timestamp, seg.checkPoint.pos.Timestamp)
	assert.Equal(t, int64(0), seg.numRows)
	assert.True(t, seg.isNew.Load().(bool))
	assert.False(t, seg.isFlushed.Load().(bool))

	replica.updateStatistics(0, 10)
	assert.Equal(t, int64(10), seg.numRows)

	cpPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(10)}
	cp := &segmentCheckPoint{int64(10), *cpPos}
	err = replica.addNormalSegment(1, 1, 2, "insert-01", int64(10), []*datapb.FieldBinlog{getSimpleFieldBinlog()}, cp)
	assert.NoError(t, err)
	assert.True(t, replica.hasSegment(1, true))
	assert.Equal(t, 1, len(replica.normalSegments))
	seg, ok = replica.normalSegments[UniqueID(1)]
	assert.True(t, ok)
	require.NotNil(t, seg)
	assert.Equal(t, UniqueID(1), seg.segmentID)
	assert.Equal(t, UniqueID(1), seg.collectionID)
	assert.Equal(t, UniqueID(2), seg.partitionID)
	assert.Equal(t, "insert-01", seg.channelName)
	assert.Equal(t, cpPos.ChannelName, seg.checkPoint.pos.ChannelName)
	assert.Equal(t, cpPos.Timestamp, seg.checkPoint.pos.Timestamp)
	assert.Equal(t, int64(10), seg.numRows)
	assert.False(t, seg.isNew.Load().(bool))
	assert.False(t, seg.isFlushed.Load().(bool))

	err = replica.addNormalSegment(1, 100000, 2, "invalid", int64(0), []*datapb.FieldBinlog{getSimpleFieldBinlog()}, &segmentCheckPoint{})
	assert.Error(t, err)

	replica.updateStatistics(1, 10)
	assert.Equal(t, int64(20), seg.numRows)

	segPos := replica.listNewSegmentsStartPositions()
	assert.Equal(t, 1, len(segPos))
	assert.Equal(t, UniqueID(0), segPos[0].SegmentID)
	assert.Equal(t, "insert-01", segPos[0].StartPosition.ChannelName)
	assert.Equal(t, Timestamp(100), segPos[0].StartPosition.Timestamp)

	assert.Equal(t, 0, len(replica.newSegments))
	assert.Equal(t, 2, len(replica.normalSegments))

	cps := replica.listSegmentsCheckPoints()
	assert.Equal(t, 2, len(cps))
	assert.Equal(t, startPos.Timestamp, cps[UniqueID(0)].pos.Timestamp)
	assert.Equal(t, int64(0), cps[UniqueID(0)].numRows)
	assert.Equal(t, cp.pos.Timestamp, cps[UniqueID(1)].pos.Timestamp)
	assert.Equal(t, int64(10), cps[UniqueID(1)].numRows)

	updates, err := replica.getSegmentStatisticsUpdates(0)
	assert.NoError(t, err)
	assert.Equal(t, int64(10), updates.NumRows)

	updates, err = replica.getSegmentStatisticsUpdates(1)
	assert.NoError(t, err)
	assert.Equal(t, int64(20), updates.NumRows)

	replica.updateSegmentCheckPoint(0)
	assert.Equal(t, int64(10), replica.normalSegments[UniqueID(0)].checkPoint.numRows)
	replica.updateSegmentCheckPoint(1)
	assert.Equal(t, int64(20), replica.normalSegments[UniqueID(1)].checkPoint.numRows)

	err = replica.addFlushedSegment(1, 1, 2, "insert-01", int64(0), []*datapb.FieldBinlog{getSimpleFieldBinlog()})
	assert.Nil(t, err)

	totalSegments := replica.filterSegments("insert-01", common.InvalidPartitionID)
	assert.Equal(t, len(totalSegments), 3)
}

func TestSegmentReplica_UpdatePKRange(t *testing.T) {
	seg := &Segment{
		pkFilter: bloom.NewWithEstimates(100000, 0.005),
		maxPK:    math.MinInt64,
		minPK:    math.MaxInt64,
	}

	cases := make([]int64, 0, 100)
	for i := 0; i < 100; i++ {
		cases = append(cases, rand.Int63())
	}
	buf := make([]byte, 8)
	for _, c := range cases {
		seg.updatePKRange([]int64{c})

		assert.LessOrEqual(t, seg.minPK, c)
		assert.GreaterOrEqual(t, seg.maxPK, c)

		common.Endian.PutUint64(buf, uint64(c))
		assert.True(t, seg.pkFilter.Test(buf))
	}
}

func TestReplica_UpdatePKRange(t *testing.T) {
	rc := &RootCoordFactory{}
	collID := UniqueID(1)
	partID := UniqueID(2)
	chanName := "insert-02"
	startPos := &internalpb.MsgPosition{ChannelName: chanName, Timestamp: Timestamp(100)}
	endPos := &internalpb.MsgPosition{ChannelName: chanName, Timestamp: Timestamp(200)}
	cpPos := &internalpb.MsgPosition{ChannelName: chanName, Timestamp: Timestamp(10)}
	cp := &segmentCheckPoint{int64(10), *cpPos}

	replica, err := newReplica(context.Background(), rc, collID)
	assert.Nil(t, err)
	replica.minIOKV = &mockMinioKV{}

	err = replica.addNewSegment(1, collID, partID, chanName, startPos, endPos)
	assert.Nil(t, err)
	err = replica.addNormalSegment(2, collID, partID, chanName, 100, []*datapb.FieldBinlog{getSimpleFieldBinlog()}, cp)
	assert.Nil(t, err)

	segNew := replica.newSegments[1]
	segNormal := replica.normalSegments[2]

	cases := make([]int64, 0, 100)
	for i := 0; i < 100; i++ {
		cases = append(cases, rand.Int63())
	}
	buf := make([]byte, 8)
	for _, c := range cases {
		replica.updateSegmentPKRange(1, []int64{c}) // new segment
		replica.updateSegmentPKRange(2, []int64{c}) // normal segment
		replica.updateSegmentPKRange(3, []int64{c}) // non-exist segment

		assert.LessOrEqual(t, segNew.minPK, c)
		assert.GreaterOrEqual(t, segNew.maxPK, c)
		assert.LessOrEqual(t, segNormal.minPK, c)
		assert.GreaterOrEqual(t, segNormal.maxPK, c)

		common.Endian.PutUint64(buf, uint64(c))
		assert.True(t, segNew.pkFilter.Test(buf))
		assert.True(t, segNormal.pkFilter.Test(buf))

	}

}
