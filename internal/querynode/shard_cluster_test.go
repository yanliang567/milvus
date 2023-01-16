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
	"errors"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type mockNodeDetector struct {
	initNodes []nodeEvent
	evtCh     chan nodeEvent
}

func (m *mockNodeDetector) watchNodes(collectionID int64, replicaID int64, vchannelName string) ([]nodeEvent, <-chan nodeEvent) {
	return m.initNodes, m.evtCh
}

func (m *mockNodeDetector) Close() {}

type mockSegmentDetector struct {
	initSegments []segmentEvent
	evtCh        chan segmentEvent
}

func (m *mockSegmentDetector) watchSegments(collectionID int64, replicaID int64, vchannelName string) ([]segmentEvent, <-chan segmentEvent) {
	return m.initSegments, m.evtCh
}

func (m *mockSegmentDetector) Close() {}

type mockShardQueryNode struct {
	statisticResponse     *internalpb.GetStatisticsResponse
	statisticErr          error
	searchResult          *internalpb.SearchResults
	searchErr             error
	queryResult           *internalpb.RetrieveResults
	queryErr              error
	loadSegmentsResults   *commonpb.Status
	loadSegmentsErr       error
	releaseSegmentsResult *commonpb.Status
	releaseSegmentsErr    error
}

func (m *mockShardQueryNode) GetStatistics(ctx context.Context, req *querypb.GetStatisticsRequest) (*internalpb.GetStatisticsResponse, error) {
	return m.statisticResponse, m.statisticErr
}

func (m *mockShardQueryNode) Search(_ context.Context, _ *querypb.SearchRequest) (*internalpb.SearchResults, error) {
	return m.searchResult, m.searchErr
}

func (m *mockShardQueryNode) Query(_ context.Context, _ *querypb.QueryRequest) (*internalpb.RetrieveResults, error) {
	return m.queryResult, m.queryErr
}

func (m *mockShardQueryNode) LoadSegments(ctx context.Context, in *querypb.LoadSegmentsRequest) (*commonpb.Status, error) {
	return m.loadSegmentsResults, m.loadSegmentsErr
}

func (m *mockShardQueryNode) ReleaseSegments(ctx context.Context, in *querypb.ReleaseSegmentsRequest) (*commonpb.Status, error) {
	return m.releaseSegmentsResult, m.releaseSegmentsErr
}

func (m *mockShardQueryNode) Stop() error {
	return nil
}

func buildMockQueryNode(nodeID int64, addr string) shardQueryNode {
	return &mockShardQueryNode{
		statisticResponse: &internalpb.GetStatisticsResponse{
			Stats: []*commonpb.KeyValuePair{
				{
					Key:   "row_count",
					Value: "0",
				},
			},
		},
		searchResult: &internalpb.SearchResults{},
		queryResult:  &internalpb.RetrieveResults{},
	}
}

func segmentEventsToSyncInfo(events []segmentEvent) []*querypb.ReplicaSegmentsInfo {
	infos := make([]*querypb.ReplicaSegmentsInfo, 0, len(events))
	for _, event := range events {
		for _, nodeID := range event.nodeIDs {
			infos = append(infos, &querypb.ReplicaSegmentsInfo{
				NodeId:      nodeID,
				SegmentIds:  []int64{event.segmentID},
				PartitionId: event.partitionID,
				Versions:    []int64{0},
			})
		}
	}

	return infos
}

func TestShardCluster_Create(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)

	t.Run("empty shard cluster", func(t *testing.T) {
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{}, &mockSegmentDetector{}, buildMockQueryNode)
		assert.NotPanics(t, func() { sc.Close() })
		// close twice
		assert.NotPanics(t, func() { sc.Close() })
	})

	t.Run("init nodes", func(t *testing.T) {
		nodeEvent := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
				isLeader: true,
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvent,
			}, &mockSegmentDetector{}, buildMockQueryNode)
		defer sc.Close()

		for _, e := range nodeEvent {
			node, has := sc.getNode(e.nodeID)
			assert.True(t, has)
			assert.Equal(t, e.nodeAddr, node.nodeAddr)
		}
		sc.mut.Lock()
		defer sc.mut.Unlock()

		require.NotNil(t, sc.leader)
		assert.Equal(t, int64(1), sc.leader.nodeID)
	})

	t.Run("init segments", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoading,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)
		defer sc.Close()

		for _, e := range segmentEvents {
			sc.mut.RLock()
			segment, has := sc.segments[e.segmentID]
			_, inCluster := sc.pickNode(e)
			sc.mut.RUnlock()
			if inCluster {
				assert.True(t, has)
				assert.Equal(t, e.segmentID, segment.segmentID)
				assert.Contains(t, e.nodeIDs, segment.nodeID)
				assert.Equal(t, e.state, segment.state)
			} else {
				assert.False(t, has)
			}
		}
		assert.EqualValues(t, unavailable, sc.state.Load())
	})
}

func TestShardCluster_nodeEvent(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)

	t.Run("only nodes", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}
		evtCh := make(chan nodeEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
				evtCh:     evtCh,
			}, &mockSegmentDetector{}, buildMockQueryNode)
		defer sc.Close()

		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_3",
			eventType: nodeAdd,
		}
		// same event
		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_3",
			eventType: nodeAdd,
		}

		assert.Eventually(t, func() bool {
			node, has := sc.getNode(3)
			return has && node.nodeAddr == "addr_3"
		}, time.Second, time.Millisecond)

		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_new",
			eventType: nodeAdd,
		}
		assert.Eventually(t, func() bool {
			node, has := sc.getNode(3)
			return has && node.nodeAddr == "addr_new"
		}, time.Second, time.Millisecond)

		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_new",
			eventType: nodeDel,
		}
		assert.Eventually(t, func() bool {
			_, has := sc.getNode(3)
			return !has
		}, time.Second, time.Millisecond)
		assert.Equal(t, int32(available), sc.state.Load())

		evtCh <- nodeEvent{
			nodeID:    4,
			nodeAddr:  "addr_new",
			eventType: nodeDel,
		}

		close(evtCh)
	})

	t.Run("with segments", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}
		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoading,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
		}

		evtCh := make(chan nodeEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
				evtCh:     evtCh,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)
		defer sc.Close()

		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_3",
			eventType: nodeAdd,
		}

		assert.Eventually(t, func() bool {
			node, has := sc.getNode(3)
			return has && node.nodeAddr == "addr_3"
		}, time.Second, time.Millisecond)

		evtCh <- nodeEvent{
			nodeID:    3,
			nodeAddr:  "addr_new",
			eventType: nodeAdd,
		}
		assert.Eventually(t, func() bool {
			node, has := sc.getNode(3)
			return has && node.nodeAddr == "addr_new"
		}, time.Second, time.Millisecond)

		// remove node 2
		evtCh <- nodeEvent{
			nodeID:    2,
			nodeAddr:  "addr_2",
			eventType: nodeDel,
		}
		assert.Eventually(t, func() bool {
			_, has := sc.getNode(2)
			return !has
		}, time.Second, time.Millisecond)
		assert.Equal(t, int32(unavailable), sc.state.Load())

		segment, has := sc.getSegment(2)
		assert.True(t, has)
		assert.Equal(t, segmentStateOffline, segment.state)

		close(evtCh)
	})
}

func TestShardCluster_segmentEvent(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)

	t.Run("from loading", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoading,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoading,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateLoading,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{1},
			state:     segmentStateLoading,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 3,
			nodeIDs:   []int64{3},
			state:     segmentStateOffline,
			eventType: segmentAdd,
		}
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == 3 && seg.state == segmentStateOffline
		}, time.Second, time.Millisecond)
		// put this check behind other make sure event is processed
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoading
		}, time.Second, time.Millisecond)

		// node id not match
		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}
		// will not change
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoading
		}, time.Second, time.Millisecond)
		close(evtCh)
	})

	t.Run("from loaded", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}
		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateLoaded,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SyncSegments(segmentEventsToSyncInfo(segmentEvents), segmentStateLoaded)

		// make reference greater than 0
		_, versionID := sc.segmentAllocations(nil)
		defer sc.finishUsage(versionID)

		evtCh <- segmentEvent{
			segmentID: 4,
			nodeIDs:   []int64{4},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}
		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{1},
			state:     segmentStateLoading,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 3,
			nodeIDs:   []int64{3},
			state:     segmentStateOffline,
			eventType: segmentAdd,
		}

		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoading
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == 3 && seg.state == segmentStateOffline
		}, time.Second, time.Millisecond)
		// put this check behind other make sure event is processed
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)

		_, has := sc.getSegment(4)
		assert.False(t, has)

	})

	t.Run("from loaded, node changed", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SyncSegments(segmentEventsToSyncInfo(segmentEvents), segmentStateLoaded)

		// make reference greater than 0
		_, versionID := sc.segmentAllocations(nil)
		defer sc.finishUsage(versionID)

		// bring segment online in the other querynode
		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{1},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)

		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)

	})

	t.Run("from offline", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateOffline,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SyncSegments(segmentEventsToSyncInfo(nil), segmentStateLoaded)

		evtCh <- segmentEvent{
			segmentID: 3,
			nodeIDs:   []int64{3},
			state:     segmentStateOffline,
			eventType: segmentAdd,
		}
		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{1},
			state:     segmentStateLoading,
			eventType: segmentAdd,
		}

		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}

		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoading
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		// put this check behind other make sure event is processed
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == 3 && seg.state == segmentStateOffline
		}, time.Second, time.Millisecond)

		close(evtCh)
	})

	t.Run("remove segments", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoading,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		evtCh <- segmentEvent{
			segmentID: 3,
			nodeIDs:   []int64{3},
			eventType: segmentDel,
		}
		evtCh <- segmentEvent{
			segmentID: 1,
			nodeIDs:   []int64{1},
			eventType: segmentDel,
		}
		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			eventType: segmentDel,
		}

		assert.Eventually(t, func() bool {
			_, has := sc.getSegment(1)
			return !has
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			_, has := sc.getSegment(2)
			return !has
		}, time.Second, time.Millisecond)
		// put this check behind other make sure event is processed
		assert.Eventually(t, func() bool {
			_, has := sc.getSegment(3)
			return !has
		}, time.Second, time.Millisecond)
	})

	t.Run("remove failed", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoading,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SyncSegments(segmentEventsToSyncInfo(nil), segmentStateLoaded)
		// non-exist segment
		evtCh <- segmentEvent{
			segmentID: 4,
			nodeIDs:   []int64{3},
			eventType: segmentDel,
		}
		// segment node id not match
		evtCh <- segmentEvent{
			segmentID: 3,
			nodeIDs:   []int64{4},
			eventType: segmentDel,
		}

		// use add segment as event process signal
		evtCh <- segmentEvent{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
			eventType: segmentAdd,
		}
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)

		_, has := sc.getSegment(3)
		assert.True(t, has)
	})
}

func TestShardCluster_SyncSegments(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)

	t.Run("sync new segments", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()
		sc.SetupFirstVersion()

		sc.SyncSegments([]*querypb.ReplicaSegmentsInfo{
			{
				NodeId:     1,
				SegmentIds: []int64{1},
				Versions:   []int64{1},
			},
			{
				NodeId:     2,
				SegmentIds: []int64{2},
				Versions:   []int64{1},
			},
			{
				NodeId:     3,
				SegmentIds: []int64{3},
				Versions:   []int64{1},
			},
		}, segmentStateLoaded)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == 3 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)

	})

	t.Run("sync existing segments", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateOffline,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()
		sc.SetupFirstVersion()

		sc.SyncSegments([]*querypb.ReplicaSegmentsInfo{
			{
				NodeId:     1,
				SegmentIds: []int64{1},
				Versions:   []int64{1},
			},
			{
				NodeId:     2,
				SegmentIds: []int64{2},
				Versions:   []int64{1},
			},
			{
				NodeId:     3,
				SegmentIds: []int64{3},
				Versions:   []int64{1},
			},
		}, segmentStateLoaded)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == 1 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == 2 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == 3 && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
	})

	t.Run("sync segments with offline nodes", func(t *testing.T) {
		nodeEvents := []nodeEvent{}

		segmentEvents := []segmentEvent{}

		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()
		sc.SetupFirstVersion()

		sc.SyncSegments([]*querypb.ReplicaSegmentsInfo{
			{
				NodeId:     1,
				SegmentIds: []int64{1},
				Versions:   []int64{1},
			},
			{
				NodeId:     2,
				SegmentIds: []int64{2},
				Versions:   []int64{1},
			},
			{
				NodeId:     3,
				SegmentIds: []int64{3},
				Versions:   []int64{1},
			},
		}, segmentStateLoaded)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(1)
			return has && seg.nodeID == common.InvalidNodeID && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(2)
			return has && seg.nodeID == common.InvalidNodeID && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
		assert.Eventually(t, func() bool {
			seg, has := sc.getSegment(3)
			return has && seg.nodeID == common.InvalidNodeID && seg.state == segmentStateLoaded
		}, time.Second, time.Millisecond)
	})
}

var streamingDoNothing = func(context.Context) error { return nil }
var streamingError = func(context.Context) error { return errors.New("mock streaming error") }

func TestShardCluster_Search(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)
	ctx := context.Background()

	t.Run("search unavailable cluster", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateOffline,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		require.EqualValues(t, unavailable, sc.state.Load())

		_, err := sc.Search(ctx, &querypb.SearchRequest{
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("search wrong channel", func(t *testing.T) {
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{}, &mockSegmentDetector{}, buildMockQueryNode)

		defer sc.Close()

		_, err := sc.Search(ctx, &querypb.SearchRequest{
			DmlChannels: []string{vchannelName + "_suffix"},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("normal search", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		result, err := sc.Search(ctx, &querypb.SearchRequest{
			Req: &internalpb.SearchRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.NoError(t, err)
		assert.Equal(t, len(nodeEvents), len(result))
	})

	t.Run("with streaming fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Search(ctx, &querypb.SearchRequest{
			Req: &internalpb.SearchRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, func(ctx context.Context) error { return errors.New("mocked") })
		assert.Error(t, err)
	})

	t.Run("partial fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, func(nodeID int64, addr string) shardQueryNode {
				if nodeID != 2 { // hard code error one
					return buildMockQueryNode(nodeID, addr)
				}
				return &mockShardQueryNode{
					searchErr: errors.New("mocked error"),
					queryErr:  errors.New("mocked error"),
				}
			})

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Search(ctx, &querypb.SearchRequest{
			Req: &internalpb.SearchRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("test meta error", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}
		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)
		//mock meta error
		sc.mutVersion.Lock()
		sc.currentVersion.segments[3] = shardSegmentInfo{
			segmentID: 3,
			nodeID:    3, // node does not exist
			state:     segmentStateLoaded,
		}
		sc.mutVersion.Unlock()

		defer sc.Close()

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Search(ctx, &querypb.SearchRequest{
			Req: &internalpb.SearchRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})
}

func TestShardCluster_Query(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)
	ctx := context.Background()

	t.Run("query unavailable cluster", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateOffline,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, unavailable, sc.state.Load())

		_, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})
	t.Run("query wrong channel", func(t *testing.T) {
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{}, &mockSegmentDetector{}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()

		_, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName + "_suffix"},
		}, streamingDoNothing)
		assert.Error(t, err)
	})
	t.Run("normal query", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		result, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.NoError(t, err)
		assert.Equal(t, len(nodeEvents), len(result))
	})
	t.Run("with streaming fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, func(ctx context.Context) error { return errors.New("mocked") })
		assert.Error(t, err)
	})

	t.Run("partial fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, func(nodeID int64, addr string) shardQueryNode {
				if nodeID != 2 { // hard code error one
					return buildMockQueryNode(nodeID, addr)
				}
				return &mockShardQueryNode{
					searchErr: errors.New("mocked error"),
					queryErr:  errors.New("mocked error"),
				}
			})

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})
	t.Run("test meta error", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)
		//mock meta error
		sc.mutVersion.Lock()
		sc.currentVersion.segments[3] = shardSegmentInfo{
			segmentID: 3,
			nodeID:    3, // node does not exist
			state:     segmentStateLoaded,
		}
		sc.mutVersion.Unlock()

		defer sc.Close()

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.Query(ctx, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{
				Base: &commonpb.MsgBase{},
			},
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

}

func TestShardCluster_GetStatistics(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)
	ctx := context.Background()

	t.Run("get statistics on unavailable cluster", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
			{
				nodeID:   3,
				nodeAddr: "addr_3",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateOffline,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateOffline,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{3},
				state:     segmentStateOffline,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{initNodes: nodeEvents}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		require.EqualValues(t, unavailable, sc.state.Load())

		_, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("get statistics on wrong channel", func(t *testing.T) {
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{}, &mockSegmentDetector{}, buildMockQueryNode)

		defer sc.Close()

		_, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName + "_suffix"},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("normal get statistics", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		result, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.NoError(t, err)
		assert.Equal(t, len(nodeEvents), len(result))
	})

	t.Run("with streaming fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName},
		}, func(ctx context.Context) error { return errors.New("mocked") })
		assert.Error(t, err)
	})

	t.Run("partial fail", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 3,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, func(nodeID int64, addr string) shardQueryNode {
				if nodeID != 2 { // hard code error one
					return buildMockQueryNode(nodeID, addr)
				}
				return &mockShardQueryNode{
					statisticErr: errors.New("mocked error"),
					searchErr:    errors.New("mocked error"),
					queryErr:     errors.New("mocked error"),
				}
			})

		defer sc.Close()
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})

	t.Run("test meta error", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}
		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}

		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)
		// setup first version
		sc.SetupFirstVersion()
		setupSegmentForShardCluster(sc, segmentEvents)
		//mock meta error
		sc.mutVersion.Lock()
		sc.currentVersion.segments[3] = shardSegmentInfo{
			segmentID: 3,
			nodeID:    3, // node does not exist
			state:     segmentStateLoaded,
		}
		sc.mutVersion.Unlock()
		defer sc.Close()

		require.EqualValues(t, available, sc.state.Load())

		_, err := sc.GetStatistics(ctx, &querypb.GetStatisticsRequest{
			DmlChannels: []string{vchannelName},
		}, streamingDoNothing)
		assert.Error(t, err)
	})
}

func TestShardCluster_Version(t *testing.T) {
	collectionID := int64(1)
	vchannelName := "dml_1_1_v0"
	replicaID := int64(0)
	version := int64(1)
	//	ctx := context.Background()
	t.Run("alloc with non-serviceable", func(t *testing.T) {
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{}, &mockSegmentDetector{}, buildMockQueryNode)
		defer sc.Close()

		_, v := sc.segmentAllocations(nil)
		assert.Equal(t, int64(0), v)
	})

	t.Run("normal alloc & finish", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SetupFirstVersion()
		_, version := sc.segmentAllocations(nil)

		sc.mut.RLock()
		assert.Equal(t, version, sc.currentVersion.versionID)
		assert.Equal(t, int64(1), sc.currentVersion.inUse.Load())
		sc.mut.RUnlock()

		sc.finishUsage(version)
		sc.mut.RLock()

		assert.Equal(t, int64(0), sc.currentVersion.inUse.Load())
		sc.mut.RUnlock()
	})

	t.Run("wait segments online", func(t *testing.T) {
		nodeEvents := []nodeEvent{
			{
				nodeID:   1,
				nodeAddr: "addr_1",
			},
			{
				nodeID:   2,
				nodeAddr: "addr_2",
			},
		}

		segmentEvents := []segmentEvent{
			{
				segmentID: 1,
				nodeIDs:   []int64{1},
				state:     segmentStateLoaded,
			},
			{
				segmentID: 2,
				nodeIDs:   []int64{2},
				state:     segmentStateLoaded,
			},
		}
		evtCh := make(chan segmentEvent, 10)
		sc := NewShardCluster(collectionID, replicaID, vchannelName, version,
			&mockNodeDetector{
				initNodes: nodeEvents,
			}, &mockSegmentDetector{
				initSegments: segmentEvents,
				evtCh:        evtCh,
			}, buildMockQueryNode)
		defer sc.Close()

		sc.SetupFirstVersion()

		assert.True(t, sc.segmentsOnline([]shardSegmentInfo{{nodeID: 1, segmentID: 1}, {nodeID: 2, segmentID: 2}}))
		assert.False(t, sc.segmentsOnline([]shardSegmentInfo{{nodeID: 1, segmentID: 1}, {nodeID: 2, segmentID: 2}, {nodeID: 1, segmentID: 3}}))

		sig := make(chan struct{})
		go func() {
			sc.waitSegmentsOnline([]shardSegmentInfo{{nodeID: 1, segmentID: 1}, {nodeID: 2, segmentID: 2}, {nodeID: 1, segmentID: 3}})
			close(sig)
		}()

		evtCh <- segmentEvent{
			eventType: segmentAdd,
			segmentID: 3,
			nodeIDs:   []int64{1, 4},
			state:     segmentStateLoaded,
		}

		<-sig
		assert.True(t, sc.segmentsOnline([]shardSegmentInfo{{nodeID: 1, segmentID: 1}, {nodeID: 2, segmentID: 2}, {nodeID: 1, segmentID: 3}}))
	})
}

func setupSegmentForShardCluster(sc *ShardCluster, segmentEvents []segmentEvent) {
	for _, evt := range segmentEvents {
		sc.SyncSegments([]*querypb.ReplicaSegmentsInfo{
			{
				NodeId:      evt.nodeIDs[0],
				PartitionId: evt.partitionID,
				SegmentIds:  []int64{evt.segmentID},
				Versions:    []int64{0},
			},
		}, evt.state)
	}
}

type ShardClusterSuite struct {
	suite.Suite

	collectionID      int64
	otherCollectionID int64
	vchannelName      string
	otherVchannelName string

	replicaID int64
	version   int64

	sc *ShardCluster
}

func (suite *ShardClusterSuite) SetupSuite() {
	suite.collectionID = 1
	suite.otherCollectionID = 2
	suite.vchannelName = "dml_1_1_v0"
	suite.otherVchannelName = "dml_1_2_v0"
	suite.replicaID = 0
	suite.version = 1
}

func (suite *ShardClusterSuite) SetupTest() {
	nodeEvents := []nodeEvent{
		{
			nodeID:   1,
			nodeAddr: "addr_1",
			isLeader: true,
		},
		{
			nodeID:   2,
			nodeAddr: "addr_2",
		},
	}

	segmentEvents := []segmentEvent{
		{
			segmentID: 1,
			nodeIDs:   []int64{1},
			state:     segmentStateLoaded,
		},
		{
			segmentID: 2,
			nodeIDs:   []int64{2},
			state:     segmentStateLoaded,
		},
	}
	suite.sc = NewShardCluster(suite.collectionID, suite.replicaID, suite.vchannelName, suite.version,
		&mockNodeDetector{
			initNodes: nodeEvents,
		}, &mockSegmentDetector{
			initSegments: segmentEvents,
		}, buildMockQueryNode)
	suite.sc.SetupFirstVersion()
	for _, evt := range segmentEvents {
		suite.sc.SyncSegments([]*querypb.ReplicaSegmentsInfo{
			{
				NodeId:      evt.nodeIDs[0],
				PartitionId: evt.partitionID,
				SegmentIds:  []int64{evt.segmentID},
				Versions:    []int64{0},
			},
		}, segmentStateLoaded)
	}
}

func (suite *ShardClusterSuite) TearDownTest() {
	suite.sc.Close()
	suite.sc = nil
}

func (suite *ShardClusterSuite) TestReleaseSegments() {
	type TestCase struct {
		tag        string
		segmentIDs []int64
		nodeID     int64
		scope      querypb.DataScope

		expectAlloc map[int64][]int64
		expectError bool
		force       bool
	}

	cases := []TestCase{
		{
			tag:        "normal release",
			segmentIDs: []int64{2},
			nodeID:     2,
			scope:      querypb.DataScope_All,
			expectAlloc: map[int64][]int64{
				1: {1},
			},
			expectError: false,
			force:       false,
		},
	}

	for _, test := range cases {
		suite.Run(test.tag, func() {
			suite.TearDownTest()
			suite.SetupTest()

			err := suite.sc.ReleaseSegments(context.Background(), &querypb.ReleaseSegmentsRequest{
				Base:       &commonpb.MsgBase{},
				NodeID:     test.nodeID,
				SegmentIDs: test.segmentIDs,
				Scope:      test.scope,
			}, test.force)
			if test.expectError {
				suite.Error(err)
			} else {
				suite.NoError(err)
				alloc, vid := suite.sc.segmentAllocations(nil)
				suite.sc.finishUsage(vid)
				suite.Equal(test.expectAlloc, alloc)
			}
		})
	}
}

func TestShardClusterSuite(t *testing.T) {
	suite.Run(t, new(ShardClusterSuite))
}
