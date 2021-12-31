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
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/types"

	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
)

func TestMain(t *testing.M) {
	rand.Seed(time.Now().Unix())
	Params.DataNodeCfg.InitAlias("datanode-alias-1")
	Params.Init()
	// change to specific channel for test
	Params.DataNodeCfg.TimeTickChannelName = Params.DataNodeCfg.TimeTickChannelName + strconv.Itoa(rand.Int())
	code := t.Run()
	os.Exit(code)
}

func TestDataNode(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	node := newIDLEDataNodeMock(ctx)
	etcdCli, err := etcd.GetEtcdClient(&Params.BaseParams)
	assert.Nil(t, err)
	defer etcdCli.Close()
	node.SetEtcdClient(etcdCli)
	err = node.Init()
	assert.Nil(t, err)
	err = node.Start()
	assert.Nil(t, err)

	t.Run("Test WatchDmChannels ", func(t *testing.T) {
		emptyNode := &DataNode{}

		status, err := emptyNode.WatchDmChannels(ctx, &datapb.WatchDmChannelsRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})

	t.Run("Test SetRootCoord", func(t *testing.T) {
		emptyDN := &DataNode{}
		tests := []struct {
			inrc        types.RootCoord
			isvalid     bool
			description string
		}{
			{nil, false, "nil input"},
			{&RootCoordFactory{}, true, "valid input"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				err := emptyDN.SetRootCoord(test.inrc)
				if test.isvalid {
					assert.NoError(t, err)
				} else {
					assert.Error(t, err)
				}
			})
		}
	})

	t.Run("Test SetDataCoord", func(t *testing.T) {
		emptyDN := &DataNode{}
		tests := []struct {
			inrc        types.DataCoord
			isvalid     bool
			description string
		}{
			{nil, false, "nil input"},
			{&DataCoordFactory{}, true, "valid input"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				err := emptyDN.SetDataCoord(test.inrc)
				if test.isvalid {
					assert.NoError(t, err)
				} else {
					assert.Error(t, err)
				}
			})
		}
	})

	t.Run("Test GetComponentStates", func(t *testing.T) {
		stat, err := node.GetComponentStates(node.ctx)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, stat.Status.ErrorCode)
	})

	t.Run("Test NewDataSyncService", func(t *testing.T) {
		t.Skip()
		ctx, cancel := context.WithCancel(context.Background())
		node2 := newIDLEDataNodeMock(ctx)
		err = node2.Start()
		assert.Nil(t, err)
		dmChannelName := "fake-by-dev-rootcoord-dml-channel-test-NewDataSyncService"

		vchan := &datapb.VchannelInfo{
			CollectionID:      1,
			ChannelName:       dmChannelName,
			UnflushedSegments: []*datapb.SegmentInfo{},
		}

		require.Equal(t, 0, len(node2.vchan2FlushChs))
		require.Equal(t, 0, len(node2.vchan2SyncService))

		err := node2.NewDataSyncService(vchan)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(node2.vchan2FlushChs))
		assert.Equal(t, 1, len(node2.vchan2SyncService))

		err = node2.NewDataSyncService(vchan)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(node2.vchan2FlushChs))
		assert.Equal(t, 1, len(node2.vchan2SyncService))

		cancel()
		<-node2.ctx.Done()
		err = node2.Stop()
		assert.Nil(t, err)
	})

	t.Run("Test FlushSegments", func(t *testing.T) {
		dmChannelName := "fake-by-dev-rootcoord-dml-channel-test-FlushSegments"

		node1 := newIDLEDataNodeMock(context.TODO())
		node1.SetEtcdClient(etcdCli)
		err = node1.Init()
		assert.Nil(t, err)
		err = node1.Start()
		assert.Nil(t, err)
		defer func() {
			err := node1.Stop()
			assert.Nil(t, err)
		}()

		vchan := &datapb.VchannelInfo{
			CollectionID:      1,
			ChannelName:       dmChannelName,
			UnflushedSegments: []*datapb.SegmentInfo{},
			FlushedSegments:   []*datapb.SegmentInfo{},
		}
		err := node1.NewDataSyncService(vchan)
		assert.Nil(t, err)

		service, ok := node1.vchan2SyncService[dmChannelName]
		assert.True(t, ok)
		err = service.replica.addNewSegment(0, 1, 1, dmChannelName, &internalpb.MsgPosition{}, &internalpb.MsgPosition{})
		assert.Nil(t, err)

		req := &datapb.FlushSegmentsRequest{
			Base:         &commonpb.MsgBase{},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{0},
		}

		wg := sync.WaitGroup{}
		wg.Add(2)

		go func() {
			defer wg.Done()

			status, err := node1.FlushSegments(node1.ctx, req)
			assert.NoError(t, err)
			assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
		}()

		go func() {
			defer wg.Done()

			timeTickMsgPack := msgstream.MsgPack{}
			timeTickMsg := &msgstream.TimeTickMsg{
				BaseMsg: msgstream.BaseMsg{
					BeginTimestamp: Timestamp(0),
					EndTimestamp:   Timestamp(0),
					HashValues:     []uint32{0},
				},
				TimeTickMsg: internalpb.TimeTickMsg{
					Base: &commonpb.MsgBase{
						MsgType:   commonpb.MsgType_TimeTick,
						MsgID:     UniqueID(0),
						Timestamp: math.MaxUint64,
						SourceID:  0,
					},
				},
			}
			timeTickMsgPack.Msgs = append(timeTickMsgPack.Msgs, timeTickMsg)

			// pulsar produce
			msFactory := msgstream.NewPmsFactory()
			m := map[string]interface{}{
				"pulsarAddress":  Params.DataNodeCfg.PulsarAddress,
				"receiveBufSize": 1024,
				"pulsarBufSize":  1024}
			err = msFactory.SetParams(m)
			assert.NoError(t, err)
			insertStream, err := msFactory.NewMsgStream(node1.ctx)
			assert.NoError(t, err)
			insertStream.AsProducer([]string{dmChannelName})
			insertStream.Start()
			defer insertStream.Close()

			err = insertStream.Broadcast(&timeTickMsgPack)
			assert.NoError(t, err)

			err = insertStream.Broadcast(&timeTickMsgPack)
			assert.NoError(t, err)
		}()

		wg.Wait()
		// dup call
		status, err := node1.FlushSegments(node1.ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

		// failure call
		req = &datapb.FlushSegmentsRequest{
			Base:         &commonpb.MsgBase{},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{1},
		}

		status, err = node1.FlushSegments(node1.ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)

		req = &datapb.FlushSegmentsRequest{
			Base:           &commonpb.MsgBase{},
			DbID:           0,
			CollectionID:   1,
			SegmentIDs:     []int64{},
			MarkSegmentIDs: []int64{2},
		}

		status, err = node1.FlushSegments(node1.ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)

		// manual inject meta error
		node1.chanMut.Lock()
		node1.vchan2FlushChs[dmChannelName+"1"] = node1.vchan2FlushChs[dmChannelName]
		delete(node1.vchan2FlushChs, dmChannelName)
		node1.chanMut.Unlock()
		node1.segmentCache.Remove(0)

		req = &datapb.FlushSegmentsRequest{
			Base:         &commonpb.MsgBase{},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{0},
		}

		status, err = node1.FlushSegments(node1.ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)

	})

	t.Run("Test GetTimeTickChannel", func(t *testing.T) {
		_, err := node.GetTimeTickChannel(node.ctx)
		assert.NoError(t, err)
	})

	t.Run("Test GetStatisticsChannel", func(t *testing.T) {
		_, err := node.GetStatisticsChannel(node.ctx)
		assert.NoError(t, err)
	})

	t.Run("Test getSystemInfoMetrics", func(t *testing.T) {
		emptyNode := &DataNode{}
		emptyNode.session = &sessionutil.Session{ServerID: 1}

		req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.NoError(t, err)
		resp, err := emptyNode.getSystemInfoMetrics(context.TODO(), req)
		assert.NoError(t, err)
		log.Info("Test DataNode.getSystemInfoMetrics",
			zap.String("name", resp.ComponentName),
			zap.String("response", resp.Response))
	})

	t.Run("Test GetMetrics", func(t *testing.T) {
		node := &DataNode{}
		node.session = &sessionutil.Session{ServerID: 1}
		// server is closed
		node.State.Store(internalpb.StateCode_Abnormal)
		resp, err := node.GetMetrics(ctx, &milvuspb.GetMetricsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		node.State.Store(internalpb.StateCode_Healthy)

		// failed to parse metric type
		invalidRequest := "invalid request"
		resp, err = node.GetMetrics(ctx, &milvuspb.GetMetricsRequest{
			Request: invalidRequest,
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)

		// unsupported metric type
		unsupportedMetricType := "unsupported"
		req, err := metricsinfo.ConstructRequestByMetricType(unsupportedMetricType)
		assert.NoError(t, err)
		resp, err = node.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)

		// normal case
		req, err = metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.NoError(t, err)
		resp, err = node.GetMetrics(node.ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		log.Info("Test DataNode.GetMetrics",
			zap.String("name", resp.ComponentName),
			zap.String("response", resp.Response))
	})

	t.Run("Test BackGroundGC", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		node := newIDLEDataNodeMock(ctx)

		vchanNameCh := make(chan string)
		node.clearSignal = vchanNameCh
		go node.BackGroundGC(vchanNameCh)

		testDataSyncs := []struct {
			dmChannelName string
		}{
			{"fake-by-dev-rootcoord-dml-backgroundgc-1"},
			{"fake-by-dev-rootcoord-dml-backgroundgc-2"},
			{"fake-by-dev-rootcoord-dml-backgroundgc-3"},
			{""},
			{""},
		}

		for i, test := range testDataSyncs {
			if i <= 2 {

				err = node.NewDataSyncService(&datapb.VchannelInfo{CollectionID: 1, ChannelName: test.dmChannelName})

				assert.Nil(t, err)

				vchanNameCh <- test.dmChannelName
			}
		}

		assert.Eventually(t, func() bool {
			node.chanMut.Lock()
			defer node.chanMut.Unlock()
			return len(node.vchan2FlushChs) == 0
		}, time.Second, time.Millisecond)

		cancel()
	})

	t.Run("Test ReleaseDataSyncService", func(t *testing.T) {
		dmChannelName := "fake-by-dev-rootcoord-dml-channel-test-NewDataSyncService"

		vchan := &datapb.VchannelInfo{
			CollectionID:      1,
			ChannelName:       dmChannelName,
			UnflushedSegments: []*datapb.SegmentInfo{},
		}

		err := node.NewDataSyncService(vchan)
		require.NoError(t, err)
		require.Equal(t, 1, len(node.vchan2FlushChs))
		require.Equal(t, 1, len(node.vchan2SyncService))
		time.Sleep(100 * time.Millisecond)

		node.ReleaseDataSyncService(dmChannelName)
		assert.Equal(t, 0, len(node.vchan2FlushChs))
		assert.Equal(t, 0, len(node.vchan2SyncService))

		s, ok := node.vchan2SyncService[dmChannelName]
		assert.False(t, ok)
		assert.Nil(t, s)
	})

	t.Run("Test GetChannelName", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		node := newIDLEDataNodeMock(ctx)

		testCollIDs := []UniqueID{0, 1, 2, 1}
		testSegIDs := []UniqueID{10, 11, 12, 13}
		testchanNames := []string{"a", "b", "c", "d"}

		node.chanMut.Lock()
		for i, name := range testchanNames {
			replica := &SegmentReplica{
				collectionID: testCollIDs[i],
				newSegments:  make(map[UniqueID]*Segment),
			}

			err = replica.addNewSegment(testSegIDs[i], testCollIDs[i], 0, name, &internalpb.MsgPosition{}, nil)
			assert.Nil(t, err)
			node.vchan2SyncService[name] = &dataSyncService{collectionID: testCollIDs[i], replica: replica}
		}
		node.chanMut.Unlock()

		type Test struct {
			inCollID         UniqueID
			expectedChannels []string

			inSegID         UniqueID
			expectedChannel string
		}
		tests := []Test{
			{0, []string{"a"}, 10, "a"},
			{1, []string{"b", "d"}, 11, "b"},
			{2, []string{"c"}, 12, "c"},
			{3, []string{}, 13, "d"},
			{3, []string{}, 100, ""},
		}

		for _, test := range tests {
			actualChannels := node.getChannelNamesbyCollectionID(test.inCollID)
			assert.ElementsMatch(t, test.expectedChannels, actualChannels)

			actualChannel := node.getChannelNamebySegmentID(test.inSegID)
			assert.Equal(t, test.expectedChannel, actualChannel)
		}

		cancel()
	})

	cancel()
	<-node.ctx.Done()
	err = node.Stop()
	require.Nil(t, err)
}

func TestWatchChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	node := newIDLEDataNodeMock(ctx)
	etcdCli, err := etcd.GetEtcdClient(&Params.BaseParams)
	assert.Nil(t, err)
	defer etcdCli.Close()
	node.SetEtcdClient(etcdCli)
	err = node.Init()
	assert.Nil(t, err)
	err = node.Start()
	assert.Nil(t, err)
	err = node.Register()
	assert.Nil(t, err)

	defer cancel()

	t.Run("test watch channel", func(t *testing.T) {
		kv := etcdkv.NewEtcdKV(etcdCli, Params.DataNodeCfg.MetaRootPath)
		oldInvalidCh := "datanode-etcd-test-by-dev-rootcoord-dml-channel-invalid"
		path := fmt.Sprintf("%s/%d/%s", Params.DataNodeCfg.ChannelWatchSubPath, node.NodeID, oldInvalidCh)
		err = kv.Save(path, string([]byte{23}))
		assert.NoError(t, err)

		ch := fmt.Sprintf("datanode-etcd-test-by-dev-rootcoord-dml-channel_%d", rand.Int31())
		path = fmt.Sprintf("%s/%d/%s", Params.DataNodeCfg.ChannelWatchSubPath, node.NodeID, ch)
		c := make(chan struct{})
		go func() {
			ec := kv.WatchWithPrefix(fmt.Sprintf("%s/%d", Params.DataNodeCfg.ChannelWatchSubPath, node.NodeID))
			c <- struct{}{}
			cnt := 0
			for {
				evt := <-ec
				for _, event := range evt.Events {
					if strings.Contains(string(event.Kv.Key), ch) {
						cnt++
					}
				}
				if cnt >= 2 {
					break
				}
			}
			c <- struct{}{}
		}()
		// wait for check goroutine start Watch
		<-c

		vchan := &datapb.VchannelInfo{
			CollectionID:      1,
			ChannelName:       ch,
			UnflushedSegments: []*datapb.SegmentInfo{},
		}
		info := &datapb.ChannelWatchInfo{
			State: datapb.ChannelWatchState_Uncomplete,
			Vchan: vchan,
		}
		val, err := proto.Marshal(info)
		assert.Nil(t, err)
		err = kv.Save(path, string(val))
		assert.Nil(t, err)

		// wait for check goroutine received 2 events
		<-c
		node.chanMut.RLock()
		_, has := node.vchan2SyncService[ch]
		node.chanMut.RUnlock()
		assert.True(t, has)

		err = kv.RemoveWithPrefix(fmt.Sprintf("%s/%d", Params.DataNodeCfg.ChannelWatchSubPath, node.NodeID))
		assert.Nil(t, err)
		//TODO there is not way to sync Release done, use sleep for now
		time.Sleep(100 * time.Millisecond)

		node.chanMut.RLock()
		_, has = node.vchan2SyncService[ch]
		node.chanMut.RUnlock()
		assert.False(t, has)
	})

	t.Run("handle watch info failed", func(t *testing.T) {
		node.handleWatchInfo("test1", []byte{23})

		node.chanMut.RLock()
		_, has := node.vchan2SyncService["test1"]
		assert.False(t, has)
		node.chanMut.RUnlock()

		info := datapb.ChannelWatchInfo{
			Vchan: nil,
			State: datapb.ChannelWatchState_Uncomplete,
		}
		bs, err := proto.Marshal(&info)
		assert.NoError(t, err)
		node.handleWatchInfo("test2", bs)

		node.chanMut.RLock()
		_, has = node.vchan2SyncService["test2"]
		assert.False(t, has)
		node.chanMut.RUnlock()

		info = datapb.ChannelWatchInfo{
			Vchan: &datapb.VchannelInfo{},
			State: datapb.ChannelWatchState_Uncomplete,
		}
		bs, err = proto.Marshal(&info)
		assert.NoError(t, err)

		node.msFactory = &FailMessageStreamFactory{
			node.msFactory,
		}
		node.handleWatchInfo("test3", bs)
		node.chanMut.RLock()
		_, has = node.vchan2SyncService["test3"]
		assert.False(t, has)
		node.chanMut.RUnlock()

	})
}

func TestDataNode_GetComponentStates(t *testing.T) {
	n := &DataNode{}
	n.State.Store(internalpb.StateCode_Healthy)
	resp, err := n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, common.NotRegisteredID, resp.State.NodeID)
	n.session = &sessionutil.Session{}
	n.session.UpdateRegistered(true)
	resp, err = n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
}
