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
	"os"
	"path"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"stathat.com/c/consistent"
)

func checkWatchInfoWithState(t *testing.T, kv kv.MetaKv, state datapb.ChannelWatchState, nodeID UniqueID, channelName string, collectionID UniqueID) {
	prefix := Params.DataCoordCfg.ChannelWatchSubPath

	info, err := kv.Load(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName))
	assert.NoError(t, err)
	assert.NotNil(t, info)

	watchInfo, err := parseWatchInfo("fakeKey", []byte(info))
	assert.NoError(t, err)
	assert.Equal(t, watchInfo.GetState(), state)
	assert.Equal(t, watchInfo.Vchan.GetChannelName(), channelName)
	assert.Equal(t, watchInfo.Vchan.GetCollectionID(), collectionID)
}

func getOpsWithWatchInfo(nodeID UniqueID, ch *channel) ChannelOpSet {
	var ops ChannelOpSet
	ops.Add(nodeID, []*channel{ch})

	for _, op := range ops {
		op.ChannelWatchInfos = []*datapb.ChannelWatchInfo{{}}
	}
	return ops
}

func TestChannelManager_StateTransfer(t *testing.T) {
	metakv := getMetaKv(t)
	defer func() {
		metakv.RemoveWithPrefix("")
		metakv.Close()
	}()

	p := "/tmp/milvus_ut/rdb_data"
	os.Setenv("ROCKSMQ_PATH", p)

	prefix := Params.DataCoordCfg.ChannelWatchSubPath

	var (
		collectionID = UniqueID(9)
		nodeID       = UniqueID(119)
		channel1     = "channel1"
	)

	getWatchInfoWithState := func(state datapb.ChannelWatchState) *datapb.ChannelWatchInfo {
		return &datapb.ChannelWatchInfo{
			Vchan: &datapb.VchannelInfo{
				CollectionID: collectionID,
				ChannelName:  channel1,
			},
			State: state,
		}
	}

	t.Run("toWatch-WatchSuccess", func(t *testing.T) {
		metakv.RemoveWithPrefix("")
		ctx, cancel := context.WithCancel(context.TODO())
		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			chManager.watchChannelStatesLoop(ctx)
			wg.Done()
		}()

		chManager.AddNode(nodeID)
		chManager.Watch(&channel{channel1, collectionID})
		data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_WatchSuccess))
		require.NoError(t, err)
		err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channel1), string(data))
		require.NoError(t, err)

		// TODO: cancel could arrive earlier than etcd action watch channel
		// if etcd has poor response latency.
		time.Sleep(time.Second)
		cancel()
		wg.Wait()

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_WatchSuccess, nodeID, channel1, collectionID)
	})

	t.Run("ToWatch-WatchFail-ToRelease", func(t *testing.T) {
		metakv.RemoveWithPrefix("")
		ctx, cancel := context.WithCancel(context.TODO())
		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			chManager.watchChannelStatesLoop(ctx)
			wg.Done()
		}()

		chManager.AddNode(nodeID)
		chManager.Watch(&channel{channel1, collectionID})
		data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_WatchFailure))
		require.NoError(t, err)
		err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channel1), string(data))
		require.NoError(t, err)

		// TODO: cancel could arrive earlier than etcd action watch channel
		// if etcd has poor response latency.
		time.Sleep(time.Second)
		cancel()
		wg.Wait()
		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToRelease, nodeID, channel1, collectionID)
	})

	t.Run("ToWatch-Timeout", func(t *testing.T) {
		metakv.RemoveWithPrefix("")
		ctx, cancel := context.WithCancel(context.TODO())
		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			chManager.watchChannelStatesLoop(ctx)
			wg.Done()
		}()

		chManager.AddNode(nodeID)
		chManager.Watch(&channel{channel1, collectionID})

		// simulating timeout behavior of startOne, cuz 20s is a long wait
		e := &ackEvent{
			ackType:     watchTimeoutAck,
			channelName: channel1,
			nodeID:      nodeID,
		}
		chManager.stateTimer.notifyTimeoutWatcher(e)
		chManager.stateTimer.stopIfExsit(e)

		// TODO: cancel could arrive earlier than etcd action watch channel
		// if etcd has poor response latency.
		time.Sleep(time.Second)
		cancel()
		wg.Wait()
		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToRelease, nodeID, channel1, collectionID)
	})

	t.Run("toRelease-ReleaseSuccess-Delete-reassign-ToWatch", func(t *testing.T) {
		var oldNode = UniqueID(120)

		metakv.RemoveWithPrefix("")
		ctx, cancel := context.WithCancel(context.TODO())
		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			chManager.watchChannelStatesLoop(ctx)
			wg.Done()
		}()

		chManager.store = &ChannelStore{
			store: metakv,
			channelsInfo: map[int64]*NodeChannelInfo{
				nodeID: {nodeID, []*channel{
					{channel1, collectionID},
				}},
			},
		}

		err = chManager.Release(nodeID, channel1)
		assert.NoError(t, err)
		chManager.AddNode(oldNode)

		data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ReleaseSuccess))
		require.NoError(t, err)
		err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channel1), string(data))
		require.NoError(t, err)

		// TODO: cancel could arrive earlier than etcd action watch channel
		// if etcd has poor response latency.
		time.Sleep(time.Second)
		cancel()
		wg.Wait()

		w, err := metakv.Load(path.Join(prefix, strconv.FormatInt(nodeID, 10)))
		assert.Error(t, err)
		assert.Empty(t, w)

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, oldNode, channel1, collectionID)
	})

	t.Run("toRelease-ReleaseFail-CleanUpAndDelete-Reassign-ToWatch", func(t *testing.T) {
		var oldNode = UniqueID(121)

		metakv.RemoveWithPrefix("")
		ctx, cancel := context.WithCancel(context.TODO())
		factory := msgstream.NewRmsFactory()
		_, err := factory.NewMsgStream(context.TODO())
		require.NoError(t, err)
		chManager, err := NewChannelManager(metakv, newMockHandler(), withMsgstreamFactory(factory))
		require.NoError(t, err)

		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			chManager.watchChannelStatesLoop(ctx)
			wg.Done()
		}()

		chManager.store = &ChannelStore{
			store: metakv,
			channelsInfo: map[int64]*NodeChannelInfo{
				nodeID: {nodeID, []*channel{
					{channel1, collectionID},
				}},
			},
		}

		err = chManager.Release(nodeID, channel1)
		assert.NoError(t, err)
		chManager.AddNode(oldNode)

		data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ReleaseFailure))
		require.NoError(t, err)
		err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channel1), string(data))
		require.NoError(t, err)

		// TODO: cancel could arrive earlier than etcd action watch channel
		// if etcd has poor response latency.
		time.Sleep(time.Second)
		cancel()
		wg.Wait()

		w, err := metakv.Load(path.Join(prefix, strconv.FormatInt(nodeID, 10)))
		assert.Error(t, err)
		assert.Empty(t, w)

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, oldNode, channel1, collectionID)
	})

}

func TestChannelManager(t *testing.T) {
	metakv := getMetaKv(t)
	defer func() {
		metakv.RemoveWithPrefix("")
		metakv.Close()
	}()

	prefix := Params.DataCoordCfg.ChannelWatchSubPath
	t.Run("test AddNode", func(t *testing.T) {
		// Note: this test is based on the default registerPolicy
		defer metakv.RemoveWithPrefix("")
		var (
			collectionID       = UniqueID(8)
			nodeID, nodeToAdd  = UniqueID(118), UniqueID(811)
			channel1, channel2 = "channel1", "channel2"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)
		chManager.store = &ChannelStore{
			store: metakv,
			channelsInfo: map[int64]*NodeChannelInfo{
				nodeID: {nodeID, []*channel{
					{channel1, collectionID},
					{channel2, collectionID},
				}},
			},
		}

		err = chManager.AddNode(nodeToAdd)
		assert.NoError(t, err)

		chInfo := chManager.store.GetNode(nodeID)
		assert.Equal(t, 2, len(chInfo.Channels))
		chInfo = chManager.store.GetNode(nodeToAdd)
		assert.Equal(t, 0, len(chInfo.Channels))

		err = chManager.Watch(&channel{"channel-3", collectionID})
		assert.NoError(t, err)

		chInfo = chManager.store.GetNode(nodeToAdd)
		assert.Equal(t, 1, len(chInfo.Channels))
		chManager.stateTimer.removeTimers([]string{"channel-3"})

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, nodeToAdd, "channel-3", collectionID)
	})

	t.Run("test Watch", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var (
			collectionID = UniqueID(7)
			nodeID       = UniqueID(117)
			bufferCh     = "bufferID"
			chanToAdd    = "new-channel-watch"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		err = chManager.Watch(&channel{bufferCh, collectionID})
		assert.NoError(t, err)
		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, bufferID, bufferCh, collectionID)

		chManager.store.Add(nodeID)
		err = chManager.Watch(&channel{chanToAdd, collectionID})
		assert.NoError(t, err)
		chManager.stateTimer.removeTimers([]string{chanToAdd})

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, nodeID, chanToAdd, collectionID)
	})

	t.Run("test Release", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var (
			collectionID               = UniqueID(4)
			nodeID, invalidNodeID      = UniqueID(114), UniqueID(999)
			channelName, invalidChName = "to-release", "invalid-to-release"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)
		chManager.store = &ChannelStore{
			store: metakv,
			channelsInfo: map[int64]*NodeChannelInfo{
				nodeID: {nodeID, []*channel{{channelName, collectionID}}},
			},
		}

		err = chManager.Release(invalidNodeID, invalidChName)
		assert.Error(t, err)

		err = chManager.Release(nodeID, channelName)
		assert.NoError(t, err)
		chManager.stateTimer.stopIfExsit(&ackEvent{releaseSuccessAck, channelName, nodeID})

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToRelease, nodeID, channelName, collectionID)
	})

	t.Run("test toDelete", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var collectionID = UniqueID(5)

		tests := []struct {
			isvalid bool
			nodeID  UniqueID
			chName  string
		}{
			{true, UniqueID(125), "normal-chan"},
			{true, UniqueID(115), "to-delete-chan"},
			{false, UniqueID(9), "invalid-chan"},
		}

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		// prepare tests
		for _, test := range tests {
			if test.isvalid {
				chManager.store.Add(test.nodeID)
				ops := getOpsWithWatchInfo(test.nodeID, &channel{test.chName, collectionID})
				err = chManager.store.Update(ops)
				require.NoError(t, err)

				info, err := metakv.Load(path.Join(prefix, strconv.FormatInt(test.nodeID, 10), test.chName))
				require.NoError(t, err)
				require.NotNil(t, info)
			}
		}

		remainTest, reassignTest := tests[0], tests[1]
		err = chManager.toDelete(reassignTest.nodeID, reassignTest.chName)
		assert.NoError(t, err)
		chManager.stateTimer.stopIfExsit(&ackEvent{releaseSuccessAck, reassignTest.chName, reassignTest.nodeID})

		// test no nodes are removed from store
		nodesID := chManager.store.GetNodes()
		assert.Equal(t, 2, len(nodesID))

		// test nodes of reassignTest contains no channel
		nodeChanInfo := chManager.store.GetNode(reassignTest.nodeID)
		assert.Equal(t, 0, len(nodeChanInfo.Channels))

		// test all channels are assgined to node of remainTest
		nodeChanInfo = chManager.store.GetNode(remainTest.nodeID)
		assert.Equal(t, 2, len(nodeChanInfo.Channels))
		assert.ElementsMatch(t, []*channel{{remainTest.chName, collectionID}, {reassignTest.chName, collectionID}}, nodeChanInfo.Channels)

		// Delete node of reassginTest and try to toDelete node in remainTest
		err = chManager.DeleteNode(reassignTest.nodeID)
		require.NoError(t, err)

		err = chManager.toDelete(remainTest.nodeID, remainTest.chName)
		assert.NoError(t, err)
		chManager.stateTimer.stopIfExsit(&ackEvent{releaseSuccessAck, reassignTest.chName, reassignTest.nodeID})

		// channel is added to bufferID because there's only one node left
		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, bufferID, remainTest.chName, collectionID)
	})
	t.Run("test cleanUpAndDelete", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var collectionID = UniqueID(6)

		tests := []struct {
			isvalid bool
			nodeID  UniqueID
			chName  string
		}{
			{true, UniqueID(126), "normal-chan"},
			{true, UniqueID(116), "to-delete-chan"},
			{false, UniqueID(9), "invalid-chan"},
		}

		factory := msgstream.NewRmsFactory()
		_, err := factory.NewMsgStream(context.TODO())
		require.NoError(t, err)
		chManager, err := NewChannelManager(metakv, newMockHandler(), withMsgstreamFactory(factory))

		require.NoError(t, err)

		// prepare tests
		for _, test := range tests {
			if test.isvalid {
				chManager.store.Add(test.nodeID)
				ops := getOpsWithWatchInfo(test.nodeID, &channel{test.chName, collectionID})
				err = chManager.store.Update(ops)
				require.NoError(t, err)

				info, err := metakv.Load(path.Join(prefix, strconv.FormatInt(test.nodeID, 10), test.chName))
				require.NoError(t, err)
				require.NotNil(t, info)
			}
		}

		remainTest, reassignTest := tests[0], tests[1]
		err = chManager.cleanUpAndDelete(reassignTest.nodeID, reassignTest.chName)
		assert.NoError(t, err)
		chManager.stateTimer.stopIfExsit(&ackEvent{releaseSuccessAck, reassignTest.chName, reassignTest.nodeID})

		// test no nodes are removed from store
		nodesID := chManager.store.GetNodes()
		assert.Equal(t, 2, len(nodesID))

		// test nodes of reassignTest contains no channel
		nodeChanInfo := chManager.store.GetNode(reassignTest.nodeID)
		assert.Equal(t, 0, len(nodeChanInfo.Channels))

		// test all channels are assgined to node of remainTest
		nodeChanInfo = chManager.store.GetNode(remainTest.nodeID)
		assert.Equal(t, 2, len(nodeChanInfo.Channels))
		assert.ElementsMatch(t, []*channel{{remainTest.chName, collectionID}, {reassignTest.chName, collectionID}}, nodeChanInfo.Channels)

		// Delete node of reassginTest and try to cleanUpAndDelete node in remainTest
		err = chManager.DeleteNode(reassignTest.nodeID)
		require.NoError(t, err)

		err = chManager.cleanUpAndDelete(remainTest.nodeID, remainTest.chName)
		assert.NoError(t, err)
		chManager.stateTimer.stopIfExsit(&ackEvent{releaseSuccessAck, reassignTest.chName, reassignTest.nodeID})

		// channel is added to bufferID because there's only one node left
		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, bufferID, remainTest.chName, collectionID)
	})

	t.Run("test getChannelByNodeAndName", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var (
			nodeID       = UniqueID(113)
			collectionID = UniqueID(3)
			channelName  = "get-channel-by-node-and-name"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		ch := chManager.getChannelByNodeAndName(nodeID, channelName)
		assert.Nil(t, ch)

		chManager.store.Add(nodeID)
		ch = chManager.getChannelByNodeAndName(nodeID, channelName)
		assert.Nil(t, ch)

		chManager.store = &ChannelStore{
			store: metakv,
			channelsInfo: map[int64]*NodeChannelInfo{
				nodeID: {nodeID, []*channel{{channelName, collectionID}}},
			},
		}
		ch = chManager.getChannelByNodeAndName(nodeID, channelName)
		assert.NotNil(t, ch)
		assert.Equal(t, collectionID, ch.CollectionID)
		assert.Equal(t, channelName, ch.Name)
	})

	t.Run("test fillChannelWatchInfoWithState", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")
		var (
			nodeID       = UniqueID(111)
			collectionID = UniqueID(1)
			channelName  = "fill-channel-watchInfo-with-state"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)

		tests := []struct {
			inState datapb.ChannelWatchState

			description string
		}{
			{datapb.ChannelWatchState_ToWatch, "fill toWatch state"},
			{datapb.ChannelWatchState_ToRelease, "fill toRelase state"},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				ops := getReleaseOp(nodeID, &channel{channelName, collectionID})
				for _, op := range ops {
					chs := chManager.fillChannelWatchInfoWithState(op, test.inState)
					assert.Equal(t, 1, len(chs))
					assert.Equal(t, channelName, chs[0])
					assert.Equal(t, 1, len(op.ChannelWatchInfos))
					assert.Equal(t, test.inState, op.ChannelWatchInfos[0].GetState())

					chManager.stateTimer.removeTimers(chs)
				}
			})
		}
	})

	t.Run("test updateWithTimer", func(t *testing.T) {
		var (
			nodeID       = UniqueID(112)
			collectionID = UniqueID(2)
			channelName  = "update-with-timer"
		)

		chManager, err := NewChannelManager(metakv, newMockHandler())
		require.NoError(t, err)
		chManager.store.Add(nodeID)

		opSet := getReleaseOp(nodeID, &channel{channelName, collectionID})

		chManager.updateWithTimer(opSet, datapb.ChannelWatchState_ToWatch)
		chManager.stateTimer.stopIfExsit(&ackEvent{watchSuccessAck, channelName, nodeID})

		checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, nodeID, channelName, collectionID)
	})
}

func TestChannelManager_Reload(t *testing.T) {
	metakv := getMetaKv(t)
	defer func() {
		metakv.RemoveWithPrefix("")
		metakv.Close()
	}()

	var (
		nodeID       = UniqueID(200)
		collectionID = UniqueID(2)
		channelName  = "channel-checkOldNodes"
	)
	prefix := Params.DataCoordCfg.ChannelWatchSubPath

	getWatchInfoWithState := func(state datapb.ChannelWatchState) *datapb.ChannelWatchInfo {
		return &datapb.ChannelWatchInfo{
			Vchan: &datapb.VchannelInfo{
				CollectionID: collectionID,
				ChannelName:  channelName,
			},
			State:     state,
			TimeoutTs: time.Now().Add(20 * time.Second).UnixNano(),
		}
	}

	t.Run("test checkOldNodes", func(t *testing.T) {
		metakv.RemoveWithPrefix("")

		t.Run("ToWatch", func(t *testing.T) {
			defer metakv.RemoveWithPrefix("")
			data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ToWatch))
			require.NoError(t, err)
			chManager, err := NewChannelManager(metakv, newMockHandler())
			require.NoError(t, err)
			err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName), string(data))
			require.NoError(t, err)

			chManager.checkOldNodes([]UniqueID{nodeID})
			_, ok := chManager.stateTimer.runningTimers.Load(channelName)
			assert.True(t, ok)
			chManager.stateTimer.removeTimers([]string{channelName})
		})

		t.Run("ToRelease", func(t *testing.T) {
			defer metakv.RemoveWithPrefix("")
			data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ToRelease))
			require.NoError(t, err)
			chManager, err := NewChannelManager(metakv, newMockHandler())
			require.NoError(t, err)
			err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName), string(data))
			require.NoError(t, err)
			err = chManager.checkOldNodes([]UniqueID{nodeID})
			assert.NoError(t, err)

			_, ok := chManager.stateTimer.runningTimers.Load(channelName)
			assert.True(t, ok)
			chManager.stateTimer.removeTimers([]string{channelName})
		})

		t.Run("WatchFail", func(t *testing.T) {
			defer metakv.RemoveWithPrefix("")
			chManager, err := NewChannelManager(metakv, newMockHandler())
			require.NoError(t, err)
			chManager.store = &ChannelStore{
				store: metakv,
				channelsInfo: map[int64]*NodeChannelInfo{
					nodeID: {nodeID, []*channel{{channelName, collectionID}}}},
			}

			data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_WatchFailure))
			require.NoError(t, err)
			err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName), string(data))
			require.NoError(t, err)
			err = chManager.checkOldNodes([]UniqueID{nodeID})
			assert.NoError(t, err)

			checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToRelease, nodeID, channelName, collectionID)
			chManager.stateTimer.removeTimers([]string{channelName})
		})

		t.Run("ReleaseSuccess", func(t *testing.T) {
			defer metakv.RemoveWithPrefix("")
			chManager, err := NewChannelManager(metakv, newMockHandler())
			require.NoError(t, err)
			data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ReleaseSuccess))
			chManager.store = &ChannelStore{
				store: metakv,
				channelsInfo: map[int64]*NodeChannelInfo{
					nodeID: {nodeID, []*channel{{channelName, collectionID}}}},
			}
			chManager.AddNode(bufferID)

			require.NoError(t, err)
			err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName), string(data))
			require.NoError(t, err)
			err = chManager.checkOldNodes([]UniqueID{nodeID})
			assert.NoError(t, err)

			v, err := metakv.Load(path.Join(prefix, strconv.FormatInt(nodeID, 10)))
			assert.Error(t, err)
			assert.Empty(t, v)
		})

		t.Run("ReleaseFail", func(t *testing.T) {
			defer metakv.RemoveWithPrefix("")
			chManager, err := NewChannelManager(metakv, newMockHandler())
			require.NoError(t, err)
			data, err := proto.Marshal(getWatchInfoWithState(datapb.ChannelWatchState_ReleaseSuccess))
			chManager.store = &ChannelStore{
				store: metakv,
				channelsInfo: map[int64]*NodeChannelInfo{
					nodeID: {nodeID, []*channel{{channelName, collectionID}}},
					999:    {999, []*channel{}},
				},
			}
			require.NoError(t, err)
			err = metakv.Save(path.Join(prefix, strconv.FormatInt(nodeID, 10), channelName), string(data))
			require.NoError(t, err)
			err = chManager.checkOldNodes([]UniqueID{nodeID, 999})
			assert.NoError(t, err)

			time.Sleep(time.Second)
			v, err := metakv.Load(path.Join(prefix, strconv.FormatInt(nodeID, 10)))
			assert.Error(t, err)
			assert.Empty(t, v)

			checkWatchInfoWithState(t, metakv, datapb.ChannelWatchState_ToWatch, 999, channelName, collectionID)
		})
	})

	t.Run("test reload with data", func(t *testing.T) {
		defer metakv.RemoveWithPrefix("")

		ctx, cancel := context.WithCancel(context.TODO())
		defer cancel()

		hash := consistent.New()
		cm, err := NewChannelManager(metakv, newMockHandler(), withFactory(NewConsistentHashChannelPolicyFactory(hash)))
		assert.Nil(t, err)
		assert.Nil(t, cm.AddNode(1))
		assert.Nil(t, cm.AddNode(2))
		assert.Nil(t, cm.Watch(&channel{"channel1", 1}))
		assert.Nil(t, cm.Watch(&channel{"channel2", 1}))

		hash2 := consistent.New()
		cm2, err := NewChannelManager(metakv, newMockHandler(), withFactory(NewConsistentHashChannelPolicyFactory(hash2)))
		assert.Nil(t, err)
		assert.Nil(t, cm2.Startup(ctx, []int64{1, 2}))
		assert.Nil(t, cm2.AddNode(3))
		assert.True(t, cm2.Match(3, "channel1"))
		assert.True(t, cm2.Match(3, "channel2"))
	})
}

func TestChannelManager_RemoveChannel(t *testing.T) {
	metakv := getMetaKv(t)
	defer func() {
		metakv.RemoveWithPrefix("")
		metakv.Close()
	}()

	type fields struct {
		store RWChannelStore
	}
	type args struct {
		channelName string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"test remove existed channel",
			fields{
				store: &ChannelStore{
					store: metakv,
					channelsInfo: map[int64]*NodeChannelInfo{
						1: {
							NodeID: 1,
							Channels: []*channel{
								{"ch1", 1},
							},
						},
					},
				},
			},
			args{
				"ch1",
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &ChannelManager{
				store: tt.fields.store,
			}
			err := c.RemoveChannel(tt.args.channelName)
			assert.Equal(t, tt.wantErr, err != nil)
			_, ch := c.findChannel(tt.args.channelName)
			assert.Nil(t, ch)
		})
	}
}

func TestChannelManager_HelperFunc(t *testing.T) {
	c := &ChannelManager{}
	t.Run("test getOldOnlines", func(t *testing.T) {
		tests := []struct {
			nodes  []int64
			oNodes []int64

			expectedOut []int64
			desription  string
		}{
			{[]int64{}, []int64{}, []int64{}, "empty both"},
			{[]int64{1}, []int64{}, []int64{}, "empty oNodes"},
			{[]int64{}, []int64{1}, []int64{}, "empty nodes"},
			{[]int64{1}, []int64{1}, []int64{1}, "same one"},
			{[]int64{1, 2}, []int64{1}, []int64{1}, "same one 2"},
			{[]int64{1}, []int64{1, 2}, []int64{1}, "same one 3"},
			{[]int64{1, 2}, []int64{1, 2}, []int64{1, 2}, "same two"},
		}

		for _, test := range tests {
			t.Run(test.desription, func(t *testing.T) {
				nodes := c.getOldOnlines(test.nodes, test.oNodes)
				assert.ElementsMatch(t, test.expectedOut, nodes)
			})
		}
	})
}
