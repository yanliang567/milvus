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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
)

func TestFlowGraphDeleteNode_delete(t *testing.T) {
	t.Run("test delete", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		deleteData, err := genFlowGraphDeleteData()
		assert.NoError(t, err)

		wg := &sync.WaitGroup{}
		wg.Add(1)
		err = deleteNode.delete(deleteData, defaultSegmentID, wg)
		assert.NoError(t, err)
	})

	t.Run("test segment delete error", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		deleteData, err := genFlowGraphDeleteData()
		assert.NoError(t, err)

		wg := &sync.WaitGroup{}
		wg.Add(1)
		deleteData.deleteTimestamps[defaultSegmentID] = deleteData.deleteTimestamps[defaultSegmentID][:len(deleteData.deleteTimestamps)/2]
		err = deleteNode.delete(deleteData, defaultSegmentID, wg)
		assert.Error(t, err)
	})

	t.Run("test no target segment", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)
		wg := &sync.WaitGroup{}
		wg.Add(1)
		err = deleteNode.delete(nil, defaultSegmentID, wg)
		assert.Error(t, err)
	})

	t.Run("test invalid segmentType", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeGrowing)
		assert.NoError(t, err)

		wg := &sync.WaitGroup{}
		wg.Add(1)
		err = deleteNode.delete(&deleteData{}, defaultSegmentID, wg)
		assert.Error(t, err)
	})
}

func TestFlowGraphDeleteNode_operate(t *testing.T) {
	t.Run("test operate", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		msgDeleteMsg := genDeleteMsg(defaultCollectionID, schemapb.DataType_Int64, defaultDelLength)
		dMsg := deleteMsg{
			deleteMessages: []*msgstream.DeleteMsg{
				msgDeleteMsg,
			},
		}
		msg := []flowgraph.Msg{&dMsg}
		deleteNode.Operate(msg)
		s, err := historical.getSegmentByID(defaultSegmentID, segmentTypeSealed)
		pks := make([]primaryKey, defaultMsgLength)
		for i := 0; i < defaultMsgLength; i++ {
			pks[i] = newInt64PrimaryKey(int64(i))
		}
		s.updateBloomFilter(pks)
		assert.Nil(t, err)
		buf := make([]byte, 8)
		for i := 0; i < defaultMsgLength; i++ {
			common.Endian.PutUint64(buf, uint64(i))
			assert.True(t, s.pkFilter.Test(buf))
		}
	})

	t.Run("test invalid partitionID", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		msgDeleteMsg := genDeleteMsg(defaultCollectionID, schemapb.DataType_Int64, defaultDelLength)
		msgDeleteMsg.PartitionID = common.InvalidPartitionID
		assert.NoError(t, err)
		dMsg := deleteMsg{
			deleteMessages: []*msgstream.DeleteMsg{
				msgDeleteMsg,
			},
		}
		msg := []flowgraph.Msg{&dMsg}
		deleteNode.Operate(msg)
	})

	t.Run("test collection partition not exist", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		msgDeleteMsg := genDeleteMsg(defaultCollectionID, schemapb.DataType_Int64, defaultDelLength)
		msgDeleteMsg.CollectionID = 9999
		msgDeleteMsg.PartitionID = -1
		assert.NoError(t, err)
		dMsg := deleteMsg{
			deleteMessages: []*msgstream.DeleteMsg{
				msgDeleteMsg,
			},
		}
		msg := []flowgraph.Msg{&dMsg}
		assert.Panics(t, func() {
			deleteNode.Operate(msg)
		})
	})

	t.Run("test partition not exist", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		msgDeleteMsg := genDeleteMsg(defaultCollectionID, schemapb.DataType_Int64, defaultDelLength)
		msgDeleteMsg.PartitionID = 9999
		assert.NoError(t, err)
		dMsg := deleteMsg{
			deleteMessages: []*msgstream.DeleteMsg{
				msgDeleteMsg,
			},
		}
		msg := []flowgraph.Msg{&dMsg}
		assert.Panics(t, func() {
			deleteNode.Operate(msg)
		})
	})

	t.Run("test invalid input length", func(t *testing.T) {
		historical, err := genSimpleReplica()
		assert.NoError(t, err)
		deleteNode := newDeleteNode(historical, defaultCollectionID, defaultChannelName)

		err = historical.addSegment(defaultSegmentID,
			defaultPartitionID,
			defaultCollectionID,
			defaultDMLChannel,
			defaultSegmentVersion,
			segmentTypeSealed)
		assert.NoError(t, err)

		msgDeleteMsg := genDeleteMsg(defaultCollectionID, schemapb.DataType_Int64, defaultDelLength)
		dMsg := deleteMsg{
			deleteMessages: []*msgstream.DeleteMsg{
				msgDeleteMsg,
			},
		}
		msg := []flowgraph.Msg{&dMsg, &dMsg}
		deleteNode.Operate(msg)
	})
}
