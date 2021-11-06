// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package querynode

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
)

func getFilterDeleteNode(ctx context.Context) (*filterDeleteNode, error) {
	historical, err := genSimpleReplica()
	if err != nil {
		return nil, err
	}

	historical.addExcludedSegments(defaultCollectionID, nil)
	return newFilteredDeleteNode(historical, defaultCollectionID, defaultPartitionID), nil
}

func TestFlowGraphFilterDeleteNode_filterDeleteNode(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fg, err := getFilterDeleteNode(ctx)
	assert.NoError(t, err)
	fg.Name()
}

func TestFlowGraphFilterDeleteNode_filterInvalidDeleteMessage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Run("delete valid test", func(t *testing.T) {
		msg, err := genSimpleDeleteMsg()
		assert.NoError(t, err)
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		res := fg.filterInvalidDeleteMessage(msg)
		assert.NotNil(t, res)
	})

	t.Run("test delete no collection", func(t *testing.T) {
		msg, err := genSimpleDeleteMsg()
		assert.NoError(t, err)
		msg.CollectionID = UniqueID(1003)
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		res := fg.filterInvalidDeleteMessage(msg)
		assert.Nil(t, res)
	})

	t.Run("test delete not target collection", func(t *testing.T) {
		msg, err := genSimpleDeleteMsg()
		assert.NoError(t, err)
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		fg.collectionID = UniqueID(1000)
		res := fg.filterInvalidDeleteMessage(msg)
		assert.Nil(t, res)
	})

	t.Run("test delete no data", func(t *testing.T) {
		msg, err := genSimpleDeleteMsg()
		assert.NoError(t, err)
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		msg.Timestamps = make([]Timestamp, 0)
		msg.PrimaryKeys = make([]IntPrimaryKey, 0)
		res := fg.filterInvalidDeleteMessage(msg)
		assert.Nil(t, res)
	})
}

func TestFlowGraphFilterDeleteNode_Operate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	genFilterDeleteMsg := func() []flowgraph.Msg {
		dMsg, err := genSimpleDeleteMsg()
		assert.NoError(t, err)
		msg := flowgraph.GenerateMsgStreamMsg([]msgstream.TsMsg{dMsg}, 0, 1000, nil, nil)
		return []flowgraph.Msg{msg}
	}

	t.Run("valid test", func(t *testing.T) {
		msg := genFilterDeleteMsg()
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		res := fg.Operate(msg)
		assert.NotNil(t, res)
	})

	t.Run("invalid input length", func(t *testing.T) {
		msg := genFilterDeleteMsg()
		fg, err := getFilterDeleteNode(ctx)
		assert.NoError(t, err)
		var m flowgraph.Msg
		msg = append(msg, m)
		res := fg.Operate(msg)
		assert.NotNil(t, res)
	})
}
