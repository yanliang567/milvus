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

package querycoord

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

var globalMetaTestDir = "/tmp/milvus_test/global_meta"

func getMockGlobalMetaBroker(ctx context.Context) (*globalMetaBroker, *mocks.DataCoord, *mocks.RootCoord, error) {
	dc := &mocks.DataCoord{}
	rc := &mocks.RootCoord{}
	handler, err := newGlobalMetaBroker(ctx, rc, dc, nil, nil)

	return handler, dc, rc, err
}

func TestGlobalMetaBroker_describeCollection(t *testing.T) {

	ctx, cancel := context.WithCancel(context.Background())

	t.Run("success case", func(t *testing.T) {
		rootCoord := &mocks.RootCoord{}
		defer cancel()

		handler, err := newGlobalMetaBroker(ctx, rootCoord, nil, nil, nil)
		require.NoError(t, err)

		schema := genDefaultCollectionSchema(false)
		rootCoord.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
			Schema: schema,
		}, nil)

		result, err := handler.describeCollection(ctx, defaultCollectionID)
		assert.NoError(t, err)
		assert.Equal(t, schema, result)
	})

	t.Run("failure case", func(t *testing.T) {
		rootCoord := &mocks.RootCoord{}
		defer cancel()

		handler, err := newGlobalMetaBroker(ctx, rootCoord, nil, nil, nil)
		require.NoError(t, err)

		rootCoord.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Return(nil, errors.New("mocked error"))
		_, err = handler.describeCollection(ctx, defaultCollectionID)
		assert.Error(t, err)
	})
}

func TestGlobalMetaBroker_DataCoord(t *testing.T) {
	refreshParams()
	ctx, cancel := context.WithCancel(context.Background())
	dataCoord := newDataCoordMock(ctx)

	cm := storage.NewLocalChunkManager(storage.RootPath(globalMetaTestDir))
	defer cm.RemoveWithPrefix("")
	handler, err := newGlobalMetaBroker(ctx, nil, dataCoord, nil, cm)
	assert.Nil(t, err)

	t.Run("successCase", func(t *testing.T) {
		_, _, err = handler.getRecoveryInfo(ctx, defaultCollectionID, defaultPartitionID)
		assert.Nil(t, err)
		_, err = handler.getSegmentStates(ctx, defaultSegmentID)
		assert.Nil(t, err)
	})

	t.Run("returnError", func(t *testing.T) {
		dataCoord.returnError = true
		_, _, err = handler.getRecoveryInfo(ctx, defaultCollectionID, defaultPartitionID)
		assert.Error(t, err)
		_, err = handler.getSegmentStates(ctx, defaultSegmentID)
		assert.Error(t, err)
		dataCoord.returnError = false
	})

	t.Run("returnGrpcError", func(t *testing.T) {
		dataCoord.returnGrpcError = true
		_, _, err = handler.getRecoveryInfo(ctx, defaultCollectionID, defaultPartitionID)
		assert.Error(t, err)
		_, err = handler.getSegmentStates(ctx, defaultSegmentID)
		assert.Error(t, err)
		dataCoord.returnGrpcError = false
	})

	cancel()
}

//func TestGlobalMetaBroker_IndexCoord(t *testing.T) {
//	refreshParams()
//	ctx, cancel := context.WithCancel(context.Background())
//	rootCoord := newRootCoordMock(ctx)
//	rootCoord.enableIndex = true
//	rootCoord.createCollection(defaultCollectionID)
//	rootCoord.createPartition(defaultCollectionID, defaultPartitionID)
//	indexCoord, err := newIndexCoordMock(globalMetaTestDir)
//	assert.Nil(t, err)
//
//	cm := storage.NewLocalChunkManager(storage.RootPath(globalMetaTestDir))
//	defer cm.RemoveWithPrefix("")
//	handler, err := newGlobalMetaBroker(ctx, rootCoord, nil, indexCoord, cm)
//	assert.Nil(t, err)
//
//	t.Run("successCase", func(t *testing.T) {
//		indexFilePathInfos, err := handler.getIndexFilePaths(ctx, int64(100))
//		assert.Nil(t, err)
//		assert.Equal(t, 1, len(indexFilePathInfos))
//		indexInfos, err := handler.getIndexInfo(ctx, defaultCollectionID, defaultSegmentID, genDefaultCollectionSchema(false))
//		assert.Nil(t, err)
//		assert.Equal(t, 1, len(indexInfos))
//	})
//
//	t.Run("returnError", func(t *testing.T) {
//		indexCoord.returnError = true
//		indexFilePathInfos, err := handler.getIndexFilePaths(ctx, int64(100))
//		assert.Error(t, err)
//		assert.Nil(t, indexFilePathInfos)
//		indexInfos, err := handler.getIndexInfo(ctx, defaultCollectionID, defaultSegmentID, genDefaultCollectionSchema(false))
//		assert.Error(t, err)
//		assert.Nil(t, indexInfos)
//		indexCoord.returnError = false
//	})
//
//	t.Run("returnGrpcError", func(t *testing.T) {
//		indexCoord.returnGrpcError = true
//		indexFilePathInfos, err := handler.getIndexFilePaths(ctx, int64(100))
//		assert.Error(t, err)
//		assert.Nil(t, indexFilePathInfos)
//		indexInfos, err := handler.getIndexInfo(ctx, defaultCollectionID, defaultSegmentID, genDefaultCollectionSchema(false))
//		assert.Error(t, err)
//		assert.Nil(t, indexInfos)
//		indexCoord.returnGrpcError = false
//	})
//
//	cancel()
//}

func TestGetDataSegmentInfosByIDs(t *testing.T) {
	refreshParams()
	ctx, cancel := context.WithCancel(context.Background())
	dataCoord := newDataCoordMock(ctx)

	cm := storage.NewLocalChunkManager(storage.RootPath(globalMetaTestDir))
	defer cm.RemoveWithPrefix("")
	handler, err := newGlobalMetaBroker(ctx, nil, dataCoord, nil, cm)
	assert.Nil(t, err)

	segmentInfos, err := handler.getDataSegmentInfosByIDs(ctx, []int64{1})
	assert.Nil(t, err)
	assert.Equal(t, 1, len(segmentInfos))

	dataCoord.returnError = true
	segmentInfos2, err := handler.getDataSegmentInfosByIDs(ctx, []int64{1})
	assert.Error(t, err)
	assert.Empty(t, segmentInfos2)

	dataCoord.returnError = false
	dataCoord.returnGrpcError = true
	segmentInfos3, err := handler.getDataSegmentInfosByIDs(ctx, []int64{1})
	assert.Error(t, err)
	assert.Empty(t, segmentInfos3)

	cancel()
}
