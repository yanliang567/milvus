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

package msgstream

import (
	"errors"
	"strconv"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
)

// InsertRepackFunc is used to repack messages after hash by primary key
func InsertRepackFunc(tsMsgs []TsMsg, hashKeys [][]int32) (map[int32]*MsgPack, error) {
	result := make(map[int32]*MsgPack)
	for i, request := range tsMsgs {
		if request.Type() != commonpb.MsgType_Insert {
			return nil, errors.New("msg's must be Insert")
		}
		insertRequest := request.(*InsertMsg)
		keys := hashKeys[i]

		timestampLen := len(insertRequest.Timestamps)
		rowIDLen := len(insertRequest.RowIDs)
		rowDataLen := len(insertRequest.RowData)
		keysLen := len(keys)

		if keysLen != timestampLen || keysLen != rowIDLen || keysLen != rowDataLen {
			return nil, errors.New("the length of hashValue, timestamps, rowIDs, RowData are not equal")
		}
		for index, key := range keys {
			_, ok := result[key]
			if !ok {
				msgPack := MsgPack{}
				result[key] = &msgPack
			}

			sliceRequest := internalpb.InsertRequest{
				Base: &commonpb.MsgBase{
					MsgType:   commonpb.MsgType_Insert,
					MsgID:     insertRequest.Base.MsgID,
					Timestamp: insertRequest.Timestamps[index],
					SourceID:  insertRequest.Base.SourceID,
				},
				DbID:           insertRequest.DbID,
				CollectionID:   insertRequest.CollectionID,
				PartitionID:    insertRequest.PartitionID,
				CollectionName: insertRequest.CollectionName,
				PartitionName:  insertRequest.PartitionName,
				SegmentID:      insertRequest.SegmentID,
				ShardName:      insertRequest.ShardName,
				Timestamps:     []uint64{insertRequest.Timestamps[index]},
				RowIDs:         []int64{insertRequest.RowIDs[index]},
				RowData:        []*commonpb.Blob{insertRequest.RowData[index]},
			}

			insertMsg := &InsertMsg{
				BaseMsg: BaseMsg{
					Ctx: request.TraceCtx(),
				},
				InsertRequest: sliceRequest,
			}
			result[key].Msgs = append(result[key].Msgs, insertMsg)
		}
	}
	return result, nil
}

// DeleteRepackFunc is used to repack messages after hash by primary key
func DeleteRepackFunc(tsMsgs []TsMsg, hashKeys [][]int32) (map[int32]*MsgPack, error) {
	result := make(map[int32]*MsgPack)
	for i, request := range tsMsgs {
		if request.Type() != commonpb.MsgType_Delete {
			return nil, errors.New("msg's must be Delete")
		}
		deleteRequest := request.(*DeleteMsg)
		keys := hashKeys[i]

		if len(keys) != 1 {
			return nil, errors.New("len(msg.hashValue) must equal 1, but it is: " + strconv.Itoa(len(keys)))
		}

		timestampLen := len(deleteRequest.Timestamps)
		pkLen := len(deleteRequest.PrimaryKeys)
		keysLen := len(keys)

		if keysLen != timestampLen || keysLen != pkLen {
			return nil, errors.New("the length of hashValue, timestamps, primaryKeys are not equal")
		}

		for index, key := range keys {
			_, ok := result[key]
			if !ok {
				msgPack := MsgPack{}
				result[key] = &msgPack
			}

			sliceRequest := internalpb.DeleteRequest{
				Base: &commonpb.MsgBase{
					MsgType:   commonpb.MsgType_Delete,
					MsgID:     deleteRequest.Base.MsgID,
					Timestamp: deleteRequest.Timestamps[index],
					SourceID:  deleteRequest.Base.SourceID,
				},
				DbID:           deleteRequest.DbID,
				CollectionID:   deleteRequest.CollectionID,
				PartitionID:    deleteRequest.PartitionID,
				CollectionName: deleteRequest.CollectionName,
				PartitionName:  deleteRequest.PartitionName,
				ShardName:      deleteRequest.ShardName,
				Timestamps:     []uint64{deleteRequest.Timestamps[index]},
				PrimaryKeys:    []int64{deleteRequest.PrimaryKeys[index]},
			}

			deleteMsg := &DeleteMsg{
				BaseMsg: BaseMsg{
					Ctx: request.TraceCtx(),
				},
				DeleteRequest: sliceRequest,
			}
			result[key].Msgs = append(result[key].Msgs, deleteMsg)
		}
	}
	return result, nil
}

// DefaultRepackFunc is used to repack messages after hash by primary key
func DefaultRepackFunc(tsMsgs []TsMsg, hashKeys [][]int32) (map[int32]*MsgPack, error) {
	result := make(map[int32]*MsgPack)
	for i, request := range tsMsgs {
		keys := hashKeys[i]
		if len(keys) != 1 {
			return nil, errors.New("len(msg.hashValue) must equal 1, but it is: " + strconv.Itoa(len(keys)))
		}
		key := keys[0]
		_, ok := result[key]
		if !ok {
			msgPack := MsgPack{}
			result[key] = &msgPack
		}
		result[key].Msgs = append(result[key].Msgs, request)
	}
	return result, nil
}
