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
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	rowIDFieldID     FieldID = 0
	timestampFieldID FieldID = 1
)

type (
	// UniqueID is an identifier that is guaranteed to be unique among all the collections, partitions and segments
	UniqueID = typeutil.UniqueID
	// Timestamp is timestamp
	Timestamp = typeutil.Timestamp
	// FieldID is to uniquely identify the field
	FieldID = int64
	// IntPrimaryKey is the primary key of int type
	IntPrimaryKey = typeutil.IntPrimaryKey
	// DSL is the Domain Specific Language
	DSL = string
	// Channel is the virtual channel
	Channel = string
	// ConsumeSubName is consumer's subscription name of the message stream
	ConsumeSubName = string
)

// TimeRange is a range of time periods
type TimeRange struct {
	timestampMin Timestamp
	timestampMax Timestamp
}

// loadType is load collection or load partition
type loadType = querypb.LoadType

const (
	loadTypeCollection = querypb.LoadType_LoadCollection
	loadTypePartition  = querypb.LoadType_LoadPartition
)
