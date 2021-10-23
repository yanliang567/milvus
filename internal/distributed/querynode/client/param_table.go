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

package grpcquerynodeclient

import (
	"strconv"
	"sync"

	"github.com/milvus-io/milvus/internal/distributed/grpcconfigs"
	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/util/paramtable"
)

// ParamTable is a derived struct of paramtable.BaseTable. It achieves Composition by
// embedding paramtable.BaseTable. It is used to quickly and easily access the system configuration.
type ParamTable struct {
	paramtable.BaseTable

	ClientMaxSendSize int
	ClientMaxRecvSize int
}

// Params is a package scoped variable of type ParamTable.
var Params ParamTable
var once sync.Once

// Init is an override method of BaseTable's Init. It mainly calls the
// Init of BaseTable and do some other initialization.
func (pt *ParamTable) Init() {
	once.Do(func() {
		pt.BaseTable.Init()

		pt.initClientMaxSendSize()
		pt.initClientMaxRecvSize()
	})
}

func (pt *ParamTable) initClientMaxSendSize() {
	var err error

	valueStr, err := pt.Load("queryNode.grpc.clientMaxSendSize")
	if err != nil { // not set
		pt.ClientMaxSendSize = grpcconfigs.DefaultClientMaxSendSize
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil { // not in valid format
		log.Warn("Failed to parse queryNode.grpc.clientMaxSendSize, set to default",
			zap.String("queryNode.grpc.clientMaxSendSize", valueStr),
			zap.Error(err))

		pt.ClientMaxSendSize = grpcconfigs.DefaultClientMaxSendSize
	} else {
		pt.ClientMaxSendSize = value
	}

	log.Debug("initClientMaxSendSize",
		zap.Int("queryNode.grpc.clientMaxSendSize", pt.ClientMaxSendSize))
}

func (pt *ParamTable) initClientMaxRecvSize() {
	var err error

	valueStr, err := pt.Load("queryNode.grpc.clientMaxRecvSize")
	if err != nil { // not set
		pt.ClientMaxRecvSize = grpcconfigs.DefaultClientMaxRecvSize
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil { // not in valid format
		log.Warn("Failed to parse queryNode.grpc.clientMaxRecvSize, set to default",
			zap.String("queryNode.grpc.clientMaxRecvSize", valueStr),
			zap.Error(err))

		pt.ClientMaxRecvSize = grpcconfigs.DefaultClientMaxRecvSize
	} else {
		pt.ClientMaxRecvSize = value
	}

	log.Debug("initClientMaxRecvSize",
		zap.Int("queryNode.grpc.clientMaxRecvSize", pt.ClientMaxRecvSize))
}
