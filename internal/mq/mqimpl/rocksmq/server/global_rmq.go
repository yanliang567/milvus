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

package server

import (
	"errors"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/paramtable"

	"go.uber.org/zap"
)

// Rmq is global rocksmq instance that will be initialized only once
var Rmq *rocksmq

// once is used to init global rocksmq
var once sync.Once

// Params provide params that rocksmq needs
var params paramtable.BaseTable

// InitRmq is deprecate implementation of global rocksmq. will be removed later
func InitRmq(rocksdbName string, idAllocator allocator.GIDAllocator) error {
	var err error
	params.Init()
	Rmq, err = NewRocksMQ(params, rocksdbName, idAllocator)
	return err
}

// InitRocksMQ init global rocksmq single instance
func InitRocksMQ(path string) error {
	var finalErr error
	once.Do(func() {
		params.Init()
		log.Debug("initializing global rmq", zap.String("path", path))
		var fi os.FileInfo
		fi, finalErr = os.Stat(path)
		if os.IsNotExist(finalErr) {
			finalErr = os.MkdirAll(path, os.ModePerm)
			if finalErr != nil {
				return
			}
		} else {
			if !fi.IsDir() {
				errMsg := "can't create a directory because there exists a file with the same name"
				finalErr = errors.New(errMsg)
				return
			}
		}

		rawRmqPageSize, err := params.Load("rocksmq.rocksmqPageSize")
		if err == nil && rawRmqPageSize != "" {
			rmqPageSize, err := strconv.ParseInt(rawRmqPageSize, 10, 64)
			if err == nil {
				atomic.StoreInt64(&RocksmqPageSize, rmqPageSize)
			} else {
				log.Warn("rocksmq.rocksmqPageSize is invalid, using default value 2G")
			}
		}

		Rmq, finalErr = NewRocksMQ(params, path, nil)
	})
	return finalErr
}

// CloseRocksMQ is used to close global rocksmq
func CloseRocksMQ() {
	log.Debug("Close Rocksmq!")
	if Rmq != nil && Rmq.store != nil {
		Rmq.Close()
	}
}
