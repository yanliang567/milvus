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

package rocksmq

import (
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"go.uber.org/zap"

	rocksdbkv "github.com/milvus-io/milvus/internal/kv/rocksdb"
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
	Rmq, err = NewRocksMQ(rocksdbName, idAllocator)
	return err
}

// InitRocksMQ init global rocksmq single instance
func InitRocksMQ() error {
	var err error
	once.Do(func() {
		params.Init()
		rocksdbName, _ := params.Load("_RocksmqPath")
		log.Debug("RocksmqPath=" + rocksdbName)
		_, err = os.Stat(rocksdbName)
		if os.IsNotExist(err) {
			err = os.MkdirAll(rocksdbName, os.ModePerm)
			if err != nil {
				errMsg := "Create dir " + rocksdbName + " failed"
				panic(errMsg)
			}
		}

		kvname := rocksdbName + "_kv"
		rocksdbKV, err := rocksdbkv.NewRocksdbKV(kvname)
		if err != nil {
			panic(err)
		}
		idAllocator := allocator.NewGlobalIDAllocator("rmq_id", rocksdbKV)
		_ = idAllocator.Initialize()

		rawRmqPageSize, err := params.Load("rocksmq.rocksmqPageSize")
		if err == nil && rawRmqPageSize != "" {
			rmqPageSize, err := strconv.ParseInt(rawRmqPageSize, 10, 64)
			if err == nil {
				atomic.StoreInt64(&RocksmqPageSize, rmqPageSize)
			} else {
				log.Warn("rocksmq.rocksmqPageSize is invalid, using default value 2G")
			}
		}
		rawRmqRetentionTimeInMinutes, err := params.Load("rocksmq.retentionTimeInMinutes")
		if err == nil && rawRmqRetentionTimeInMinutes != "" {
			rawRmqRetentionTimeInMinutes, err := strconv.ParseInt(rawRmqRetentionTimeInMinutes, 10, 64)
			if err == nil {
				atomic.StoreInt64(&RocksmqRetentionTimeInMinutes, rawRmqRetentionTimeInMinutes)
			} else {
				log.Warn("rocksmq.retentionTimeInMinutes is invalid, using default value 3 days")
			}
		}
		rawRmqRetentionSizeInMB, err := params.Load("rocksmq.retentionSizeInMB")
		if err == nil && rawRmqRetentionSizeInMB != "" {
			rawRmqRetentionSizeInMB, err := strconv.ParseInt(rawRmqRetentionSizeInMB, 10, 64)
			if err == nil {
				atomic.StoreInt64(&RocksmqRetentionSizeInMB, rawRmqRetentionSizeInMB)
			} else {
				log.Warn("rocksmq.retentionSizeInMB is invalid, using default value 0")
			}
		}
		log.Debug("", zap.Any("RocksmqRetentionTimeInMinutes", RocksmqRetentionTimeInMinutes),
			zap.Any("RocksmqRetentionSizeInMB", RocksmqRetentionSizeInMB), zap.Any("RocksmqPageSize", RocksmqPageSize))
		Rmq, err = NewRocksMQ(rocksdbName, idAllocator)
		if err != nil {
			panic(err)
		}
	})
	return err
}

// CloseRocksMQ is used to close global rocksmq
func CloseRocksMQ() {
	log.Debug("Close Rocksmq!")
	if Rmq != nil && Rmq.store != nil {
		Rmq.Close()
	}
}
