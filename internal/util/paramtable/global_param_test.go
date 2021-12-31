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

package paramtable

import (
	"fmt"
	"log"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/stretchr/testify/assert"
)

func shouldPanic(t *testing.T, name string, f func()) {
	defer func() { recover() }()
	f()
	t.Errorf("%s should have panicked", name)
}

func TestGlobalParamTable(t *testing.T) {
	var GlobalParams GlobalParamTable
	GlobalParams.Init()

	t.Run("test rootCoordConfig", func(t *testing.T) {
		Params := GlobalParams.RootCoordCfg

		assert.NotEqual(t, Params.PulsarAddress, "")
		t.Logf("pulsar address = %s", Params.PulsarAddress)

		assert.NotEqual(t, Params.MetaRootPath, "")
		t.Logf("meta root path = %s", Params.MetaRootPath)

		assert.NotEqual(t, Params.KvRootPath, "")
		t.Logf("kv root path = %s", Params.KvRootPath)

		assert.Equal(t, Params.MsgChannelSubName, "by-dev-rootCoord")
		t.Logf("msg channel sub name = %s", Params.MsgChannelSubName)

		assert.Equal(t, Params.TimeTickChannel, "by-dev-rootcoord-timetick")
		t.Logf("master time tick channel = %s", Params.TimeTickChannel)

		assert.Equal(t, Params.StatisticsChannel, "by-dev-rootcoord-statistics")
		t.Logf("master statistics channel = %s", Params.StatisticsChannel)

		assert.Equal(t, Params.DmlChannelName, "by-dev-rootcoord-dml")
		t.Logf("dml channel = %s", Params.DmlChannelName)

		assert.Equal(t, Params.DeltaChannelName, "by-dev-rootcoord-delta")
		t.Logf("delta channel = %s", Params.DeltaChannelName)

		assert.NotEqual(t, Params.MaxPartitionNum, 0)
		t.Logf("master MaxPartitionNum = %d", Params.MaxPartitionNum)

		assert.NotEqual(t, Params.MinSegmentSizeToEnableIndex, 0)
		t.Logf("master MinSegmentSizeToEnableIndex = %d", Params.MinSegmentSizeToEnableIndex)

		assert.NotEqual(t, Params.DefaultPartitionName, "")
		t.Logf("default partition name = %s", Params.DefaultPartitionName)

		assert.NotEqual(t, Params.DefaultIndexName, "")
		t.Logf("default index name = %s", Params.DefaultIndexName)

		assert.NotZero(t, Params.Timeout)
		t.Logf("master timeout = %d", Params.Timeout)

		Params.CreatedTime = time.Now()
		Params.UpdatedTime = time.Now()
		t.Logf("created time: %v", Params.CreatedTime)
		t.Logf("updated time: %v", Params.UpdatedTime)
	})

	t.Run("test proxyConfig", func(t *testing.T) {
		Params := GlobalParams.ProxyCfg

		t.Logf("MetaRootPath: %s", Params.MetaRootPath)

		t.Logf("PulsarAddress: %s", Params.PulsarAddress)

		t.Logf("RocksmqPath: %s", Params.RocksmqPath)

		t.Logf("TimeTickInterval: %v", Params.TimeTickInterval)

		assert.Equal(t, Params.ProxySubName, "by-dev-proxy-0")
		t.Logf("ProxySubName: %s", Params.ProxySubName)

		assert.Equal(t, Params.ProxyTimeTickChannelNames, []string{"by-dev-proxyTimeTick-0"})
		t.Logf("ProxyTimeTickChannelNames: %v", Params.ProxyTimeTickChannelNames)

		t.Logf("MsgStreamTimeTickBufSize: %d", Params.MsgStreamTimeTickBufSize)

		t.Logf("MaxNameLength: %d", Params.MaxNameLength)

		t.Logf("MaxFieldNum: %d", Params.MaxFieldNum)

		t.Logf("MaxShardNum: %d", Params.MaxShardNum)

		t.Logf("MaxDimension: %d", Params.MaxDimension)

		t.Logf("DefaultPartitionName: %s", Params.DefaultPartitionName)

		t.Logf("DefaultIndexName: %s", Params.DefaultIndexName)

		t.Logf("PulsarMaxMessageSize: %d", Params.PulsarMaxMessageSize)

		//t.Logf("RoleName: %s", typeutil.ProxyRole)

		t.Logf("MaxTaskNum: %d", Params.MaxTaskNum)
	})

	t.Run("test proxyConfig panic", func(t *testing.T) {
		Params := GlobalParams.ProxyCfg

		shouldPanic(t, "proxy.timeTickInterval", func() {
			Params.BaseParams.Save("proxy.timeTickInterval", "")
			Params.initTimeTickInterval()
		})

		shouldPanic(t, "proxy.msgStream.timeTick.bufSize", func() {
			Params.BaseParams.Save("proxy.msgStream.timeTick.bufSize", "abc")
			Params.initMsgStreamTimeTickBufSize()
		})

		shouldPanic(t, "proxy.maxNameLength", func() {
			Params.BaseParams.Save("proxy.maxNameLength", "abc")
			Params.initMaxNameLength()
		})

		shouldPanic(t, "proxy.maxFieldNum", func() {
			Params.BaseParams.Save("proxy.maxFieldNum", "abc")
			Params.initMaxFieldNum()
		})

		shouldPanic(t, "proxy.maxShardNum", func() {
			Params.BaseParams.Save("proxy.maxShardNum", "abc")
			Params.initMaxShardNum()
		})

		shouldPanic(t, "proxy.maxDimension", func() {
			Params.BaseParams.Save("proxy.maxDimension", "-asdf")
			Params.initMaxDimension()
		})

		shouldPanic(t, "proxy.maxTaskNum", func() {
			Params.BaseParams.Save("proxy.maxTaskNum", "-asdf")
			Params.initMaxTaskNum()
		})
	})

	t.Run("test queryCoordConfig", func(t *testing.T) {
		Params := GlobalParams.QueryCoordCfg

		assert.Equal(t, Params.SearchChannelPrefix, "by-dev-search")
		t.Logf("query coord search channel = %s", Params.SearchChannelPrefix)

		assert.Equal(t, Params.SearchResultChannelPrefix, "by-dev-searchResult")
		t.Logf("query coord search result channel = %s", Params.SearchResultChannelPrefix)

		assert.Equal(t, Params.StatsChannelName, "by-dev-query-node-stats")
		t.Logf("query coord stats channel = %s", Params.StatsChannelName)

		assert.Equal(t, Params.TimeTickChannelName, "by-dev-queryTimeTick")
		t.Logf("query coord  time tick channel = %s", Params.TimeTickChannelName)
	})

	t.Run("test queryNodeConfig", func(t *testing.T) {
		Params := GlobalParams.QueryNodeCfg

		address := Params.PulsarAddress
		split := strings.Split(address, ":")
		assert.Equal(t, "pulsar", split[0])
		assert.Equal(t, "6650", split[len(split)-1])

		cacheSize := Params.CacheSize
		assert.Equal(t, int64(32), cacheSize)
		err := os.Setenv("CACHE_SIZE", "2")
		assert.NoError(t, err)
		Params.initCacheSize()
		assert.Equal(t, int64(2), Params.CacheSize)
		err = os.Setenv("CACHE_SIZE", "32")
		assert.NoError(t, err)
		Params.initCacheSize()
		assert.Equal(t, int64(32), Params.CacheSize)

		endPoint := Params.MinioEndPoint
		equal := endPoint == "localhost:9000" || endPoint == "minio:9000"
		assert.Equal(t, equal, true)

		accessKeyID := Params.MinioAccessKeyID
		assert.Equal(t, accessKeyID, "minioadmin")

		secretAccessKey := Params.MinioSecretAccessKey
		assert.Equal(t, secretAccessKey, "minioadmin")

		useSSL := Params.MinioUseSSLStr
		assert.Equal(t, useSSL, false)

		interval := Params.StatsPublishInterval
		assert.Equal(t, 1000, interval)

		bufSize := Params.SearchReceiveBufSize
		assert.Equal(t, int64(512), bufSize)

		bufSize = Params.SearchResultReceiveBufSize
		assert.Equal(t, int64(64), bufSize)

		bufSize = Params.SearchPulsarBufSize
		assert.Equal(t, int64(512), bufSize)

		length := Params.FlowGraphMaxQueueLength
		assert.Equal(t, int32(1024), length)

		maxParallelism := Params.FlowGraphMaxParallelism
		assert.Equal(t, int32(1024), maxParallelism)

		Params.QueryNodeID = 3
		Params.initMsgChannelSubName()
		name := Params.MsgChannelSubName
		assert.Equal(t, name, "by-dev-queryNode-3")

		name = Params.StatsChannelName
		assert.Equal(t, name, "by-dev-query-node-stats")

		ch := Params.QueryTimeTickChannelName
		assert.Equal(t, ch, "by-dev-queryTimeTick")

		path := Params.MetaRootPath
		fmt.Println(path)
	})

	t.Run("test dataCoordConfig", func(t *testing.T) {
		Params := GlobalParams.DataCoordCfg

		assert.Equal(t, Params.InsertChannelPrefixName, "by-dev-insert-channel-")
		t.Logf("DataCoord insert channel = %s", Params.InsertChannelPrefixName)

		assert.Equal(t, Params.TimeTickChannelName, "by-dev-datacoord-timetick-channel")
		t.Logf("DataCoord timetick channel = %s", Params.TimeTickChannelName)

		assert.Equal(t, Params.SegmentInfoChannelName, "by-dev-segment-info-channel")
		t.Logf("DataCoord segment info channel = %s", Params.SegmentInfoChannelName)

		assert.Equal(t, Params.DataCoordSubscriptionName, "by-dev-dataCoord")
		t.Logf("DataCoord subscription channel = %s", Params.DataCoordSubscriptionName)
	})

	t.Run("test dataNodeConfig", func(t *testing.T) {
		Params := GlobalParams.DataNodeCfg

		Params.NodeID = 2
		Params.initMsgChannelSubName()

		id := Params.NodeID
		log.Println("NodeID:", id)

		alias := Params.Alias
		log.Println("Alias:", alias)

		length := Params.FlowGraphMaxQueueLength
		log.Println("flowGraphMaxQueueLength:", length)

		maxParallelism := Params.FlowGraphMaxParallelism
		log.Println("flowGraphMaxParallelism:", maxParallelism)

		size := Params.FlushInsertBufferSize
		log.Println("FlushInsertBufferSize:", size)

		path1 := Params.InsertBinlogRootPath
		log.Println("InsertBinlogRootPath:", path1)

		address := Params.PulsarAddress
		log.Println("PulsarAddress:", address)

		path1 = Params.ClusterChannelPrefix
		assert.Equal(t, path1, "by-dev")
		log.Println("ClusterChannelPrefix:", Params.ClusterChannelPrefix)

		name := Params.TimeTickChannelName
		assert.Equal(t, name, "by-dev-datacoord-timetick-channel")
		log.Println("TimeTickChannelName:", name)

		name = Params.MsgChannelSubName
		assert.Equal(t, name, "by-dev-dataNode-2")
		log.Println("MsgChannelSubName:", name)

		path1 = Params.MetaRootPath
		log.Println("MetaRootPath:", path1)

		id1 := Params.MinioAccessKeyID
		log.Println("MinioAccessKeyID:", id1)

		key := Params.MinioSecretAccessKey
		log.Println("MinioSecretAccessKey:", key)

		useSSL := Params.MinioUseSSL
		log.Println("MinioUseSSL:", useSSL)

		name = Params.MinioBucketName
		log.Println("MinioBucketName:", name)

		Params.CreatedTime = time.Now()
		log.Println("CreatedTime: ", Params.CreatedTime)

		Params.UpdatedTime = time.Now()
		log.Println("UpdatedTime: ", Params.UpdatedTime)

		assert.Equal(t, path.Join("files", "insert_log"), Params.InsertBinlogRootPath)

		assert.Equal(t, path.Join("files", "stats_log"), Params.StatsBinlogRootPath)
	})

	t.Run("test indexCoordConfig", func(t *testing.T) {
		Params := GlobalParams.IndexCoordCfg

		t.Logf("Address: %v", Params.Address)

		t.Logf("Port: %v", Params.Port)

		t.Logf("KvRootPath: %v", Params.KvRootPath)

		t.Logf("MetaRootPath: %v", Params.MetaRootPath)

		t.Logf("MinIOAddress: %v", Params.MinIOAddress)

		t.Logf("MinIOAccessKeyID: %v", Params.MinIOAccessKeyID)

		t.Logf("MinIOSecretAccessKey: %v", Params.MinIOSecretAccessKey)

		t.Logf("MinIOUseSSL: %v", Params.MinIOUseSSL)

		t.Logf("MinioBucketName: %v", Params.MinioBucketName)

		Params.CreatedTime = time.Now()
		t.Logf("CreatedTime: %v", Params.CreatedTime)

		Params.UpdatedTime = time.Now()
		t.Logf("UpdatedTime: %v", Params.UpdatedTime)

		t.Logf("IndexStorageRootPath: %v", Params.IndexStorageRootPath)
	})

	t.Run("test indexNodeConfig", func(t *testing.T) {
		Params := GlobalParams.IndexNodeCfg

		t.Logf("IP: %v", Params.IP)

		t.Logf("Address: %v", Params.Address)

		t.Logf("Port: %v", Params.Port)

		t.Logf("NodeID: %v", Params.NodeID)

		t.Logf("Alias: %v", Params.Alias)

		t.Logf("MetaRootPath: %v", Params.MetaRootPath)

		t.Logf("MinIOAddress: %v", Params.MinIOAddress)

		t.Logf("MinIOAccessKeyID: %v", Params.MinIOAccessKeyID)

		t.Logf("MinIOSecretAccessKey: %v", Params.MinIOSecretAccessKey)

		t.Logf("MinIOUseSSL: %v", Params.MinIOUseSSL)

		t.Logf("MinioBucketName: %v", Params.MinioBucketName)

		t.Logf("SimdType: %v", Params.SimdType)

		Params.CreatedTime = time.Now()
		t.Logf("CreatedTime: %v", Params.CreatedTime)

		Params.UpdatedTime = time.Now()
		t.Logf("UpdatedTime: %v", Params.UpdatedTime)

		t.Logf("IndexStorageRootPath: %v", Params.IndexStorageRootPath)
	})
}

func TestGrpcServerParams(t *testing.T) {
	role := typeutil.DataNodeRole
	var Params GrpcServerConfig
	Params.InitOnce(role)

	assert.Equal(t, Params.Domain, role)
	t.Logf("Domain = %s", Params.Domain)

	assert.NotEqual(t, Params.IP, "")
	t.Logf("IP = %s", Params.IP)

	assert.NotZero(t, Params.Port)
	t.Logf("Port = %d", Params.Port)

	t.Logf("Address = %s", Params.GetAddress())

	assert.NotNil(t, Params.Listener)
	t.Logf("Listener = %d", Params.Listener)

	assert.NotZero(t, Params.ServerMaxRecvSize)
	t.Logf("ServerMaxRecvSize = %d", Params.ServerMaxRecvSize)

	Params.Remove(role + ".grpc.serverMaxRecvSize")
	Params.initServerMaxRecvSize()
	assert.Equal(t, Params.ServerMaxRecvSize, DefaultServerMaxRecvSize)

	assert.NotZero(t, Params.ServerMaxSendSize)
	t.Logf("ServerMaxSendSize = %d", Params.ServerMaxSendSize)

	Params.Remove(role + ".grpc.serverMaxSendSize")
	Params.initServerMaxSendSize()
	assert.Equal(t, Params.ServerMaxSendSize, DefaultServerMaxSendSize)
}

func TestGrpcClientParams(t *testing.T) {
	role := typeutil.DataNodeRole
	var Params GrpcClientConfig
	Params.InitOnce(role)

	assert.Equal(t, Params.Domain, role)
	t.Logf("Domain = %s", Params.Domain)

	assert.NotEqual(t, Params.IP, "")
	t.Logf("IP = %s", Params.IP)

	assert.NotZero(t, Params.Port)
	t.Logf("Port = %d", Params.Port)

	t.Logf("Address = %s", Params.GetAddress())

	assert.NotNil(t, Params.Listener)
	t.Logf("Listener = %d", Params.Listener)

	assert.NotZero(t, Params.ClientMaxRecvSize)
	t.Logf("ClientMaxRecvSize = %d", Params.ClientMaxRecvSize)

	Params.Remove(role + ".grpc.clientMaxRecvSize")
	Params.initClientMaxRecvSize()
	assert.Equal(t, Params.ClientMaxRecvSize, DefaultClientMaxRecvSize)

	assert.NotZero(t, Params.ClientMaxSendSize)
	t.Logf("ClientMaxSendSize = %d", Params.ClientMaxSendSize)

	Params.Remove(role + ".grpc.clientMaxSendSize")
	Params.initClientMaxSendSize()
	assert.Equal(t, Params.ClientMaxSendSize, DefaultClientMaxSendSize)
}

func TestCheckPortAvailable(t *testing.T) {
	num := 10
	for i := 0; i < num; i++ {
		port := GetAvailablePort()
		assert.Equal(t, CheckPortAvailable(port), true)
	}
}
