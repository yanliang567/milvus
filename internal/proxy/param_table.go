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

package proxy

import (
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/util/paramtable"
)

const (
	// SuggestPulsarMaxMessageSize defines the maximum size of Pulsar message.
	SuggestPulsarMaxMessageSize = 5 * 1024 * 1024
)

// ParamTable is a derived struct of paramtable.BaseTable. It achieves Composition by
// embedding paramtable.BaseTable. It is used to quickly and easily access the system configuration.
type ParamTable struct {
	paramtable.BaseTable

	// NetworkPort & IP are not used
	NetworkPort int
	IP          string

	NetworkAddress string

	Alias string

	EtcdEndpoints    []string
	MetaRootPath     string
	RootCoordAddress string
	PulsarAddress    string

	RocksmqPath string // not used in Proxy

	ProxyID                  UniqueID
	TimeTickInterval         time.Duration
	MsgStreamTimeTickBufSize int64
	MaxNameLength            int64
	MaxFieldNum              int64
	MaxShardNum              int32
	MaxDimension             int64
	DefaultPartitionName     string
	DefaultIndexName         string

	// --- Channels ---
	ClusterChannelPrefix      string
	ProxyTimeTickChannelNames []string
	ProxySubName              string

	// required from query coord
	SearchResultChannelNames   []string
	RetrieveResultChannelNames []string

	MaxTaskNum int64

	PulsarMaxMessageSize int

	CreatedTime time.Time
	UpdatedTime time.Time
}

// Params is a package scoped variable of type ParamTable.
var Params ParamTable
var once sync.Once

// InitOnce ensures that Init is only called once.
func (pt *ParamTable) InitOnce() {
	once.Do(func() {
		pt.Init()
	})
}

// Init is an override method of BaseTable's Init. It mainly calls the
// Init of BaseTable and do some other initialization.
func (pt *ParamTable) Init() {
	pt.BaseTable.Init()
	err := pt.LoadYaml("advanced/proxy.yaml")
	if err != nil {
		panic(err)
	}

	pt.initEtcdEndpoints()
	pt.initMetaRootPath()
	pt.initPulsarAddress()
	pt.initRocksmqPath()
	pt.initTimeTickInterval()
	// Has to init global msgchannel prefix before other channel names
	pt.initClusterMsgChannelPrefix()
	pt.initProxySubName()
	pt.initProxyTimeTickChannelNames()
	pt.initMsgStreamTimeTickBufSize()
	pt.initMaxNameLength()
	pt.initMaxFieldNum()
	pt.initMaxShardNum()
	pt.initMaxDimension()
	pt.initDefaultPartitionName()
	pt.initDefaultIndexName()

	pt.initPulsarMaxMessageSize()

	pt.initMaxTaskNum()

	pt.initRoleName()
}

// InitAlias initialize Alias member.
func (pt *ParamTable) InitAlias(alias string) {
	pt.Alias = alias
}

func (pt *ParamTable) initPulsarAddress() {
	ret, err := pt.Load("_PulsarAddress")
	if err != nil {
		panic(err)
	}
	pt.PulsarAddress = ret
}

func (pt *ParamTable) initRocksmqPath() {
	path, err := pt.Load("_RocksmqPath")
	if err != nil {
		panic(err)
	}
	pt.RocksmqPath = path
}

func (pt *ParamTable) initTimeTickInterval() {
	intervalStr, err := pt.Load("proxy.timeTickInterval")
	if err != nil {
		panic(err)
	}
	interval, err := strconv.Atoi(intervalStr)
	if err != nil {
		panic(err)
	}
	pt.TimeTickInterval = time.Duration(interval) * time.Millisecond
}

func (pt *ParamTable) initClusterMsgChannelPrefix() {
	config, err := pt.Load("msgChannel.chanNamePrefix.cluster")
	if err != nil {
		panic(err)
	}
	pt.ClusterChannelPrefix = config
}

func (pt *ParamTable) initProxySubName() {
	config, err := pt.Load("msgChannel.subNamePrefix.proxySubNamePrefix")
	if err != nil {
		panic(err)
	}
	s := []string{pt.ClusterChannelPrefix, config, strconv.FormatInt(pt.ProxyID, 10)}
	pt.ProxySubName = strings.Join(s, "-")
}

func (pt *ParamTable) initProxyTimeTickChannelNames() {
	config, err := pt.Load("msgChannel.chanNamePrefix.proxyTimeTick")
	if err != nil {
		panic(err)
	}
	s := []string{pt.ClusterChannelPrefix, config, "0"}
	prefix := strings.Join(s, "-")
	pt.ProxyTimeTickChannelNames = []string{prefix}
}

func (pt *ParamTable) initMsgStreamTimeTickBufSize() {
	pt.MsgStreamTimeTickBufSize = pt.ParseInt64("proxy.msgStream.timeTick.bufSize")
}

func (pt *ParamTable) initMaxNameLength() {
	str, err := pt.Load("proxy.maxNameLength")
	if err != nil {
		panic(err)
	}
	maxNameLength, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	pt.MaxNameLength = maxNameLength
}

func (pt *ParamTable) initMaxShardNum() {
	str, err := pt.Load("proxy.maxShardNum")
	if err != nil {
		panic(err)
	}
	maxShardNum, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	pt.MaxShardNum = int32(maxShardNum)
}

func (pt *ParamTable) initMaxFieldNum() {
	str, err := pt.Load("proxy.maxFieldNum")
	if err != nil {
		panic(err)
	}
	maxFieldNum, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	pt.MaxFieldNum = maxFieldNum
}

func (pt *ParamTable) initMaxDimension() {
	str, err := pt.Load("proxy.maxDimension")
	if err != nil {
		panic(err)
	}
	maxDimension, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	pt.MaxDimension = maxDimension
}

func (pt *ParamTable) initDefaultPartitionName() {
	name, err := pt.Load("common.defaultPartitionName")
	if err != nil {
		panic(err)
	}
	pt.DefaultPartitionName = name
}

func (pt *ParamTable) initDefaultIndexName() {
	name, err := pt.Load("common.defaultIndexName")
	if err != nil {
		panic(err)
	}
	pt.DefaultIndexName = name
}

func (pt *ParamTable) initPulsarMaxMessageSize() {
	// pulsarHost, err := pt.Load("pulsar.address")
	// if err != nil {
	// 	panic(err)
	// }

	// pulsarRestPort, err := pt.Load("pulsar.rest-port")
	// if err != nil {
	// 	panic(err)
	// }

	// protocol := "http"
	// url := "/admin/v2/brokers/configuration/runtime"
	// runtimeConfig, err := GetPulsarConfig(protocol, pulsarHost, pulsarRestPort, url)
	// if err != nil {
	// 	panic(err)
	// }
	// maxMessageSizeStr := fmt.Sprintf("%v", runtimeConfig[PulsarMaxMessageSizeKey])
	// pt.PulsarMaxMessageSize, err = strconv.Atoi(maxMessageSizeStr)
	// if err != nil {
	// 	panic(err)
	// }

	maxMessageSizeStr, err := pt.Load("pulsar.maxMessageSize")
	if err != nil {
		pt.PulsarMaxMessageSize = SuggestPulsarMaxMessageSize
	} else {
		maxMessageSize, err := strconv.Atoi(maxMessageSizeStr)
		if err != nil {
			pt.PulsarMaxMessageSize = SuggestPulsarMaxMessageSize
		} else {
			pt.PulsarMaxMessageSize = maxMessageSize
		}
	}
}

func (pt *ParamTable) initRoleName() {
	pt.RoleName = "proxy"
}

func (pt *ParamTable) initEtcdEndpoints() {
	endpoints, err := pt.Load("_EtcdEndpoints")
	if err != nil {
		panic(err)
	}
	pt.EtcdEndpoints = strings.Split(endpoints, ",")
}

func (pt *ParamTable) initMetaRootPath() {
	rootPath, err := pt.Load("etcd.rootPath")
	if err != nil {
		panic(err)
	}
	subPath, err := pt.Load("etcd.metaSubPath")
	if err != nil {
		panic(err)
	}
	pt.MetaRootPath = path.Join(rootPath, subPath)
}

func (pt *ParamTable) initMaxTaskNum() {
	str, err := pt.Load("proxy.maxTaskNum")
	if err != nil {
		panic(err)
	}
	maxTaskNum, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	pt.MaxTaskNum = maxTaskNum
}
