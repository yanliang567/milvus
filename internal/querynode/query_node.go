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

/*
#cgo pkg-config: milvus_segcore milvus_common

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
#include "segcore/segcore_init_c.h"
#include "common/init_c.h"

*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"path"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/concurrency"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/gc"
	"github.com/milvus-io/milvus/internal/util/hardware"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/panjf2000/ants/v2"
	"github.com/samber/lo"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
)

// make sure QueryNode implements types.QueryNode
var _ types.QueryNode = (*QueryNode)(nil)

// make sure QueryNode implements types.QueryNodeComponent
var _ types.QueryNodeComponent = (*QueryNode)(nil)

var Params *paramtable.ComponentParam = paramtable.Get()

// rateCol is global rateCollector in QueryNode.
var rateCol *rateCollector

// QueryNode communicates with outside services and union all
// services in querynode package.
//
// QueryNode implements `types.Component`, `types.QueryNode` interfaces.
//
//	`rootCoord` is a grpc client of root coordinator.
//	`indexCoord` is a grpc client of index coordinator.
//	`stateCode` is current statement of this query node, indicating whether it's healthy.
type QueryNode struct {
	queryNodeLoopCtx    context.Context
	queryNodeLoopCancel context.CancelFunc

	wg sync.WaitGroup

	stateCode atomic.Value
	stopOnce  sync.Once

	//call once
	initOnce sync.Once

	// internal components
	metaReplica ReplicaInterface

	// tSafeReplica
	tSafeReplica TSafeReplicaInterface

	// dataSyncService
	dataSyncService *dataSyncService

	// segment loader
	loader *segmentLoader

	// etcd client
	etcdCli *clientv3.Client
	address string

	factory   dependency.Factory
	scheduler *taskScheduler

	sessionMu sync.Mutex
	session   *sessionutil.Session
	eventCh   <-chan *sessionutil.SessionEvent

	vectorStorage storage.ChunkManager
	etcdKV        *etcdkv.EtcdKV

	// shard cluster service, handle shard leader functions
	ShardClusterService *ShardClusterService
	//shard query service, handles shard-level query & search
	queryShardService *queryShardService

	// pool for load/release channel
	taskPool *concurrency.Pool

	IsStandAlone bool
}

var queryNode *QueryNode = nil

func GetQueryNode() *QueryNode {
	return queryNode
}

// NewQueryNode will return a QueryNode with abnormal state.
func NewQueryNode(ctx context.Context, factory dependency.Factory) *QueryNode {
	ctx1, cancel := context.WithCancel(ctx)

	queryNode = &QueryNode{
		queryNodeLoopCtx:    ctx1,
		queryNodeLoopCancel: cancel,
		factory:             factory,
		IsStandAlone:        os.Getenv(metricsinfo.DeployModeEnvKey) == metricsinfo.StandaloneDeployMode,
	}

	queryNode.tSafeReplica = newTSafeReplica()
	queryNode.scheduler = newTaskScheduler(ctx1, queryNode.tSafeReplica)
	queryNode.UpdateStateCode(commonpb.StateCode_Abnormal)

	return queryNode
}

func (node *QueryNode) initSession() error {
	node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.EtcdCfg.MetaRootPath.GetValue(), node.etcdCli)
	if node.session == nil {
		return fmt.Errorf("session is nil, the etcd client connection may have failed")
	}
	node.session.Init(typeutil.QueryNodeRole, node.address, false, true)
	return nil
}

// Register register query node at etcd
func (node *QueryNode) Register() error {
	node.session.Register()
	// start liveness check
	go node.session.LivenessCheck(node.queryNodeLoopCtx, func() {
		log.Error("Query Node disconnected from etcd, process will exit", zap.Int64("Server Id", paramtable.GetNodeID()))
		if err := node.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
		// manually send signal to starter goroutine
		if node.session.TriggerKill {
			if p, err := os.FindProcess(os.Getpid()); err == nil {
				p.Signal(syscall.SIGINT)
			}
		}
	})

	//TODO Reset the logger
	//Params.initLogCfg()
	return nil
}

// initRateCollector creates and starts rateCollector in QueryNode.
func (node *QueryNode) initRateCollector() error {
	var err error
	rateCol, err = newRateCollector()
	if err != nil {
		return err
	}
	rateCol.Register(metricsinfo.NQPerSecond)
	rateCol.Register(metricsinfo.SearchThroughput)
	rateCol.Register(metricsinfo.InsertConsumeThroughput)
	rateCol.Register(metricsinfo.DeleteConsumeThroughput)
	return nil
}

// InitSegcore set init params of segCore, such as chunckRows, SIMD type...
func (node *QueryNode) InitSegcore() {
	cEasyloggingYaml := C.CString(path.Join(Params.BaseTable.GetConfigDir(), paramtable.DefaultEasyloggingYaml))
	C.SegcoreInit(cEasyloggingYaml)
	C.free(unsafe.Pointer(cEasyloggingYaml))

	cpuNum := runtime.GOMAXPROCS(0)
	C.SegcoreSetThreadPoolNum(C.uint32_t(cpuNum))

	// override segcore chunk size
	cChunkRows := C.int64_t(Params.QueryNodeCfg.ChunkRows.GetAsInt64())
	C.SegcoreSetChunkRows(cChunkRows)

	nlist := C.int64_t(Params.QueryNodeCfg.SmallIndexNlist.GetAsInt64())
	C.SegcoreSetNlist(nlist)

	nprobe := C.int64_t(Params.QueryNodeCfg.SmallIndexNProbe.GetAsInt64())
	C.SegcoreSetNprobe(nprobe)

	// override segcore SIMD type
	cSimdType := C.CString(Params.CommonCfg.SimdType.GetValue())
	C.SegcoreSetSimdType(cSimdType)
	C.free(unsafe.Pointer(cSimdType))

	// override segcore index slice size
	cIndexSliceSize := C.int64_t(Params.CommonCfg.IndexSliceSize.GetAsInt64())
	C.InitIndexSliceSize(cIndexSliceSize)

	cThreadCoreCoefficient := C.int64_t(Params.CommonCfg.ThreadCoreCoefficient.GetAsInt64())
	C.InitThreadCoreCoefficient(cThreadCoreCoefficient)

	cCPUNum := C.int(hardware.GetCPUNum())
	C.InitCpuNum(cCPUNum)

	initcore.InitLocalStorageConfig(Params)
}

// Init function init historical and streaming module to manage segments
func (node *QueryNode) Init() error {
	var initError error
	node.initOnce.Do(func() {
		//ctx := context.Background()
		log.Info("QueryNode session info", zap.String("metaPath", Params.EtcdCfg.MetaRootPath.GetValue()))
		err := node.initSession()
		if err != nil {
			log.Error("QueryNode init session failed", zap.Error(err))
			initError = err
			return
		}

		node.factory.Init(Params)

		err = node.initRateCollector()
		if err != nil {
			log.Error("QueryNode init rateCollector failed", zap.Int64("nodeID", paramtable.GetNodeID()), zap.Error(err))
			initError = err
			return
		}
		log.Info("QueryNode init rateCollector done", zap.Int64("nodeID", paramtable.GetNodeID()))

		node.vectorStorage, err = node.factory.NewPersistentStorageChunkManager(node.queryNodeLoopCtx)
		if err != nil {
			log.Error("QueryNode init vector storage failed", zap.Error(err))
			initError = err
			return
		}

		node.etcdKV = etcdkv.NewEtcdKV(node.etcdCli, Params.EtcdCfg.MetaRootPath.GetValue())
		log.Info("queryNode try to connect etcd success", zap.Any("MetaRootPath", Params.EtcdCfg.MetaRootPath))

		cpuNum := runtime.GOMAXPROCS(0)

		node.taskPool, err = concurrency.NewPool(cpuNum, ants.WithPreAlloc(true))
		if err != nil {
			log.Error("QueryNode init channel pool failed", zap.Error(err))
			initError = err
			return
		}

		node.metaReplica = newCollectionReplica()

		node.loader = newSegmentLoader(
			node.metaReplica,
			node.etcdKV,
			node.vectorStorage,
			node.factory)

		node.dataSyncService = newDataSyncService(node.queryNodeLoopCtx, node.metaReplica, node.tSafeReplica, node.factory)

		node.InitSegcore()

		if Params.QueryNodeCfg.GCHelperEnabled.GetAsBool() {
			action := func(GOGC uint32) {
				debug.SetGCPercent(int(GOGC))
			}
			gc.NewTuner(Params.QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(), uint32(Params.QueryNodeCfg.MinimumGOGCConfig.GetAsInt()), uint32(Params.QueryNodeCfg.MaximumGOGCConfig.GetAsInt()), action)
		} else {
			action := func(uint32) {}
			gc.NewTuner(Params.QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(), uint32(Params.QueryNodeCfg.MinimumGOGCConfig.GetAsInt()), uint32(Params.QueryNodeCfg.MaximumGOGCConfig.GetAsInt()), action)
		}

		log.Info("query node init successfully",
			zap.Int64("queryNodeID", paramtable.GetNodeID()),
			zap.String("Address", node.address),
		)
	})

	return initError
}

// Start mainly start QueryNode's query service.
func (node *QueryNode) Start() error {
	// start task scheduler
	go node.scheduler.Start()

	// create shardClusterService for shardLeader functions.
	node.ShardClusterService = newShardClusterService(node.etcdCli, node.session, node)
	// create shard-level query service
	queryShardService, err := newQueryShardService(node.queryNodeLoopCtx, node.metaReplica, node.tSafeReplica,
		node.ShardClusterService, node.factory, node.scheduler)
	if err != nil {
		return err
	}
	node.queryShardService = queryShardService

	node.UpdateStateCode(commonpb.StateCode_Healthy)
	log.Info("query node start successfully",
		zap.Int64("queryNodeID", paramtable.GetNodeID()),
		zap.String("Address", node.address),
	)
	return nil
}

// Stop mainly stop QueryNode's query service, historical loop and streaming loop.
func (node *QueryNode) Stop() error {
	node.stopOnce.Do(func() {
		log.Warn("Query node stop..")
		node.UpdateStateCode(commonpb.StateCode_Stopping)
		err := node.session.GoingStop()
		if err != nil {
			log.Warn("session fail to go stopping state", zap.Error(err))
		} else {
			noSegmentChan := node.metaReplica.getNoSegmentChan()
			select {
			case <-noSegmentChan:
			case <-time.After(Params.QueryNodeCfg.GracefulStopTimeout.GetAsDuration(time.Second)):
				log.Warn("migrate data timed out", zap.Int64("server_id", paramtable.GetNodeID()),
					zap.Int64s("sealed_segment", lo.Map(node.metaReplica.getSealedSegments(), func(t *Segment, i int) int64 {
						return t.ID()
					})),
					zap.Int64s("growing_segment", lo.Map(node.metaReplica.getGrowingSegments(), func(t *Segment, i int) int64 {
						return t.ID()
					})),
				)
			}
		}

		node.UpdateStateCode(commonpb.StateCode_Abnormal)
		node.wg.Wait()
		node.queryNodeLoopCancel()

		// close services
		if node.dataSyncService != nil {
			node.dataSyncService.close()
		}

		if node.metaReplica != nil {
			node.metaReplica.freeAll()
		}

		if node.ShardClusterService != nil {
			node.ShardClusterService.close()
		}

		if node.queryShardService != nil {
			node.queryShardService.close()
		}

		node.session.Revoke(time.Second)
	})

	return nil
}

// UpdateStateCode updata the state of query node, which can be initializing, healthy, and abnormal
func (node *QueryNode) UpdateStateCode(code commonpb.StateCode) {
	node.stateCode.Store(code)
}

// SetEtcdClient assigns parameter client to its member etcdCli
func (node *QueryNode) SetEtcdClient(client *clientv3.Client) {
	node.etcdCli = client
}

func (node *QueryNode) SetAddress(address string) {
	node.address = address
}

func (node *QueryNode) GetAddress() string {
	return node.address
}
