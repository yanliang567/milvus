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

#cgo CFLAGS: -I${SRCDIR}/../core/output/include

#cgo LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
#include "segcore/segcore_init_c.h"

*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/kv"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
)

// make sure QueryNode implements types.QueryNode
var _ types.QueryNode = (*QueryNode)(nil)

// make sure QueryNode implements types.QueryNodeComponent
var _ types.QueryNodeComponent = (*QueryNode)(nil)

var Params paramtable.GlobalParamTable

// QueryNode communicates with outside services and union all
// services in querynode package.
//
// QueryNode implements `types.Component`, `types.QueryNode` interfaces.
//  `rootCoord` is a grpc client of root coordinator.
//  `indexCoord` is a grpc client of index coordinator.
//  `stateCode` is current statement of this query node, indicating whether it's healthy.
type QueryNode struct {
	queryNodeLoopCtx    context.Context
	queryNodeLoopCancel context.CancelFunc

	stateCode atomic.Value

	//call once
	initOnce sync.Once

	// internal components
	historical *historical
	streaming  *streaming

	// tSafeReplica
	tSafeReplica TSafeReplicaInterface

	// dataSyncService
	dataSyncService *dataSyncService

	// internal services
	queryService *queryService
	statsService *statsService

	// segment loader
	loader *segmentLoader

	// etcd client
	etcdCli *clientv3.Client

	// clients
	rootCoord  types.RootCoord
	indexCoord types.IndexCoord

	msFactory msgstream.Factory
	scheduler *taskScheduler

	session *sessionutil.Session

	minioKV kv.BaseKV // minio minioKV
	etcdKV  *etcdkv.EtcdKV
}

// NewQueryNode will return a QueryNode with abnormal state.
func NewQueryNode(ctx context.Context, factory msgstream.Factory) *QueryNode {
	ctx1, cancel := context.WithCancel(ctx)
	node := &QueryNode{
		queryNodeLoopCtx:    ctx1,
		queryNodeLoopCancel: cancel,
		queryService:        nil,
		msFactory:           factory,
	}

	node.scheduler = newTaskScheduler(ctx1)
	node.UpdateStateCode(internalpb.StateCode_Abnormal)

	return node
}

func (node *QueryNode) initSession() error {
	node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.QueryNodeCfg.MetaRootPath, node.etcdCli)
	if node.session == nil {
		return fmt.Errorf("session is nil, the etcd client connection may have failed")
	}
	node.session.Init(typeutil.QueryNodeRole, Params.QueryNodeCfg.QueryNodeIP+":"+strconv.FormatInt(Params.QueryNodeCfg.QueryNodePort, 10), false, true)
	Params.QueryNodeCfg.QueryNodeID = node.session.ServerID
	Params.BaseParams.SetLogger(Params.QueryNodeCfg.QueryNodeID)
	log.Debug("QueryNode", zap.Int64("nodeID", Params.QueryNodeCfg.QueryNodeID), zap.String("node address", node.session.Address))
	return nil
}

// Register register query node at etcd
func (node *QueryNode) Register() error {
	node.session.Register()
	// start liveness check
	go node.session.LivenessCheck(node.queryNodeLoopCtx, func() {
		log.Error("Query Node disconnected from etcd, process will exit", zap.Int64("Server Id", node.session.ServerID))
		if err := node.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
		// manually send signal to starter goroutine
		if node.session.TriggerKill {
			syscall.Kill(syscall.Getpid(), syscall.SIGINT)
		}
	})

	//TODO Reset the logger
	//Params.initLogCfg()
	return nil
}

// InitSegcore set init params of segCore, such as chunckRows, SIMD type...
func (node *QueryNode) InitSegcore() {
	C.SegcoreInit()

	// override segcore chunk size
	cChunkRows := C.int64_t(Params.QueryNodeCfg.ChunkRows)
	C.SegcoreSetChunkRows(cChunkRows)

	// override segcore SIMD type
	cSimdType := C.CString(Params.QueryNodeCfg.SimdType)
	cRealSimdType := C.SegcoreSetSimdType(cSimdType)
	Params.QueryNodeCfg.SimdType = C.GoString(cRealSimdType)
	C.free(unsafe.Pointer(cRealSimdType))
	C.free(unsafe.Pointer(cSimdType))
}

// Init function init historical and streaming module to manage segments
func (node *QueryNode) Init() error {
	var initError error = nil
	node.initOnce.Do(func() {
		//ctx := context.Background()
		log.Debug("QueryNode session info", zap.String("metaPath", Params.QueryNodeCfg.MetaRootPath))
		err := node.initSession()
		if err != nil {
			log.Error("QueryNode init session failed", zap.Error(err))
			initError = err
			return
		}
		Params.QueryNodeCfg.Refresh()

		node.etcdKV = etcdkv.NewEtcdKV(node.etcdCli, Params.QueryNodeCfg.MetaRootPath)
		log.Debug("queryNode try to connect etcd success", zap.Any("MetaRootPath", Params.QueryNodeCfg.MetaRootPath))
		node.tSafeReplica = newTSafeReplica()

		streamingReplica := newCollectionReplica(node.etcdKV)
		historicalReplica := newCollectionReplica(node.etcdKV)

		node.historical = newHistorical(node.queryNodeLoopCtx,
			historicalReplica,
			node.etcdKV,
			node.tSafeReplica,
		)
		node.streaming = newStreaming(node.queryNodeLoopCtx,
			streamingReplica,
			node.msFactory,
			node.etcdKV,
			node.tSafeReplica,
		)

		node.loader = newSegmentLoader(node.queryNodeLoopCtx,
			node.rootCoord,
			node.indexCoord,
			node.historical.replica,
			node.streaming.replica,
			node.etcdKV,
			node.msFactory)

		node.statsService = newStatsService(node.queryNodeLoopCtx, node.historical.replica, node.loader.indexLoader.fieldStatsChan, node.msFactory)
		node.dataSyncService = newDataSyncService(node.queryNodeLoopCtx, streamingReplica, historicalReplica, node.tSafeReplica, node.msFactory)

		node.InitSegcore()

		if node.rootCoord == nil {
			initError = errors.New("null root coordinator detected when queryNode init")
			return
		}

		if node.indexCoord == nil {
			initError = errors.New("null index coordinator detected when queryNode init")
			return
		}

		log.Debug("query node init successfully",
			zap.Any("queryNodeID", Params.QueryNodeCfg.QueryNodeID),
			zap.Any("IP", Params.QueryNodeCfg.QueryNodeIP),
			zap.Any("Port", Params.QueryNodeCfg.QueryNodePort),
		)
	})

	return initError
}

// Start mainly start QueryNode's query service.
func (node *QueryNode) Start() error {
	var err error
	m := map[string]interface{}{
		"PulsarAddress":  Params.QueryNodeCfg.PulsarAddress,
		"ReceiveBufSize": 1024,
		"PulsarBufSize":  1024}
	err = node.msFactory.SetParams(m)
	if err != nil {
		return err
	}

	// init services and manager
	// TODO: pass node.streaming.replica to search service
	node.queryService = newQueryService(node.queryNodeLoopCtx,
		node.historical,
		node.streaming,
		node.msFactory)

	// start task scheduler
	go node.scheduler.Start()

	// start services
	go node.historical.start()
	go node.watchChangeInfo()
	go node.statsService.start()

	Params.QueryNodeCfg.CreatedTime = time.Now()
	Params.QueryNodeCfg.UpdatedTime = time.Now()

	node.UpdateStateCode(internalpb.StateCode_Healthy)
	log.Debug("query node start successfully",
		zap.Any("queryNodeID", Params.QueryNodeCfg.QueryNodeID),
		zap.Any("IP", Params.QueryNodeCfg.QueryNodeIP),
		zap.Any("Port", Params.QueryNodeCfg.QueryNodePort),
	)
	return nil
}

// Stop mainly stop QueryNode's query service, historical loop and streaming loop.
func (node *QueryNode) Stop() error {
	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	node.queryNodeLoopCancel()

	// close services
	if node.dataSyncService != nil {
		node.dataSyncService.close()
	}
	if node.historical != nil {
		node.historical.close()
	}
	if node.streaming != nil {
		node.streaming.close()
	}
	if node.queryService != nil {
		node.queryService.close()
	}
	if node.statsService != nil {
		node.statsService.close()
	}
	node.session.Revoke(time.Second)
	return nil
}

// UpdateStateCode updata the state of query node, which can be initializing, healthy, and abnormal
func (node *QueryNode) UpdateStateCode(code internalpb.StateCode) {
	node.stateCode.Store(code)
}

// SetEtcdClient assigns parameter client to its member etcdCli
func (node *QueryNode) SetEtcdClient(client *clientv3.Client) {
	node.etcdCli = client
}

// SetRootCoord assigns parameter rc to its member rootCoord.
func (node *QueryNode) SetRootCoord(rc types.RootCoord) error {
	if rc == nil {
		return errors.New("null root coordinator interface")
	}
	node.rootCoord = rc
	return nil
}

// SetIndexCoord assigns parameter index to its member indexCoord.
func (node *QueryNode) SetIndexCoord(index types.IndexCoord) error {
	if index == nil {
		return errors.New("null index coordinator interface")
	}
	node.indexCoord = index
	return nil
}

func (node *QueryNode) watchChangeInfo() {
	log.Debug("query node watchChangeInfo start")
	watchChan := node.etcdKV.WatchWithPrefix(util.ChangeInfoMetaPrefix)
	for {
		select {
		case <-node.queryNodeLoopCtx.Done():
			log.Debug("query node watchChangeInfo close")
			return
		case resp := <-watchChan:
			for _, event := range resp.Events {
				switch event.Type {
				case mvccpb.PUT:
					infoID, err := strconv.ParseInt(filepath.Base(string(event.Kv.Key)), 10, 64)
					if err != nil {
						log.Warn("Parse SealedSegmentsChangeInfo id failed", zap.Any("error", err.Error()))
						continue
					}
					log.Debug("get SealedSegmentsChangeInfo from etcd",
						zap.Any("infoID", infoID),
					)
					info := &querypb.SealedSegmentsChangeInfo{}
					err = proto.Unmarshal(event.Kv.Value, info)
					if err != nil {
						log.Warn("Unmarshal SealedSegmentsChangeInfo failed", zap.Any("error", err.Error()))
						continue
					}
					go func() {
						err = node.removeSegments(info)
						if err != nil {
							log.Warn("cleanup segments failed", zap.Any("error", err.Error()))
						}
					}()
				default:
					// do nothing
				}
			}
		}
	}
}

func (node *QueryNode) waitChangeInfo(segmentChangeInfos *querypb.SealedSegmentsChangeInfo) error {
	fn := func() error {
		for _, info := range segmentChangeInfos.Infos {
			canDoLoadBalance := true
			// make sure all query channel already received segment location changes
			// Check online segments:
			for _, segmentInfo := range info.OnlineSegments {
				if node.queryService.hasQueryCollection(segmentInfo.CollectionID) {
					qc, err := node.queryService.getQueryCollection(segmentInfo.CollectionID)
					if err != nil {
						canDoLoadBalance = false
						break
					}
					if info.OnlineNodeID == Params.QueryNodeCfg.QueryNodeID && !qc.globalSegmentManager.hasGlobalSealedSegment(segmentInfo.SegmentID) {
						canDoLoadBalance = false
						break
					}
				}
			}
			// Check offline segments:
			for _, segmentInfo := range info.OfflineSegments {
				if node.queryService.hasQueryCollection(segmentInfo.CollectionID) {
					qc, err := node.queryService.getQueryCollection(segmentInfo.CollectionID)
					if err != nil {
						canDoLoadBalance = false
						break
					}
					if info.OfflineNodeID == Params.QueryNodeCfg.QueryNodeID && qc.globalSegmentManager.hasGlobalSealedSegment(segmentInfo.SegmentID) {
						canDoLoadBalance = false
						break
					}
				}
			}
			if canDoLoadBalance {
				return nil
			}
			return errors.New(fmt.Sprintln("waitChangeInfo failed, infoID = ", segmentChangeInfos.Base.GetMsgID()))
		}

		return nil
	}

	return retry.Do(node.queryNodeLoopCtx, fn, retry.Attempts(50))
}

// remove the segments since it's already compacted or balanced to other querynodes
func (node *QueryNode) removeSegments(segmentChangeInfos *querypb.SealedSegmentsChangeInfo) error {
	err := node.waitChangeInfo(segmentChangeInfos)
	if err != nil {
		return err
	}

	node.streaming.replica.queryLock()
	node.historical.replica.queryLock()
	defer node.streaming.replica.queryUnlock()
	defer node.historical.replica.queryUnlock()
	for _, info := range segmentChangeInfos.Infos {
		// For online segments:
		for _, segmentInfo := range info.OnlineSegments {
			// delete growing segment because these segments are loaded in historical.
			hasGrowingSegment := node.streaming.replica.hasSegment(segmentInfo.SegmentID)
			if hasGrowingSegment {
				err := node.streaming.replica.removeSegment(segmentInfo.SegmentID)
				if err != nil {
					return err
				}
				log.Debug("remove growing segment in removeSegments",
					zap.Any("collectionID", segmentInfo.CollectionID),
					zap.Any("segmentID", segmentInfo.SegmentID),
					zap.Any("infoID", segmentChangeInfos.Base.GetMsgID()),
				)
			}
		}

		// For offline segments:
		for _, segmentInfo := range info.OfflineSegments {
			// load balance or compaction, remove old sealed segments.
			if info.OfflineNodeID == Params.QueryNodeCfg.QueryNodeID {
				err := node.historical.replica.removeSegment(segmentInfo.SegmentID)
				if err != nil {
					return err
				}
				log.Debug("remove sealed segment", zap.Any("collectionID", segmentInfo.CollectionID),
					zap.Any("segmentID", segmentInfo.SegmentID),
					zap.Any("infoID", segmentChangeInfos.Base.GetMsgID()),
				)
			}
		}
	}
	return nil
}
