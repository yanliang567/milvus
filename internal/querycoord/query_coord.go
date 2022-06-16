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
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/golang/protobuf/proto"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/allocator"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	handoffSegmentPrefix = "querycoord-handoff"
)

// UniqueID is an alias for the Int64 type
type UniqueID = typeutil.UniqueID

// Timestamp is an alias for the Int64 type
type Timestamp = typeutil.Timestamp

// Params is param table of query coordinator
var Params paramtable.ComponentParam

// QueryCoord is the coordinator of queryNodes
type QueryCoord struct {
	loopCtx    context.Context
	loopCancel context.CancelFunc
	loopWg     sync.WaitGroup
	kvClient   *etcdkv.EtcdKV

	initOnce sync.Once

	queryCoordID uint64
	meta         Meta
	cluster      Cluster
	handler      *channelUnsubscribeHandler
	newNodeFn    newQueryNodeFn
	scheduler    *TaskScheduler
	idAllocator  func() (UniqueID, error)
	indexChecker *IndexChecker

	metricsCacheManager *metricsinfo.MetricsCacheManager

	etcdCli          *clientv3.Client
	dataCoordClient  types.DataCoord
	rootCoordClient  types.RootCoord
	indexCoordClient types.IndexCoord
	broker           *globalMetaBroker

	session   *sessionutil.Session
	eventChan <-chan *sessionutil.SessionEvent

	stateCode atomic.Value

	factory       dependency.Factory
	chunkManager  storage.ChunkManager
	groupBalancer balancer
}

// Register register query service at etcd
func (qc *QueryCoord) Register() error {
	qc.session.Register()
	go qc.session.LivenessCheck(qc.loopCtx, func() {
		log.Error("Query Coord disconnected from etcd, process will exit", zap.Int64("Server Id", qc.session.ServerID))
		if err := qc.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
		// manually send signal to starter goroutine
		if qc.session.TriggerKill {
			if p, err := os.FindProcess(os.Getpid()); err == nil {
				p.Signal(syscall.SIGINT)
			}
		}
	})
	return nil
}

func (qc *QueryCoord) initSession() error {
	qc.session = sessionutil.NewSession(qc.loopCtx, Params.EtcdCfg.MetaRootPath, qc.etcdCli)
	if qc.session == nil {
		return fmt.Errorf("session is nil, the etcd client connection may have failed")
	}
	qc.session.Init(typeutil.QueryCoordRole, Params.QueryCoordCfg.Address, true, true)
	Params.QueryCoordCfg.SetNodeID(qc.session.ServerID)
	Params.SetLogger(qc.session.ServerID)
	return nil
}

// Init function initializes the queryCoord's meta, cluster, etcdKV and task scheduler
func (qc *QueryCoord) Init() error {
	log.Info("query coordinator start init, session info", zap.String("metaPath", Params.EtcdCfg.MetaRootPath), zap.String("address", Params.QueryCoordCfg.Address))
	var initError error
	qc.initOnce.Do(func() {
		err := qc.initSession()
		if err != nil {
			log.Error("queryCoord init session failed", zap.Error(err))
			initError = err
			return
		}
		etcdKV := etcdkv.NewEtcdKV(qc.etcdCli, Params.EtcdCfg.MetaRootPath)
		qc.kvClient = etcdKV
		log.Debug("query coordinator try to connect etcd success")

		// init id allocator
		idAllocatorKV := tsoutil.NewTSOKVBase(qc.etcdCli, Params.EtcdCfg.KvRootPath, "queryCoordTaskID")
		idAllocator := allocator.NewGlobalIDAllocator("idTimestamp", idAllocatorKV)
		initError = idAllocator.Initialize()
		if initError != nil {
			log.Error("query coordinator idAllocator initialize failed", zap.Error(initError))
			return
		}
		qc.idAllocator = func() (UniqueID, error) {
			return idAllocator.AllocOne()
		}

		qc.factory.Init(&Params)

		// init meta
		qc.meta, initError = newMeta(qc.loopCtx, qc.kvClient, qc.factory, qc.idAllocator, qc.dataCoordClient)
		if initError != nil {
			log.Error("query coordinator init meta failed", zap.Error(initError))
			return
		}

		// init channelUnsubscribeHandler
		qc.handler, initError = newChannelUnsubscribeHandler(qc.loopCtx, qc.kvClient, qc.factory)
		if initError != nil {
			log.Error("query coordinator init channelUnsubscribeHandler failed", zap.Error(initError))
			return
		}

		// init cluster
		qc.cluster, initError = newQueryNodeCluster(qc.loopCtx, qc.meta, qc.kvClient, qc.newNodeFn, qc.session, qc.handler)
		if initError != nil {
			log.Error("query coordinator init cluster failed", zap.Error(initError))
			return
		}

		qc.groupBalancer = newReplicaBalancer(qc.meta, qc.cluster)

		// NOTE: ignore the returned error
		// we only try best to reload the leader addresses
		reloadShardLeaderAddress(qc.meta, qc.cluster)

		qc.chunkManager, initError = qc.factory.NewVectorStorageChunkManager(qc.loopCtx)

		if initError != nil {
			log.Error("query coordinator init cluster failed", zap.Error(initError))
			return
		}

		//init globalMetaBroker
		qc.broker, initError = newGlobalMetaBroker(qc.loopCtx, qc.rootCoordClient, qc.dataCoordClient, qc.indexCoordClient, qc.chunkManager)
		if initError != nil {
			log.Error("query coordinator init globalMetaBroker failed", zap.Error(initError))
			return
		}

		// init task scheduler
		qc.scheduler, initError = newTaskScheduler(qc.loopCtx, qc.meta, qc.cluster, qc.kvClient, qc.broker, qc.idAllocator)
		if initError != nil {
			log.Error("query coordinator init task scheduler failed", zap.Error(initError))
			return
		}

		// init index checker
		qc.indexChecker, initError = newIndexChecker(qc.loopCtx, qc.kvClient, qc.meta, qc.cluster, qc.scheduler, qc.broker)
		if initError != nil {
			log.Error("query coordinator init index checker failed", zap.Error(initError))
			return
		}

		qc.metricsCacheManager = metricsinfo.NewMetricsCacheManager()
	})
	log.Info("QueryCoord init success")
	return initError
}

// Start function starts the goroutines to watch the meta and node updates
func (qc *QueryCoord) Start() error {
	qc.scheduler.Start()
	log.Info("start scheduler ...")

	qc.indexChecker.start()
	log.Info("start index checker ...")

	qc.handler.start()
	log.Info("start channel unsubscribe loop ...")

	Params.QueryCoordCfg.CreatedTime = time.Now()
	Params.QueryCoordCfg.UpdatedTime = time.Now()

	qc.loopWg.Add(1)
	go qc.watchNodeLoop()

	qc.loopWg.Add(1)
	go qc.watchHandoffSegmentLoop()

	if Params.QueryCoordCfg.AutoBalance {
		qc.loopWg.Add(1)
		go qc.loadBalanceSegmentLoop()
	}

	qc.UpdateStateCode(internalpb.StateCode_Healthy)

	return nil
}

// Stop function stops watching the meta and node updates
func (qc *QueryCoord) Stop() error {
	qc.UpdateStateCode(internalpb.StateCode_Abnormal)

	if qc.scheduler != nil {
		qc.scheduler.Close()
		log.Info("close scheduler ...")
	}

	if qc.indexChecker != nil {
		qc.indexChecker.close()
		log.Info("close index checker ...")
	}

	if qc.handler != nil {
		qc.handler.close()
		log.Info("close channel unsubscribe loop ...")
	}

	if qc.loopCancel != nil {
		qc.loopCancel()
		log.Info("cancel the loop of QueryCoord")
	}

	log.Warn("Query Coord stopped successfully...")
	qc.loopWg.Wait()
	qc.session.Revoke(time.Second)
	return nil
}

// UpdateStateCode updates the status of the coord, including healthy, unhealthy
func (qc *QueryCoord) UpdateStateCode(code internalpb.StateCode) {
	qc.stateCode.Store(code)
}

// NewQueryCoord creates a QueryCoord object.
func NewQueryCoord(ctx context.Context, factory dependency.Factory) (*QueryCoord, error) {
	rand.Seed(time.Now().UnixNano())
	ctx1, cancel := context.WithCancel(ctx)
	service := &QueryCoord{
		loopCtx:    ctx1,
		loopCancel: cancel,
		factory:    factory,
		newNodeFn:  newQueryNode,
	}

	service.UpdateStateCode(internalpb.StateCode_Abnormal)
	return service, nil
}

// SetEtcdClient sets etcd's client
func (qc *QueryCoord) SetEtcdClient(etcdClient *clientv3.Client) {
	qc.etcdCli = etcdClient
}

// SetRootCoord sets root coordinator's client
func (qc *QueryCoord) SetRootCoord(rootCoord types.RootCoord) error {
	if rootCoord == nil {
		return errors.New("null RootCoord interface")
	}

	qc.rootCoordClient = rootCoord
	return nil
}

// SetDataCoord sets data coordinator's client
func (qc *QueryCoord) SetDataCoord(dataCoord types.DataCoord) error {
	if dataCoord == nil {
		return errors.New("null DataCoord interface")
	}

	qc.dataCoordClient = dataCoord
	return nil
}

// SetIndexCoord sets index coordinator's client
func (qc *QueryCoord) SetIndexCoord(indexCoord types.IndexCoord) error {
	if indexCoord == nil {
		return errors.New("null IndexCoord interface")
	}

	qc.indexCoordClient = indexCoord
	return nil
}

func (qc *QueryCoord) watchNodeLoop() {
	ctx, cancel := context.WithCancel(qc.loopCtx)
	defer cancel()
	defer qc.loopWg.Done()
	log.Info("QueryCoord start watch node loop")

	unallocatedNodes := qc.getUnallocatedNodes()
	for _, n := range unallocatedNodes {
		if err := qc.allocateNode(n); err != nil {
			log.Warn("unable to allcoate node", zap.Int64("nodeID", n), zap.Error(err))
		}
	}

	offlineNodeIDs := qc.cluster.OfflineNodeIDs()
	if len(offlineNodeIDs) != 0 {
		loadBalanceSegment := &querypb.LoadBalanceRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_LoadBalanceSegments,
				SourceID: qc.session.ServerID,
			},
			BalanceReason: querypb.TriggerCondition_NodeDown,
			SourceNodeIDs: offlineNodeIDs,
		}

		baseTask := newBaseTask(qc.loopCtx, querypb.TriggerCondition_NodeDown)
		loadBalanceTask := &loadBalanceTask{
			baseTask:           baseTask,
			LoadBalanceRequest: loadBalanceSegment,
			broker:             qc.broker,
			cluster:            qc.cluster,
			meta:               qc.meta,
		}
		//TODO::deal enqueue error
		qc.scheduler.Enqueue(loadBalanceTask)
		log.Info("start a loadBalance task", zap.Any("task", loadBalanceTask))
	}

	// TODO silverxia add Rewatch logic
	qc.eventChan = qc.session.WatchServices(typeutil.QueryNodeRole, qc.cluster.GetSessionVersion()+1, nil)
	qc.handleNodeEvent(ctx)
}

func (qc *QueryCoord) allocateNode(nodeID int64) error {
	plans, err := qc.groupBalancer.addNode(nodeID)
	if err != nil {
		return err
	}
	for _, p := range plans {
		if err := qc.meta.applyReplicaBalancePlan(p); err != nil {
			log.Warn("failed to apply balance plan", zap.Error(err), zap.Any("plan", p))
		}
	}
	return nil
}

func (qc *QueryCoord) getUnallocatedNodes() []int64 {
	onlines := qc.cluster.OnlineNodeIDs()
	var ret []int64
	for _, n := range onlines {
		replica, err := qc.meta.getReplicasByNodeID(n)
		if err != nil {
			log.Warn("failed to get replica", zap.Int64("nodeID", n), zap.Error(err))
			continue
		}
		if replica == nil {
			ret = append(ret, n)
		}
	}
	return ret
}

func (qc *QueryCoord) handleNodeEvent(ctx context.Context) {
	offlineNodeCh := make(chan UniqueID, 100)
	go qc.loadBalanceNodeLoop(ctx, offlineNodeCh)

	for {
		select {
		case <-ctx.Done():
			return

		case event, ok := <-qc.eventChan:
			if !ok {
				// ErrCompacted is handled inside SessionWatcher
				log.Error("Session Watcher channel closed", zap.Int64("server id", qc.session.ServerID))
				go qc.Stop()
				if qc.session.TriggerKill {
					if p, err := os.FindProcess(os.Getpid()); err == nil {
						p.Signal(syscall.SIGINT)
					}
				}
				return
			}

			switch event.EventType {
			case sessionutil.SessionAddEvent:
				serverID := event.Session.ServerID
				log.Info("start add a QueryNode to cluster", zap.Any("nodeID", serverID))
				err := qc.cluster.RegisterNode(ctx, event.Session, serverID, disConnect)
				if err != nil {
					log.Error("QueryCoord failed to register a QueryNode", zap.Int64("nodeID", serverID), zap.String("error info", err.Error()))
					continue
				}
				go func(serverID int64) {
					if err := qc.allocateNode(serverID); err != nil {
						log.Error("unable to allcoate node", zap.Int64("nodeID", serverID), zap.Error(err))
					}
				}(serverID)
				qc.metricsCacheManager.InvalidateSystemInfoMetrics()

			case sessionutil.SessionDelEvent:
				serverID := event.Session.ServerID
				log.Info("get a del event after QueryNode down", zap.Int64("nodeID", serverID))
				nodeExist := qc.cluster.HasNode(serverID)
				if !nodeExist {
					log.Error("QueryNode not exist", zap.Int64("nodeID", serverID))
					continue
				}

				qc.cluster.StopNode(serverID)
				offlineNodeCh <- serverID
			}
		}
	}
}

func (qc *QueryCoord) loadBalanceNodeLoop(ctx context.Context, offlineNodeCh chan UniqueID) {
	for {
		select {
		case <-ctx.Done():
			return

		case node := <-offlineNodeCh:
			loadBalanceSegment := &querypb.LoadBalanceRequest{
				Base: &commonpb.MsgBase{
					MsgType:  commonpb.MsgType_LoadBalanceSegments,
					SourceID: qc.session.ServerID,
				},
				SourceNodeIDs: []int64{node},
				BalanceReason: querypb.TriggerCondition_NodeDown,
			}

			baseTask := newBaseTaskWithRetry(qc.loopCtx, querypb.TriggerCondition_NodeDown, 0)
			loadBalanceTask := &loadBalanceTask{
				baseTask:           baseTask,
				LoadBalanceRequest: loadBalanceSegment,
				broker:             qc.broker,
				cluster:            qc.cluster,
				meta:               qc.meta,
			}
			qc.metricsCacheManager.InvalidateSystemInfoMetrics()
			//TODO:: deal enqueue error
			err := qc.scheduler.Enqueue(loadBalanceTask)
			if err != nil {
				log.Warn("failed to enqueue LoadBalance task into the scheduler",
					zap.Int64("nodeID", node),
					zap.Error(err))
				offlineNodeCh <- node
				continue
			}

			log.Info("start a loadBalance task",
				zap.Int64("nodeID", node),
				zap.Int64("taskID", loadBalanceTask.getTaskID()))

			err = loadBalanceTask.waitToFinish()
			if err != nil {
				log.Warn("failed to process LoadBalance task",
					zap.Int64("nodeID", node),
					zap.Error(err))
				offlineNodeCh <- node
				continue
			}

			log.Info("LoadBalance task done, offline node is removed",
				zap.Int64("nodeID", node))
		}
	}
}

func (qc *QueryCoord) watchHandoffSegmentLoop() {
	ctx, cancel := context.WithCancel(qc.loopCtx)

	defer cancel()
	defer qc.loopWg.Done()
	log.Info("QueryCoord start watch segment loop")

	watchChan := qc.kvClient.WatchWithRevision(handoffSegmentPrefix, qc.indexChecker.revision+1)

	for {
		select {
		case <-ctx.Done():
			return
		case resp := <-watchChan:
			for _, event := range resp.Events {
				segmentInfo := &querypb.SegmentInfo{}
				err := proto.Unmarshal(event.Kv.Value, segmentInfo)
				if err != nil {
					log.Error("watchHandoffSegmentLoop: unmarshal failed", zap.Any("error", err.Error()))
					continue
				}
				switch event.Type {
				case mvccpb.PUT:
					validHandoffReq, _ := qc.indexChecker.verifyHandoffReqValid(segmentInfo)
					if Params.QueryCoordCfg.AutoHandoff && validHandoffReq {
						qc.indexChecker.enqueueHandoffReq(segmentInfo)
						log.Info("watchHandoffSegmentLoop: enqueue a handoff request to index checker", zap.Any("segment info", segmentInfo))
					} else {
						log.Info("watchHandoffSegmentLoop: collection/partition has not been loaded or autoHandoff equal to false, remove req from etcd", zap.Any("segmentInfo", segmentInfo))
						buildQuerySegmentPath := fmt.Sprintf("%s/%d/%d/%d", handoffSegmentPrefix, segmentInfo.CollectionID, segmentInfo.PartitionID, segmentInfo.SegmentID)
						err = qc.kvClient.Remove(buildQuerySegmentPath)
						if err != nil {
							log.Error("watchHandoffSegmentLoop: remove handoff segment from etcd failed", zap.Error(err))
							panic(err)
						}
					}
				default:
					// do nothing
				}
			}
		}
	}
}

func (qc *QueryCoord) loadBalanceSegmentLoop() {
	ctx, cancel := context.WithCancel(qc.loopCtx)
	defer cancel()
	defer qc.loopWg.Done()
	log.Info("QueryCoord start load balance segment loop")

	timer := time.NewTicker(time.Duration(Params.QueryCoordCfg.BalanceIntervalSeconds) * time.Second)

	var collectionInfos []*querypb.CollectionInfo
	pos := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			if pos == len(collectionInfos) {
				pos = 0
				collectionInfos = qc.meta.showCollections()
			}
			// get mem info of online nodes from cluster
			nodeID2MemUsageRate := make(map[int64]float64)
			nodeID2MemUsage := make(map[int64]uint64)
			nodeID2TotalMem := make(map[int64]uint64)
			loadBalanceTasks := make([]*loadBalanceTask, 0)
			// balance at most 20 collections in a round
			for i := 0; pos < len(collectionInfos) && i < 20; i, pos = i+1, pos+1 {
				info := collectionInfos[pos]
				replicas, err := qc.meta.getReplicasByCollectionID(info.GetCollectionID())
				if err != nil {
					log.Warn("unable to get replicas of collection", zap.Int64("collectionID", info.GetCollectionID()))
					continue
				}
				for _, replica := range replicas {
					// auto balance is executed on replica level
					onlineNodeIDs := replica.GetNodeIds()
					if len(onlineNodeIDs) == 0 {
						log.Error("loadBalanceSegmentLoop: there are no online QueryNode to balance", zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID))
						continue
					}
					var availableNodeIDs []int64
					nodeID2SegmentInfos := make(map[int64]map[UniqueID]*querypb.SegmentInfo)
					for _, nodeID := range onlineNodeIDs {
						if _, ok := nodeID2MemUsage[nodeID]; !ok {
							nodeInfo, err := qc.cluster.GetNodeInfoByID(nodeID)
							if err != nil {
								log.Warn("loadBalanceSegmentLoop: get node info from QueryNode failed",
									zap.Int64("nodeID", nodeID), zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID),
									zap.Error(err))
								continue
							}
							nodeID2MemUsageRate[nodeID] = nodeInfo.(*queryNode).memUsageRate
							nodeID2MemUsage[nodeID] = nodeInfo.(*queryNode).memUsage
							nodeID2TotalMem[nodeID] = nodeInfo.(*queryNode).totalMem
						}

						updateSegmentInfoDone := true
						leastSegmentInfos := make(map[UniqueID]*querypb.SegmentInfo)
						segmentInfos := qc.meta.getSegmentInfosByNodeAndCollection(nodeID, replica.GetCollectionID())
						for _, segmentInfo := range segmentInfos {
							leastInfo, err := qc.cluster.GetSegmentInfoByID(ctx, segmentInfo.SegmentID)
							if err != nil {
								log.Warn("loadBalanceSegmentLoop: get segment info from QueryNode failed", zap.Int64("nodeID", nodeID),
									zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID),
									zap.Error(err))
								updateSegmentInfoDone = false
								break
							}
							leastSegmentInfos[segmentInfo.SegmentID] = leastInfo
						}
						if updateSegmentInfoDone {
							availableNodeIDs = append(availableNodeIDs, nodeID)
							nodeID2SegmentInfos[nodeID] = leastSegmentInfos
						}
					}
					log.Info("loadBalanceSegmentLoop: memory usage rate of all online QueryNode", zap.Int64("collection", replica.CollectionID),
						zap.Int64("replica", replica.ReplicaID), zap.Any("mem rate", nodeID2MemUsageRate))
					if len(availableNodeIDs) <= 1 {
						log.Info("loadBalanceSegmentLoop: there are too few available query nodes to balance",
							zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID),
							zap.Int64s("onlineNodeIDs", onlineNodeIDs), zap.Int64s("availableNodeIDs", availableNodeIDs))
						continue
					}

					// check which nodes need balance and determine which segments on these nodes need to be migrated to other nodes
					memoryInsufficient := false
					for {
						sort.Slice(availableNodeIDs, func(i, j int) bool {
							return nodeID2MemUsageRate[availableNodeIDs[i]] > nodeID2MemUsageRate[availableNodeIDs[j]]
						})

						// the memoryUsageRate of the sourceNode is higher than other query node
						sourceNodeID := availableNodeIDs[0]
						dstNodeID := availableNodeIDs[len(availableNodeIDs)-1]
						memUsageRateDiff := nodeID2MemUsageRate[sourceNodeID] - nodeID2MemUsageRate[dstNodeID]
						if nodeID2MemUsageRate[sourceNodeID] <= Params.QueryCoordCfg.OverloadedMemoryThresholdPercentage &&
							memUsageRateDiff <= Params.QueryCoordCfg.MemoryUsageMaxDifferencePercentage {
							break
						}
						// if memoryUsageRate of source node is greater than 90%, and the max memUsageDiff is greater than 30%
						// then migrate the segments on source node to other query nodes
						segmentInfos := nodeID2SegmentInfos[sourceNodeID]
						// select the segment that needs balance on the source node
						selectedSegmentInfo, err := chooseSegmentToBalance(sourceNodeID, dstNodeID, segmentInfos, nodeID2MemUsage, nodeID2TotalMem, nodeID2MemUsageRate)
						if err != nil {
							// no enough memory on query nodes to balance, then notify proxy to stop insert
							memoryInsufficient = true
							break
						}
						if selectedSegmentInfo == nil {
							break
						}
						// select a segment to balance successfully, then recursive traversal whether there are other segments that can balance
						req := &querypb.LoadBalanceRequest{
							Base: &commonpb.MsgBase{
								MsgType: commonpb.MsgType_LoadBalanceSegments,
							},
							BalanceReason:    querypb.TriggerCondition_LoadBalance,
							SourceNodeIDs:    []UniqueID{sourceNodeID},
							DstNodeIDs:       []UniqueID{dstNodeID},
							SealedSegmentIDs: []UniqueID{selectedSegmentInfo.SegmentID},
						}
						baseTask := newBaseTask(qc.loopCtx, querypb.TriggerCondition_LoadBalance)
						balanceTask := &loadBalanceTask{
							baseTask:           baseTask,
							LoadBalanceRequest: req,
							broker:             qc.broker,
							cluster:            qc.cluster,
							meta:               qc.meta,
						}
						log.Info("loadBalanceSegmentLoop: generate a loadBalance task",
							zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID),
							zap.Any("task", balanceTask))
						loadBalanceTasks = append(loadBalanceTasks, balanceTask)
						nodeID2MemUsage[sourceNodeID] -= uint64(selectedSegmentInfo.MemSize)
						nodeID2MemUsage[dstNodeID] += uint64(selectedSegmentInfo.MemSize)
						nodeID2MemUsageRate[sourceNodeID] = float64(nodeID2MemUsage[sourceNodeID]) / float64(nodeID2TotalMem[sourceNodeID])
						nodeID2MemUsageRate[dstNodeID] = float64(nodeID2MemUsage[dstNodeID]) / float64(nodeID2TotalMem[dstNodeID])
						delete(nodeID2SegmentInfos[sourceNodeID], selectedSegmentInfo.SegmentID)
						nodeID2SegmentInfos[dstNodeID][selectedSegmentInfo.SegmentID] = selectedSegmentInfo
						continue
					}
					if memoryInsufficient {
						// no enough memory on query nodes to balance, then notify proxy to stop insert
						//TODO:: xige-16
						log.Warn("loadBalanceSegmentLoop: QueryNode has insufficient memory, stop inserting data", zap.Int64("collection", replica.CollectionID), zap.Int64("replica", replica.ReplicaID))
					}
				}
			}
			for _, t := range loadBalanceTasks {
				qc.scheduler.Enqueue(t)
				err := t.waitToFinish()
				if err != nil {
					// if failed, wait for next balance loop
					// it may be that the collection/partition of the balanced segment has been released
					// it also may be other abnormal errors
					log.Error("loadBalanceSegmentLoop: balance task execute failed", zap.Any("task", t))
				} else {
					log.Info("loadBalanceSegmentLoop: balance task execute success", zap.Any("task", t))
				}
			}
		}
	}
}

func chooseSegmentToBalance(sourceNodeID int64, dstNodeID int64,
	segmentInfos map[UniqueID]*querypb.SegmentInfo,
	nodeID2MemUsage map[int64]uint64,
	nodeID2TotalMem map[int64]uint64,
	nodeID2MemUsageRate map[int64]float64) (*querypb.SegmentInfo, error) {
	memoryInsufficient := true
	minMemDiffPercentage := 1.0
	var selectedSegmentInfo *querypb.SegmentInfo
	for _, info := range segmentInfos {
		dstNodeMemUsageAfterBalance := nodeID2MemUsage[dstNodeID] + uint64(info.MemSize)
		dstNodeMemUsageRateAfterBalance := float64(dstNodeMemUsageAfterBalance) / float64(nodeID2TotalMem[dstNodeID])
		// if memUsageRate of dstNode is greater than OverloadedMemoryThresholdPercentage after balance, than can't balance
		if dstNodeMemUsageRateAfterBalance < Params.QueryCoordCfg.OverloadedMemoryThresholdPercentage {
			memoryInsufficient = false
			sourceNodeMemUsageAfterBalance := nodeID2MemUsage[sourceNodeID] - uint64(info.MemSize)
			sourceNodeMemUsageRateAfterBalance := float64(sourceNodeMemUsageAfterBalance) / float64(nodeID2TotalMem[sourceNodeID])
			// assume all query node has same memory capacity
			// if the memUsageRateDiff between the two nodes does not become smaller after balance, there is no need for balance
			diffBeforBalance := nodeID2MemUsageRate[sourceNodeID] - nodeID2MemUsageRate[dstNodeID]
			diffAfterBalance := dstNodeMemUsageRateAfterBalance - sourceNodeMemUsageRateAfterBalance
			if diffAfterBalance < diffBeforBalance {
				if math.Abs(diffAfterBalance) < minMemDiffPercentage {
					selectedSegmentInfo = info
				}
			}
		}
	}

	if memoryInsufficient {
		return nil, errors.New("all QueryNode has insufficient memory")
	}

	return selectedSegmentInfo, nil
}
