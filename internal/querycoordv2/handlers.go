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

package querycoordv2

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/job"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/internal/util/hardware"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/milvus-io/milvus/internal/util/uniquegenerator"
)

// checkAnyReplicaAvailable checks if the collection has enough distinct available shards. These shards
// may come from different replica group. We only need these shards to form a replica that serves query
// requests.
func (s *Server) checkAnyReplicaAvailable(collectionID int64) bool {
	for _, replica := range s.meta.ReplicaManager.GetByCollection(collectionID) {
		isAvailable := true
		for node := range replica.Nodes {
			if s.nodeMgr.Get(node) == nil {
				isAvailable = false
				break
			}
		}
		if isAvailable {
			return true
		}
	}
	return false
}

func (s *Server) getCollectionSegmentInfo(collection int64) []*querypb.SegmentInfo {
	segments := s.dist.SegmentDistManager.GetByCollection(collection)
	infos := make(map[int64]*querypb.SegmentInfo)
	for _, segment := range segments {
		info, ok := infos[segment.GetID()]
		if !ok {
			info = &querypb.SegmentInfo{}
			infos[segment.GetID()] = info
		}
		utils.MergeMetaSegmentIntoSegmentInfo(info, segment)
	}

	return lo.Values(infos)
}

// parseBalanceRequest parses the load balance request,
// returns the collection, replica, and segments
func (s *Server) balanceSegments(ctx context.Context, req *querypb.LoadBalanceRequest, replica *meta.Replica) error {
	srcNode := req.GetSourceNodeIDs()[0]
	dstNodeSet := typeutil.NewUniqueSet(req.GetDstNodeIDs()...)
	if dstNodeSet.Len() == 0 {
		dstNodeSet.Insert(replica.GetNodes()...)
	}
	dstNodeSet.Remove(srcNode)

	toBalance := typeutil.NewSet[*meta.Segment]()
	segments := s.dist.SegmentDistManager.GetByNode(srcNode)
	allSegments := make(map[int64]*meta.Segment)
	for _, segment := range segments {
		allSegments[segment.GetID()] = segment
	}

	if len(req.GetSealedSegmentIDs()) == 0 {
		toBalance.Insert(segments...)
	} else {
		for _, segmentID := range req.GetSealedSegmentIDs() {
			segment, ok := allSegments[segmentID]
			if !ok {
				return fmt.Errorf("segment %d not found in source node %d", segmentID, srcNode)
			}
			toBalance.Insert(segment)
		}
	}

	log := log.With(
		zap.Int64("collectionID", req.GetCollectionID()),
		zap.Int64("srcNodeID", srcNode),
		zap.Int64s("destNodeIDs", dstNodeSet.Collect()),
	)
	plans := s.balancer.AssignSegment(toBalance.Collect(), dstNodeSet.Collect())
	tasks := make([]task.Task, 0, len(plans))
	for _, plan := range plans {
		log.Info("manually balance segment...",
			zap.Int64("destNodeID", plan.To),
			zap.Int64("segmentID", plan.Segment.GetID()),
		)
		task, err := task.NewSegmentTask(ctx,
			Params.QueryCoordCfg.SegmentTaskTimeout,
			req.GetBase().GetMsgID(),
			req.GetCollectionID(),
			replica.GetID(),
			task.NewSegmentAction(plan.To, task.ActionTypeGrow, plan.Segment.GetInsertChannel(), plan.Segment.GetID()),
			task.NewSegmentAction(srcNode, task.ActionTypeReduce, plan.Segment.GetInsertChannel(), plan.Segment.GetID()),
		)

		if err != nil {
			log.Warn("Create segment task for balance failed",
				zap.Int64("collection", req.GetCollectionID()),
				zap.Int64("replica", replica.GetID()),
				zap.String("channel", plan.Segment.InsertChannel),
				zap.Int64("From", srcNode),
				zap.Int64("To", plan.To),
				zap.Error(err),
			)
			continue
		}
		err = s.taskScheduler.Add(task)
		if err != nil {
			task.Cancel()
			return err
		}
		tasks = append(tasks, task)
	}
	return task.Wait(ctx, Params.QueryCoordCfg.SegmentTaskTimeout, tasks...)
}

// TODO(dragondriver): add more detail metrics
func (s *Server) getSystemInfoMetrics(
	ctx context.Context,
	req *milvuspb.GetMetricsRequest) (string, error) {

	clusterTopology := metricsinfo.QueryClusterTopology{
		Self: metricsinfo.QueryCoordInfos{
			BaseComponentInfos: metricsinfo.BaseComponentInfos{
				Name: metricsinfo.ConstructComponentName(typeutil.QueryCoordRole, Params.QueryCoordCfg.GetNodeID()),
				HardwareInfos: metricsinfo.HardwareMetrics{
					IP:           s.session.Address,
					CPUCoreCount: hardware.GetCPUNum(),
					CPUCoreUsage: hardware.GetCPUUsage(),
					Memory:       hardware.GetMemoryCount(),
					MemoryUsage:  hardware.GetUsedMemoryCount(),
					Disk:         hardware.GetDiskCount(),
					DiskUsage:    hardware.GetDiskUsage(),
				},
				SystemInfo:  metricsinfo.DeployMetrics{},
				CreatedTime: Params.QueryCoordCfg.CreatedTime.String(),
				UpdatedTime: Params.QueryCoordCfg.UpdatedTime.String(),
				Type:        typeutil.QueryCoordRole,
				ID:          s.session.ServerID,
			},
			SystemConfigurations: metricsinfo.QueryCoordConfiguration{
				SearchChannelPrefix:       Params.CommonCfg.QueryCoordSearch,
				SearchResultChannelPrefix: Params.CommonCfg.QueryCoordSearchResult,
			},
		},
		ConnectedNodes: make([]metricsinfo.QueryNodeInfos, 0),
	}
	metricsinfo.FillDeployMetricsWithEnv(&clusterTopology.Self.SystemInfo)
	nodesMetrics := s.tryGetNodesMetrics(ctx, req, s.nodeMgr.GetAll()...)
	s.fillMetricsWithNodes(&clusterTopology, nodesMetrics)

	coordTopology := metricsinfo.QueryCoordTopology{
		Cluster: clusterTopology,
		Connections: metricsinfo.ConnTopology{
			Name: metricsinfo.ConstructComponentName(typeutil.QueryCoordRole, Params.QueryCoordCfg.GetNodeID()),
			// TODO(dragondriver): fill ConnectedComponents if necessary
			ConnectedComponents: []metricsinfo.ConnectionInfo{},
		},
	}

	resp, err := metricsinfo.MarshalTopology(coordTopology)
	if err != nil {
		return "", err
	}

	return resp, nil
}

func (s *Server) fillMetricsWithNodes(topo *metricsinfo.QueryClusterTopology, nodeMetrics []*metricResp) {
	for _, metric := range nodeMetrics {
		if metric.err != nil {
			log.Warn("invalid metrics of query node was found",
				zap.Error(metric.err))
			topo.ConnectedNodes = append(topo.ConnectedNodes, metricsinfo.QueryNodeInfos{
				BaseComponentInfos: metricsinfo.BaseComponentInfos{
					HasError:    true,
					ErrorReason: metric.err.Error(),
					// Name doesn't matter here because we can't get it when error occurs, using address as the Name?
					Name: "",
					ID:   int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
				},
			})
			continue
		}

		if metric.resp.Status.ErrorCode != commonpb.ErrorCode_Success {
			log.Warn("invalid metrics of query node was found",
				zap.Any("error_code", metric.resp.Status.ErrorCode),
				zap.Any("error_reason", metric.resp.Status.Reason))
			topo.ConnectedNodes = append(topo.ConnectedNodes, metricsinfo.QueryNodeInfos{
				BaseComponentInfos: metricsinfo.BaseComponentInfos{
					HasError:    true,
					ErrorReason: metric.resp.Status.Reason,
					Name:        metric.resp.ComponentName,
					ID:          int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
				},
			})
			continue
		}

		infos := metricsinfo.QueryNodeInfos{}
		err := metricsinfo.UnmarshalComponentInfos(metric.resp.Response, &infos)
		if err != nil {
			log.Warn("invalid metrics of query node was found",
				zap.Error(err))
			topo.ConnectedNodes = append(topo.ConnectedNodes, metricsinfo.QueryNodeInfos{
				BaseComponentInfos: metricsinfo.BaseComponentInfos{
					HasError:    true,
					ErrorReason: err.Error(),
					Name:        metric.resp.ComponentName,
					ID:          int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
				},
			})
			continue
		}
		topo.ConnectedNodes = append(topo.ConnectedNodes, infos)
	}
}

type metricResp struct {
	resp *milvuspb.GetMetricsResponse
	err  error
}

func (s *Server) tryGetNodesMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest, nodes ...*session.NodeInfo) []*metricResp {
	wg := sync.WaitGroup{}
	ret := make([]*metricResp, 0, len(nodes))
	retCh := make(chan *metricResp, len(nodes))
	for _, node := range nodes {
		node := node
		wg.Add(1)
		go func() {
			defer wg.Done()

			resp, err := s.cluster.GetMetrics(ctx, node.ID(), req)
			if err != nil {
				log.Warn("failed to get metric from QueryNode",
					zap.Int64("nodeID", node.ID()))
				return
			}
			retCh <- &metricResp{
				resp: resp,
				err:  err,
			}
		}()
	}
	wg.Wait()
	close(retCh)
	for resp := range retCh {
		ret = append(ret, resp)
	}
	return ret
}

func (s *Server) fillReplicaInfo(replica *meta.Replica, withShardNodes bool) (*milvuspb.ReplicaInfo, error) {
	info := utils.Replica2ReplicaInfo(replica.Replica)

	channels := s.targetMgr.GetDmChannelsByCollection(replica.GetCollectionID())
	if len(channels) == 0 {
		msg := "failed to get channels, collection not loaded"
		log.Warn(msg)
		return nil, utils.WrapError(msg, meta.ErrCollectionNotFound)
	}
	var segments []*meta.Segment
	if withShardNodes {
		segments = s.dist.SegmentDistManager.GetByCollection(replica.GetCollectionID())
	}

	for _, channel := range channels {
		leader, ok := s.dist.ChannelDistManager.GetShardLeader(replica, channel.GetChannelName())
		var leaderInfo *session.NodeInfo
		if ok {
			leaderInfo = s.nodeMgr.Get(leader)
		}
		if leaderInfo == nil {
			msg := fmt.Sprintf("failed to get shard leader for shard %s, the collection not loaded or leader is offline", channel)
			log.Warn(msg)
			return nil, utils.WrapError(msg, session.WrapErrNodeNotFound(leader))
		}

		shard := &milvuspb.ShardReplica{
			LeaderID:      leader,
			LeaderAddr:    leaderInfo.Addr(),
			DmChannelName: channel.GetChannelName(),
			NodeIds:       []int64{leader},
		}
		if withShardNodes {
			shardNodes := lo.FilterMap(segments, func(segment *meta.Segment, _ int) (int64, bool) {
				if replica.Nodes.Contain(segment.Node) {
					return segment.Node, true
				}
				return 0, false
			})
			shard.NodeIds = append(shard.NodeIds, shardNodes...)
		}
		info.ShardReplicas = append(info.ShardReplicas, shard)
	}
	return info, nil
}

func errCode(err error) commonpb.ErrorCode {
	if errors.Is(err, job.ErrLoadParameterMismatched) {
		return commonpb.ErrorCode_IllegalArgument
	}
	return commonpb.ErrorCode_UnexpectedError
}
