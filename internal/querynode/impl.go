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

package querynode

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/mqclient"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// GetComponentStates return information about whether the node is healthy
func (node *QueryNode) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	stats := &internalpb.ComponentStates{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
	}
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		stats.Status = &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return stats, err
	}
	info := &internalpb.ComponentInfo{
		NodeID:    Params.QueryNodeID,
		Role:      typeutil.QueryNodeRole,
		StateCode: code,
	}
	stats.State = info
	return stats, nil
}

// GetTimeTickChannel returns the time tick channel
// TimeTickChannel contains many time tick messages, which will be sent by query nodes
func (node *QueryNode) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: Params.QueryTimeTickChannelName,
	}, nil
}

// GetStatisticsChannel return the statistics channel
// Statistics channel contains statistics infos of query nodes, such as segment infos, memory infos
func (node *QueryNode) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: Params.StatsChannelName,
	}, nil
}

// AddQueryChannel watch queryChannel of the collection to receive query message
func (node *QueryNode) AddQueryChannel(ctx context.Context, in *queryPb.AddQueryChannelRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	collectionID := in.CollectionID
	if node.queryService == nil {
		errMsg := "null query service, collectionID = " + fmt.Sprintln(collectionID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    errMsg,
		}
		return status, errors.New(errMsg)
	}

	if node.queryService.hasQueryCollection(collectionID) {
		log.Debug("queryCollection has been existed when addQueryChannel",
			zap.Any("collectionID", collectionID),
		)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
		return status, nil
	}

	// add search collection
	err := node.queryService.addQueryCollection(collectionID)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	log.Debug("add query collection", zap.Any("collectionID", collectionID))

	// add request channel
	sc, err := node.queryService.getQueryCollection(in.CollectionID)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	consumeChannels := []string{in.RequestChannelID}
	consumeSubName := Params.MsgChannelSubName + "-" + strconv.FormatInt(collectionID, 10) + "-" + strconv.Itoa(rand.Int())

	if Params.skipQueryChannelRecovery {
		log.Debug("Skip query channel seek back ", zap.Strings("channels", consumeChannels),
			zap.String("seek position", string(in.SeekPosition.MsgID)),
			zap.Uint64("ts", in.SeekPosition.Timestamp))
		sc.queryMsgStream.AsConsumerWithPosition(consumeChannels, consumeSubName, mqclient.SubscriptionPositionLatest)
	} else {
		sc.queryMsgStream.AsConsumer(consumeChannels, consumeSubName)
		if in.SeekPosition == nil || len(in.SeekPosition.MsgID) == 0 {
			// as consumer
			log.Debug("querynode AsConsumer: " + strings.Join(consumeChannels, ", ") + " : " + consumeSubName)
		} else {
			// seek query channel
			err = sc.queryMsgStream.Seek([]*internalpb.MsgPosition{in.SeekPosition})
			if err != nil {
				status := &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_UnexpectedError,
					Reason:    err.Error(),
				}
				return status, err
			}
			log.Debug("querynode seek query channel: ", zap.Any("consumeChannels", consumeChannels))
		}
	}

	// add result channel
	producerChannels := []string{in.ResultChannelID}
	sc.queryResultMsgStream.AsProducer(producerChannels)
	log.Debug("querynode AsProducer: " + strings.Join(producerChannels, ", "))

	// init global sealed segments
	for _, segment := range in.GlobalSealedSegments {
		err = sc.globalSegmentManager.addGlobalSegmentInfo(segment)
		if err != nil {
			status := &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			}
			return status, err
		}
	}

	// start queryCollection, message stream need to asConsumer before start
	sc.start()
	log.Debug("start query collection", zap.Any("collectionID", collectionID))

	status := &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	return status, nil
}

// RemoveQueryChannel remove queryChannel of the collection to stop receiving query message
func (node *QueryNode) RemoveQueryChannel(ctx context.Context, in *queryPb.RemoveQueryChannelRequest) (*commonpb.Status, error) {
	// if node.searchService == nil || node.searchService.searchMsgStream == nil {
	// 	errMsg := "null search service or null search result message stream"
	// 	status := &commonpb.Status{
	// 		ErrorCode: commonpb.ErrorCode_UnexpectedError,
	// 		Reason:    errMsg,
	// 	}

	// 	return status, errors.New(errMsg)
	// }

	// searchStream, ok := node.searchService.searchMsgStream.(*pulsarms.PulsarMsgStream)
	// if !ok {
	// 	errMsg := "type assertion failed for search message stream"
	// 	status := &commonpb.Status{
	// 		ErrorCode: commonpb.ErrorCode_UnexpectedError,
	// 		Reason:    errMsg,
	// 	}

	// 	return status, errors.New(errMsg)
	// }

	// resultStream, ok := node.searchService.searchResultMsgStream.(*pulsarms.PulsarMsgStream)
	// if !ok {
	// 	errMsg := "type assertion failed for search result message stream"
	// 	status := &commonpb.Status{
	// 		ErrorCode: commonpb.ErrorCode_UnexpectedError,
	// 		Reason:    errMsg,
	// 	}

	// 	return status, errors.New(errMsg)
	// }

	// // remove request channel
	// consumeChannels := []string{in.RequestChannelID}
	// consumeSubName := Params.MsgChannelSubName
	// // TODO: searchStream.RemovePulsarConsumers(producerChannels)
	// searchStream.AsConsumer(consumeChannels, consumeSubName)

	// // remove result channel
	// producerChannels := []string{in.ResultChannelID}
	// // TODO: resultStream.RemovePulsarProducer(producerChannels)
	// resultStream.AsProducer(producerChannels)

	status := &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	return status, nil
}

// WatchDmChannels create consumers on dmChannels to reveive Incremental data，which is the important part of real-time query
func (node *QueryNode) WatchDmChannels(ctx context.Context, in *queryPb.WatchDmChannelsRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	dct := &watchDmChannelsTask{
		baseTask: baseTask{
			ctx:  ctx,
			done: make(chan error),
		},
		req:  in,
		node: node,
	}

	err := node.scheduler.queue.Enqueue(dct)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		log.Warn(err.Error())
		return status, err
	}
	log.Debug("watchDmChannelsTask Enqueue done", zap.Any("collectionID", in.CollectionID))

	waitFunc := func() (*commonpb.Status, error) {
		err = dct.WaitToFinish()
		if err != nil {
			status := &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			}
			log.Warn(err.Error())
			return status, err
		}
		log.Debug("watchDmChannelsTask WaitToFinish done", zap.Any("collectionID", in.CollectionID))
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	return waitFunc()
}

// WatchDeltaChannels create consumers on dmChannels to reveive Incremental data，which is the important part of real-time query
func (node *QueryNode) WatchDeltaChannels(ctx context.Context, in *queryPb.WatchDeltaChannelsRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	dct := &watchDeltaChannelsTask{
		baseTask: baseTask{
			ctx:  ctx,
			done: make(chan error),
		},
		req:  in,
		node: node,
	}

	err := node.scheduler.queue.Enqueue(dct)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		log.Warn(err.Error())
		return status, err
	}
	log.Debug("watchDeltaChannelsTask Enqueue done", zap.Any("collectionID", in.CollectionID))

	waitFunc := func() (*commonpb.Status, error) {
		err = dct.WaitToFinish()
		if err != nil {
			status := &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			}
			log.Warn(err.Error())
			return status, err
		}
		log.Debug("watchDeltaChannelsTask WaitToFinish done", zap.Any("collectionID", in.CollectionID))
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	return waitFunc()
}

// LoadSegments load historical data into query node, historical data can be vector data or index
func (node *QueryNode) LoadSegments(ctx context.Context, in *queryPb.LoadSegmentsRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	dct := &loadSegmentsTask{
		baseTask: baseTask{
			ctx:  ctx,
			done: make(chan error),
		},
		req:  in,
		node: node,
	}

	err := node.scheduler.queue.Enqueue(dct)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		log.Warn(err.Error())
		return status, err
	}
	segmentIDs := make([]UniqueID, 0)
	for _, info := range in.Infos {
		segmentIDs = append(segmentIDs, info.SegmentID)
	}
	log.Debug("loadSegmentsTask Enqueue done", zap.Int64s("segmentIDs", segmentIDs))

	waitFunc := func() (*commonpb.Status, error) {
		err = dct.WaitToFinish()
		if err != nil {
			status := &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			}
			log.Warn(err.Error())
			return status, err
		}
		log.Debug("loadSegmentsTask WaitToFinish done", zap.Int64s("segmentIDs", segmentIDs))
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	return waitFunc()
}

// ReleaseCollection clears all data related to this collecion on the querynode
func (node *QueryNode) ReleaseCollection(ctx context.Context, in *queryPb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	dct := &releaseCollectionTask{
		baseTask: baseTask{
			ctx:  ctx,
			done: make(chan error),
		},
		req:  in,
		node: node,
	}

	err := node.scheduler.queue.Enqueue(dct)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		log.Warn(err.Error())
		return status, err
	}
	log.Debug("releaseCollectionTask Enqueue done", zap.Any("collectionID", in.CollectionID))

	func() {
		err = dct.WaitToFinish()
		if err != nil {
			log.Warn(err.Error())
			return
		}
		log.Debug("releaseCollectionTask WaitToFinish done", zap.Any("collectionID", in.CollectionID))
	}()

	status := &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	return status, nil
}

// ReleasePartitions clears all data related to this partition on the querynode
func (node *QueryNode) ReleasePartitions(ctx context.Context, in *queryPb.ReleasePartitionsRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	dct := &releasePartitionsTask{
		baseTask: baseTask{
			ctx:  ctx,
			done: make(chan error),
		},
		req:  in,
		node: node,
	}

	err := node.scheduler.queue.Enqueue(dct)
	if err != nil {
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		log.Warn(err.Error())
		return status, err
	}
	log.Debug("releasePartitionsTask Enqueue done", zap.Any("collectionID", in.CollectionID))

	func() {
		err = dct.WaitToFinish()
		if err != nil {
			log.Warn(err.Error())
			return
		}
		log.Debug("releasePartitionsTask WaitToFinish done", zap.Any("collectionID", in.CollectionID))
	}()

	status := &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	return status, nil
}

// ReleaseSegments remove the specified segments from query node according segmentIDs, partitionIDs, and collectionID
func (node *QueryNode) ReleaseSegments(ctx context.Context, in *queryPb.ReleaseSegmentsRequest) (*commonpb.Status, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		status := &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}
		return status, err
	}
	status := &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	for _, id := range in.SegmentIDs {
		err := node.historical.replica.removeSegment(id)
		if err != nil {
			// not return, try to release all segments
			status.ErrorCode = commonpb.ErrorCode_UnexpectedError
			status.Reason = err.Error()
		}
		err = node.streaming.replica.removeSegment(id)
		if err != nil {
			// not return, try to release all segments
			status.ErrorCode = commonpb.ErrorCode_UnexpectedError
			status.Reason = err.Error()
		}
	}
	return status, nil
}

// GetSegmentInfo returns segment information of the collection on the queryNode, and the information includes memSize, numRow, indexName, indexID ...
func (node *QueryNode) GetSegmentInfo(ctx context.Context, in *queryPb.GetSegmentInfoRequest) (*queryPb.GetSegmentInfoResponse, error) {
	code := node.stateCode.Load().(internalpb.StateCode)
	if code != internalpb.StateCode_Healthy {
		err := fmt.Errorf("query node %d is not ready", Params.QueryNodeID)
		res := &queryPb.GetSegmentInfoResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
		}
		return res, err
	}
	infos := make([]*queryPb.SegmentInfo, 0)

	// get info from historical
	node.historical.replica.printReplica()
	historicalSegmentInfos, err := node.historical.replica.getSegmentInfosByColID(in.CollectionID)
	if err != nil {
		log.Debug("GetSegmentInfo: get historical segmentInfo failed", zap.Int64("collectionID", in.CollectionID), zap.Error(err))
		res := &queryPb.GetSegmentInfoResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
		}
		return res, err
	}
	infos = append(infos, historicalSegmentInfos...)

	// get info from streaming
	node.streaming.replica.printReplica()
	streamingSegmentInfos, err := node.streaming.replica.getSegmentInfosByColID(in.CollectionID)
	if err != nil {
		log.Debug("GetSegmentInfo: get streaming segmentInfo failed", zap.Int64("collectionID", in.CollectionID), zap.Error(err))
		res := &queryPb.GetSegmentInfoResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
		}
		return res, err
	}
	infos = append(infos, streamingSegmentInfos...)
	log.Debug("GetSegmentInfo: get segment info from query node", zap.Int64("nodeID", node.session.ServerID), zap.Any("segment infos", infos))

	return &queryPb.GetSegmentInfoResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Infos: infos,
	}, nil
}

func (node *QueryNode) isHealthy() bool {
	code := node.stateCode.Load().(internalpb.StateCode)
	return code == internalpb.StateCode_Healthy
}

// GetMetrics return system infos of the query node, such as total memory, memory uasge, cpu usage ...
// TODO(dragondriver): cache the Metrics and set a retention to the cache
func (node *QueryNode) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	log.Debug("QueryNode.GetMetrics",
		zap.Int64("node_id", Params.QueryNodeID),
		zap.String("req", req.Request))

	if !node.isHealthy() {
		log.Warn("QueryNode.GetMetrics failed",
			zap.Int64("node_id", Params.QueryNodeID),
			zap.String("req", req.Request),
			zap.Error(errQueryNodeIsUnhealthy(Params.QueryNodeID)))

		return &milvuspb.GetMetricsResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    msgQueryNodeIsUnhealthy(Params.QueryNodeID),
			},
			Response: "",
		}, nil
	}

	metricType, err := metricsinfo.ParseMetricType(req.Request)
	if err != nil {
		log.Warn("QueryNode.GetMetrics failed to parse metric type",
			zap.Int64("node_id", Params.QueryNodeID),
			zap.String("req", req.Request),
			zap.Error(err))

		return &milvuspb.GetMetricsResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
			Response: "",
		}, nil
	}

	log.Debug("QueryNode.GetMetrics",
		zap.String("metric_type", metricType))

	if metricType == metricsinfo.SystemInfoMetrics {
		metrics, err := getSystemInfoMetrics(ctx, req, node)

		log.Debug("QueryNode.GetMetrics",
			zap.Int64("node_id", Params.QueryNodeID),
			zap.String("req", req.Request),
			zap.String("metric_type", metricType),
			zap.Any("metrics", metrics), // TODO(dragondriver): necessary? may be very large
			zap.Error(err))

		return metrics, err
	}

	log.Debug("QueryNode.GetMetrics failed, request metric type is not implemented yet",
		zap.Int64("node_id", Params.QueryNodeID),
		zap.String("req", req.Request),
		zap.String("metric_type", metricType))

	return &milvuspb.GetMetricsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    metricsinfo.MsgUnimplementedMetric,
		},
		Response: "",
	}, nil
}
