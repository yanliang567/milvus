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

import (
	"context"
	"errors"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/mq/msgstream/mqwrapper"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
)

// queryNodeFlowGraph is a TimeTickedFlowGraph in query node
type queryNodeFlowGraph struct {
	ctx          context.Context
	cancel       context.CancelFunc
	collectionID UniqueID
	channel      Channel
	flowGraph    *flowgraph.TimeTickedFlowGraph
	dmlStream    msgstream.MsgStream
	consumerCnt  int
}

// newQueryNodeFlowGraph returns a new queryNodeFlowGraph
func newQueryNodeFlowGraph(ctx context.Context,
	collectionID UniqueID,
	metaReplica ReplicaInterface,
	tSafeReplica TSafeReplicaInterface,
	channel Channel,
	factory msgstream.Factory) (*queryNodeFlowGraph, error) {

	ctx1, cancel := context.WithCancel(ctx)

	q := &queryNodeFlowGraph{
		ctx:          ctx1,
		cancel:       cancel,
		collectionID: collectionID,
		channel:      channel,
		flowGraph:    flowgraph.NewTimeTickedFlowGraph(ctx1),
	}

	dmStreamNode, err := q.newDmInputNode(ctx1, factory, collectionID, channel)
	if err != nil {
		return nil, err
	}
	var filterDmNode node = newFilteredDmNode(metaReplica, collectionID, channel)
	var insertNode node = newInsertNode(metaReplica, collectionID, channel)
	var serviceTimeNode node = newServiceTimeNode(tSafeReplica, collectionID, channel)

	q.flowGraph.AddNode(dmStreamNode)
	q.flowGraph.AddNode(filterDmNode)
	q.flowGraph.AddNode(insertNode)
	q.flowGraph.AddNode(serviceTimeNode)

	// dmStreamNode
	err = q.flowGraph.SetEdges(dmStreamNode.Name(),
		[]string{filterDmNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", dmStreamNode.Name(), err.Error())
	}

	// filterDmNode
	err = q.flowGraph.SetEdges(filterDmNode.Name(),
		[]string{insertNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", filterDmNode.Name(), err.Error())
	}

	// insertNode
	err = q.flowGraph.SetEdges(insertNode.Name(),
		[]string{serviceTimeNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", insertNode.Name(), err.Error())
	}

	// serviceTimeNode
	err = q.flowGraph.SetEdges(serviceTimeNode.Name(),
		[]string{},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", serviceTimeNode.Name(), err.Error())
	}

	return q, nil
}

// newQueryNodeDeltaFlowGraph returns a new queryNodeFlowGraph
func newQueryNodeDeltaFlowGraph(ctx context.Context,
	collectionID UniqueID,
	metaReplica ReplicaInterface,
	tSafeReplica TSafeReplicaInterface,
	channel Channel,
	factory msgstream.Factory) (*queryNodeFlowGraph, error) {

	ctx1, cancel := context.WithCancel(ctx)

	q := &queryNodeFlowGraph{
		ctx:          ctx1,
		cancel:       cancel,
		collectionID: collectionID,
		channel:      channel,
		flowGraph:    flowgraph.NewTimeTickedFlowGraph(ctx1),
	}

	dmStreamNode, err := q.newDmInputNode(ctx1, factory, collectionID, channel)
	if err != nil {
		return nil, err
	}
	var filterDeleteNode node = newFilteredDeleteNode(metaReplica, collectionID, channel)
	var deleteNode node = newDeleteNode(metaReplica, collectionID, channel)
	var serviceTimeNode node = newServiceTimeNode(tSafeReplica, collectionID, channel)

	q.flowGraph.AddNode(dmStreamNode)
	q.flowGraph.AddNode(filterDeleteNode)
	q.flowGraph.AddNode(deleteNode)
	q.flowGraph.AddNode(serviceTimeNode)

	// dmStreamNode
	err = q.flowGraph.SetEdges(dmStreamNode.Name(),
		[]string{filterDeleteNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", dmStreamNode.Name(), err.Error())
	}

	// filterDmNode
	err = q.flowGraph.SetEdges(filterDeleteNode.Name(),
		[]string{deleteNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", filterDeleteNode.Name(), err.Error())
	}

	// insertNode
	err = q.flowGraph.SetEdges(deleteNode.Name(),
		[]string{serviceTimeNode.Name()},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", deleteNode.Name(), err.Error())
	}

	// serviceTimeNode
	err = q.flowGraph.SetEdges(serviceTimeNode.Name(),
		[]string{},
	)
	if err != nil {
		return nil, fmt.Errorf("set edges failed in node: %s, err = %s", serviceTimeNode.Name(), err.Error())
	}

	return q, nil
}

// newDmInputNode returns a new inputNode
func (q *queryNodeFlowGraph) newDmInputNode(ctx context.Context, factory msgstream.Factory, collectionID UniqueID, channel Channel) (*flowgraph.InputNode, error) {
	insertStream, err := factory.NewTtMsgStream(ctx)
	if err != nil {
		return nil, err
	}

	q.dmlStream = insertStream

	maxQueueLength := Params.QueryNodeCfg.FlowGraphMaxQueueLength
	maxParallelism := Params.QueryNodeCfg.FlowGraphMaxParallelism
	name := fmt.Sprintf("dmInputNode-query-%d-%s", collectionID, channel)
	node := flowgraph.NewInputNode(insertStream, name, maxQueueLength, maxParallelism)
	return node, nil
}

// consumeFlowGraph would consume by channel and subName
func (q *queryNodeFlowGraph) consumeFlowGraph(channel Channel, subName ConsumeSubName) error {
	if q.dmlStream == nil {
		return errors.New("null dml message stream in flow graph")
	}
	q.dmlStream.AsConsumer([]string{channel}, subName)
	log.Info("query node flow graph consumes from pChannel",
		zap.Any("collectionID", q.collectionID),
		zap.Any("channel", channel),
		zap.Any("subName", subName),
	)
	q.consumerCnt++
	metrics.QueryNodeNumConsumers.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Inc()
	return nil
}

// consumeFlowGraphFromLatest would consume from latest by channel and subName
func (q *queryNodeFlowGraph) consumeFlowGraphFromLatest(channel Channel, subName ConsumeSubName) error {
	if q.dmlStream == nil {
		return errors.New("null dml message stream in flow graph")
	}
	q.dmlStream.AsConsumerWithPosition([]string{channel}, subName, mqwrapper.SubscriptionPositionLatest)
	log.Info("query node flow graph consumes from pChannel",
		zap.Any("collectionID", q.collectionID),
		zap.Any("channel", channel),
		zap.Any("subName", subName),
	)
	q.consumerCnt++
	metrics.QueryNodeNumConsumers.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Inc()
	return nil
}

// seekQueryNodeFlowGraph would seek by position
func (q *queryNodeFlowGraph) seekQueryNodeFlowGraph(position *internalpb.MsgPosition) error {
	q.dmlStream.AsConsumer([]string{position.ChannelName}, position.MsgGroup)
	err := q.dmlStream.Seek([]*internalpb.MsgPosition{position})

	ts, _ := tsoutil.ParseTS(position.GetTimestamp())
	log.Info("query node flow graph seeks from pChannel",
		zap.Int64("collectionID", q.collectionID),
		zap.String("channel", position.ChannelName),
		zap.Time("checkpointTs", ts),
		zap.Duration("tsLag", time.Since(ts)),
	)
	q.consumerCnt++
	metrics.QueryNodeNumConsumers.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Inc()
	return err
}

// close would close queryNodeFlowGraph
func (q *queryNodeFlowGraph) close() {
	q.cancel()
	q.flowGraph.Close()
	if q.dmlStream != nil && q.consumerCnt > 0 {
		metrics.QueryNodeNumConsumers.WithLabelValues(fmt.Sprint(Params.QueryNodeCfg.GetNodeID())).Sub(float64(q.consumerCnt))
	}
	log.Info("stop query node flow graph",
		zap.Any("collectionID", q.collectionID),
		zap.Any("channel", q.channel),
	)
}
