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

package utils

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
)

func GetReplicaNodesInfo(replicaMgr *meta.ReplicaManager, nodeMgr *session.NodeManager, replicaID int64) []*session.NodeInfo {
	replica := replicaMgr.Get(replicaID)
	if replica == nil {
		return nil
	}

	nodes := make([]*session.NodeInfo, 0, len(replica.Nodes))
	for node := range replica.Nodes {
		nodes = append(nodes, nodeMgr.Get(node))
	}
	return nodes
}

func GetPartitions(collectionMgr *meta.CollectionManager, broker meta.Broker, collectionID int64) ([]int64, error) {
	collection := collectionMgr.GetCollection(collectionID)
	if collection != nil {
		partitions, err := broker.GetPartitions(context.Background(), collectionID)
		return partitions, err
	}

	partitions := collectionMgr.GetPartitionsByCollection(collectionID)
	if partitions != nil {
		return lo.Map(partitions, func(partition *meta.Partition, i int) int64 {
			return partition.PartitionID
		}), nil
	}

	// todo(yah01): replace this error with a defined error
	return nil, fmt.Errorf("collection/partition not loaded")
}

// GroupNodesByReplica groups nodes by replica,
// returns ReplicaID -> NodeIDs
func GroupNodesByReplica(replicaMgr *meta.ReplicaManager, collectionID int64, nodes []int64) map[int64][]int64 {
	ret := make(map[int64][]int64)
	replicas := replicaMgr.GetByCollection(collectionID)
	for _, replica := range replicas {
		for _, node := range nodes {
			if replica.Nodes.Contain(node) {
				ret[replica.ID] = append(ret[replica.ID], node)
			}
		}
	}
	return ret
}

// GroupPartitionsByCollection groups partitions by collection,
// returns CollectionID -> Partitions
func GroupPartitionsByCollection(partitions []*meta.Partition) map[int64][]*meta.Partition {
	ret := make(map[int64][]*meta.Partition, 0)
	for _, partition := range partitions {
		collection := partition.GetCollectionID()
		ret[collection] = append(ret[collection], partition)
	}
	return ret
}

// GroupSegmentsByReplica groups segments by replica,
// returns ReplicaID -> Segments
func GroupSegmentsByReplica(replicaMgr *meta.ReplicaManager, collectionID int64, segments []*meta.Segment) map[int64][]*meta.Segment {
	ret := make(map[int64][]*meta.Segment)
	replicas := replicaMgr.GetByCollection(collectionID)
	for _, replica := range replicas {
		for _, segment := range segments {
			if replica.Nodes.Contain(segment.Node) {
				ret[replica.ID] = append(ret[replica.ID], segment)
			}
		}
	}
	return ret
}

// AssignNodesToReplicas assigns nodes to the given replicas,
// all given replicas must be the same collection,
// the given replicas have to be not in ReplicaManager
func AssignNodesToReplicas(nodeMgr *session.NodeManager, replicas ...*meta.Replica) {
	replicaNumber := len(replicas)
	nodes := nodeMgr.GetAll()
	rand.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})

	for i, node := range nodes {
		replicas[i%replicaNumber].AddNode(node.ID())
	}
}

// SpawnReplicas spawns replicas for given collection, assign nodes to them, and save them
func SpawnReplicas(replicaMgr *meta.ReplicaManager, nodeMgr *session.NodeManager, collection int64, replicaNumber int32) ([]*meta.Replica, error) {
	replicas, err := replicaMgr.Spawn(collection, replicaNumber)
	if err != nil {
		return nil, err
	}
	AssignNodesToReplicas(nodeMgr, replicas...)
	return replicas, replicaMgr.Put(replicas...)
}
