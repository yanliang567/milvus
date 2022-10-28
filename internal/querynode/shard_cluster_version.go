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
	"fmt"
	"strings"
	"sync"

	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
)

// SegmentsStatus alias for map[int64]shardSegmentInfo.
// Provides some helper function to get segment allocation.
type SegmentsStatus map[int64]shardSegmentInfo

// String implements Stringer for log.
func (s SegmentsStatus) String() string {
	// get nodeID => []segmentID
	allocation := s.GetAllocations(nil)

	var builder strings.Builder
	builder.WriteRune('{')
	for nodeID, segmentIDs := range allocation {
		builder.WriteString(fmt.Sprintf("Node %d: %v ", nodeID, segmentIDs))
	}
	builder.WriteRune('}')
	return builder.String()
}

// GetAllocations  returns node to segments mappings.
func (s SegmentsStatus) GetAllocations(partitionIDs []int64) map[int64][]int64 {
	result := make(map[int64][]int64) // nodeID => segmentIDs
	// only read operations here, no need to lock
	for _, segment := range s {
		if len(partitionIDs) > 0 && !inList(partitionIDs, segment.partitionID) {
			continue
		}

		result[segment.nodeID] = append(result[segment.nodeID], segment.segmentID)
	}
	return result
}

// Clone returns a copy of segments status data.
func (s SegmentsStatus) Clone(filter func(int64, int64) bool) SegmentsStatus {
	c := make(map[int64]shardSegmentInfo)
	for k, v := range s {
		if filter(v.segmentID, v.nodeID) {
			continue
		}
		c[k] = v
	}
	return c
}

// helper filter function that filters nothing
var filterNothing = func(int64, int64) bool { return false }

// ShardClusterVersion maintains a snapshot of sealed segments allocation.
type ShardClusterVersion struct {
	versionID   int64          // identifier for version
	segments    SegmentsStatus // nodeID => []segmentID
	current     *atomic.Bool   // is this version current
	inUse       *atomic.Int64
	lastVersion *ShardClusterVersion

	ch        chan struct{} // signal channel to notify safe remove
	closeOnce sync.Once
}

// NewShardClusterVersion creates a version with id and allocation.
func NewShardClusterVersion(vID int64, status SegmentsStatus, lastVersion *ShardClusterVersion) *ShardClusterVersion {
	log.Info("Update shard cluster version",
		zap.Int64("newVersionID", vID),
		zap.String("newAllocation", status.String()),
	)
	if lastVersion != nil {
		// ignore the expiration channel here
		_ = lastVersion.Expire()
	}
	return &ShardClusterVersion{
		versionID:   vID,
		segments:    status,
		current:     atomic.NewBool(true), // by default new version will be current
		inUse:       atomic.NewInt64(0),
		ch:          make(chan struct{}),
		lastVersion: lastVersion,
	}
}

// IsCurrent returns whether this version is current version.
func (v *ShardClusterVersion) IsCurrent() bool {
	return v.current.Load()
}

// GetAllocation returns version allocation and record in-use.
func (v *ShardClusterVersion) GetAllocation(partitionIDs []int64) map[int64][]int64 {
	v.inUse.Add(1)
	return v.segments.GetAllocations(partitionIDs)
}

// FinishUsage decreases the inUse count and cause pending change check.
func (v *ShardClusterVersion) FinishUsage() {
	v.inUse.Add(-1)
	v.checkSafeGC()
}

// Expire sets the current flag to false for this version.
// invocation shall be goroutine safe for Expire.
func (v *ShardClusterVersion) Expire() chan struct{} {
	v.current.Store(false)
	v.checkSafeGC()
	return v.ch
}

// checkSafeGC check version is safe to release changeInfo offline segments.
func (v *ShardClusterVersion) checkSafeGC() {
	if !v.IsCurrent() && v.inUse.Load() == int64(0) {
		v.closeOnce.Do(func() {
			go func() {
				if v.lastVersion != nil {
					<-v.lastVersion.Expire()
				}
				// release the reference of last version, so it could be processed by gc
				v.lastVersion = nil
				close(v.ch)
			}()
		})
	}
}
