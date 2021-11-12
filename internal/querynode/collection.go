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

/*

#cgo CFLAGS: -I${SRCDIR}/../core/output/include

#cgo LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"

*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"go.uber.org/zap"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

// Collection is a wrapper of the underlying C-structure C.CCollection
type Collection struct {
	collectionPtr C.CCollection
	id            UniqueID
	partitionIDs  []UniqueID
	schema        *schemapb.CollectionSchema
	vChannels     []Channel
	pChannels     []Channel

	vDeltaChannels []Channel
	pDeltaChannels []Channel

	loadType loadType

	releaseMu          sync.RWMutex // guards release
	releasedPartitions map[UniqueID]struct{}
	releaseTime        Timestamp
}

// ID returns collection id
func (c *Collection) ID() UniqueID {
	return c.id
}

// Schema returns the schema of collection
func (c *Collection) Schema() *schemapb.CollectionSchema {
	return c.schema
}

// addPartitionID would add a partition id to partition id list of collection
func (c *Collection) addPartitionID(partitionID UniqueID) {
	c.releaseMu.Lock()
	defer c.releaseMu.Unlock()

	log.Debug("queryNode collection add a partition", zap.Int64("collection", c.id), zap.Int64("partitionID", partitionID))
	c.partitionIDs = append(c.partitionIDs, partitionID)
	log.Debug("queryNode collection info after add a partition", zap.Int64("collectionID", c.id), zap.Int64s("partitions", c.partitionIDs), zap.Any("releasePartitions", c.releasedPartitions))
}

// removePartitionID remove the partition id from partition id list of collection
func (c *Collection) removePartitionID(partitionID UniqueID) {
	tmpIDs := make([]UniqueID, 0)
	for _, id := range c.partitionIDs {
		if id != partitionID {
			tmpIDs = append(tmpIDs, id)
		}
	}
	c.partitionIDs = tmpIDs
}

// addVChannels add virtual channels to collection
func (c *Collection) addVChannels(channels []Channel) {
OUTER:
	for _, dstChan := range channels {
		for _, srcChan := range c.vChannels {
			if dstChan == srcChan {
				log.Debug("vChannel has been existed in collection's vChannels",
					zap.Any("collectionID", c.ID()),
					zap.Any("vChannel", dstChan),
				)
				continue OUTER
			}
		}
		log.Debug("add vChannel to collection",
			zap.Any("collectionID", c.ID()),
			zap.Any("vChannel", dstChan),
		)
		c.vChannels = append(c.vChannels, dstChan)
	}
}

// getVChannels get virtual channels of collection
func (c *Collection) getVChannels() []Channel {
	return c.vChannels
}

// addPChannels add physical channels to physical channels of collection
func (c *Collection) addPChannels(channels []Channel) {
OUTER:
	for _, dstChan := range channels {
		for _, srcChan := range c.pChannels {
			if dstChan == srcChan {
				log.Debug("pChannel has been existed in collection's pChannels",
					zap.Any("collectionID", c.ID()),
					zap.Any("pChannel", dstChan),
				)
				continue OUTER
			}
		}
		log.Debug("add pChannel to collection",
			zap.Any("collectionID", c.ID()),
			zap.Any("pChannel", dstChan),
		)
		c.pChannels = append(c.pChannels, dstChan)
	}
}

// getPChannels get physical channels of collection
func (c *Collection) getPChannels() []Channel {
	return c.pChannels
}

// addPChannels add physical channels to physical channels of collection
func (c *Collection) addPDeltaChannels(channels []Channel) {
OUTER:
	for _, dstChan := range channels {
		for _, srcChan := range c.pDeltaChannels {
			if dstChan == srcChan {
				log.Debug("pChannel has been existed in collection's pChannels",
					zap.Any("collectionID", c.ID()),
					zap.Any("pChannel", dstChan),
				)
				continue OUTER
			}
		}
		log.Debug("add pChannel to collection",
			zap.Any("collectionID", c.ID()),
			zap.Any("pChannel", dstChan),
		)
		c.pDeltaChannels = append(c.pDeltaChannels, dstChan)
	}
}

// getPChannels get physical channels of collection
func (c *Collection) getPDeltaChannels() []Channel {
	return c.pDeltaChannels
}

func (c *Collection) getVDeltaChannels() []Channel {
	return c.vDeltaChannels
}

// addVChannels add virtual channels to collection
func (c *Collection) addVDeltaChannels(channels []Channel) {
OUTER:
	for _, dstChan := range channels {
		for _, srcChan := range c.vDeltaChannels {
			if dstChan == srcChan {
				log.Debug("vDeltaChannel has been existed in collection's vDeltaChannels",
					zap.Any("collectionID", c.ID()),
					zap.Any("vChannel", dstChan),
				)
				continue OUTER
			}
		}
		log.Debug("add vDeltaChannel to collection",
			zap.Any("collectionID", c.ID()),
			zap.Any("vDeltaChannel", dstChan),
		)
		c.vDeltaChannels = append(c.vDeltaChannels, dstChan)
	}
}

// setReleaseTime records when collection is released
func (c *Collection) setReleaseTime(t Timestamp) {
	c.releaseMu.Lock()
	defer c.releaseMu.Unlock()
	c.releaseTime = t
}

// getReleaseTime gets the time when collection is released
func (c *Collection) getReleaseTime() Timestamp {
	c.releaseMu.RLock()
	defer c.releaseMu.RUnlock()
	return c.releaseTime
}

// addReleasedPartition records the partition to indicate that this partition has been released
func (c *Collection) addReleasedPartition(partitionID UniqueID) {
	c.releaseMu.Lock()
	defer c.releaseMu.Unlock()

	log.Debug("queryNode collection release a partition", zap.Int64("collectionID", c.id), zap.Int64("partition", partitionID))
	c.releasedPartitions[partitionID] = struct{}{}
	partitions := make([]UniqueID, 0)
	for _, id := range c.partitionIDs {
		if id != partitionID {
			partitions = append(partitions, id)
		}
	}
	c.partitionIDs = partitions
	log.Debug("queryNode collection info after release a partition", zap.Int64("collectionID", c.id), zap.Int64s("partitions", c.partitionIDs), zap.Any("releasePartitions", c.releasedPartitions))
}

// deleteReleasedPartition remove the released partition record from collection
func (c *Collection) deleteReleasedPartition(partitionID UniqueID) {
	c.releaseMu.Lock()
	defer c.releaseMu.Unlock()

	log.Debug("queryNode collection reload a released partition", zap.Int64("collectionID", c.id), zap.Int64("partition", partitionID))
	delete(c.releasedPartitions, partitionID)
	log.Debug("queryNode collection info after reload a released partition", zap.Int64("collectionID", c.id), zap.Int64s("partitions", c.partitionIDs), zap.Any("releasePartitions", c.releasedPartitions))
}

// checkReleasedPartitions returns error if any partition has been released
func (c *Collection) checkReleasedPartitions(partitionIDs []UniqueID) error {
	c.releaseMu.RLock()
	defer c.releaseMu.RUnlock()
	for _, id := range partitionIDs {
		if _, ok := c.releasedPartitions[id]; ok {
			return errors.New("partition has been released" +
				", collectionID = " + fmt.Sprintln(c.ID()) +
				", partitionID = " + fmt.Sprintln(id))
		}
	}

	return nil
}

// setLoadType set the loading type of collection, which is loadTypeCollection or loadTypePartition
func (c *Collection) setLoadType(l loadType) {
	c.loadType = l
}

// getLoadType get the loadType of collection, which is loadTypeCollection or loadTypePartition
func (c *Collection) getLoadType() loadType {
	return c.loadType
}

// newCollection returns a new Collection
func newCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) *Collection {
	/*
		CCollection
		NewCollection(const char* schema_proto_blob);
	*/
	schemaBlob := proto.MarshalTextString(schema)

	cSchemaBlob := C.CString(schemaBlob)
	collection := C.NewCollection(cSchemaBlob)

	var newCollection = &Collection{
		collectionPtr:      collection,
		id:                 collectionID,
		schema:             schema,
		vChannels:          make([]Channel, 0),
		pChannels:          make([]Channel, 0),
		vDeltaChannels:     make([]Channel, 0),
		pDeltaChannels:     make([]Channel, 0),
		releasedPartitions: make(map[UniqueID]struct{}),
	}
	C.free(unsafe.Pointer(cSchemaBlob))

	log.Debug("create collection", zap.Int64("collectionID", collectionID))

	newCollection.setReleaseTime(Timestamp(math.MaxUint64))
	return newCollection
}

// deleteCollection delete collection and free the collection memory
func deleteCollection(collection *Collection) {
	/*
		void
		deleteCollection(CCollection collection);
	*/
	cPtr := collection.collectionPtr
	C.DeleteCollection(cPtr)

	collection.collectionPtr = nil

	log.Debug("delete collection", zap.Int64("collectionID", collection.ID()))

	collection = nil
}
