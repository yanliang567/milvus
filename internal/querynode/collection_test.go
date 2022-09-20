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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/schemapb"
)

func TestCollection_newCollection(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	assert.Equal(t, collection.ID(), collectionID)
}

func TestCollection_deleteCollection(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	assert.Equal(t, collection.ID(), collectionID)
	deleteCollection(collection)
}

func TestCollection_schema(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collectionSchema := collection.Schema()
	assert.Equal(t, schema.Name, collectionSchema.Name)
	assert.Equal(t, len(schema.Fields), len(collectionSchema.Fields))
	deleteCollection(collection)
}

func TestCollection_vChannel(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collection.addVChannels([]Channel{defaultDMLChannel})
	collection.addVChannels([]Channel{defaultDMLChannel})
	collection.addVChannels([]Channel{"TestCollection_addVChannel_channel"})

	channels := collection.getVChannels()
	assert.Equal(t, 2, len(channels))

	collection.removeVChannel(defaultDMLChannel)
	channels = collection.getVChannels()
	assert.Equal(t, 1, len(channels))
}

func TestCollection_vDeltaChannel(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collection.addVDeltaChannels([]Channel{defaultDeltaChannel})
	collection.addVDeltaChannels([]Channel{defaultDeltaChannel})
	collection.addVDeltaChannels([]Channel{"TestCollection_addVDeltaChannel_channel"})

	channels := collection.getVDeltaChannels()
	assert.Equal(t, 2, len(channels))

	collection.removeVDeltaChannel(defaultDeltaChannel)
	channels = collection.getVDeltaChannels()
	assert.Equal(t, 1, len(channels))
}

func TestCollection_pChannel(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collection.addPChannels([]Channel{"TestCollection_addPChannel_channel-0"})
	collection.addPChannels([]Channel{"TestCollection_addPChannel_channel-0"})
	collection.addPChannels([]Channel{"TestCollection_addPChannel_channel-1"})

	channels := collection.getPChannels()
	assert.Equal(t, 2, len(channels))
}

func TestCollection_pDeltaChannel(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collection.addPDeltaChannels([]Channel{"TestCollection_addPDeltaChannel_channel-0"})
	collection.addPDeltaChannels([]Channel{"TestCollection_addPDeltaChannel_channel-0"})
	collection.addPDeltaChannels([]Channel{"TestCollection_addPDeltaChannel_channel-1"})

	channels := collection.getPDeltaChannels()
	assert.Equal(t, 2, len(channels))
}

func TestCollection_releaseTime(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	t0 := Timestamp(1000)
	collection.setReleaseTime(t0, true)
	t1, released := collection.getReleaseTime()
	assert.Equal(t, t0, t1)
	assert.True(t, released)
}

func TestCollection_loadType(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)
	collection.setLoadType(loadTypeCollection)
	lt := collection.getLoadType()
	assert.Equal(t, loadTypeCollection, lt)

	collection.setLoadType(loadTypePartition)
	lt = collection.getLoadType()
	assert.Equal(t, loadTypePartition, lt)
}

func TestCollection_getFieldType(t *testing.T) {
	coll := &Collection{schema: nil}
	_, err := coll.getFieldType(100)
	assert.Error(t, err)
	coll.schema = &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				Name:     "test",
				FieldID:  100,
				DataType: schemapb.DataType_Int64,
			},
		},
	}
	// field id not found.
	_, err = coll.getFieldType(101)
	assert.Error(t, err)
	fieldType, err := coll.getFieldType(100)
	assert.NoError(t, err)
	assert.Equal(t, schemapb.DataType_Int64, fieldType)
}
