package model

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/schemapb"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
)

var (
	colID      int64 = 1
	colName          = "c"
	fieldID    int64 = 101
	fieldName        = "field110"
	partID     int64 = 20
	partName         = "testPart"
	tenantID         = "tenant-1"
	typeParams       = []*commonpb.KeyValuePair{
		{
			Key:   "field110-k1",
			Value: "field110-v1",
		},
	}
	startPositions = []*commonpb.KeyDataPair{
		{
			Key:  "k1",
			Data: []byte{byte(1)},
		},
	}

	colModel = &Collection{
		TenantID:             tenantID,
		CollectionID:         colID,
		Name:                 colName,
		AutoID:               false,
		Description:          "none",
		Fields:               []*Field{fieldModel},
		VirtualChannelNames:  []string{"vch"},
		PhysicalChannelNames: []string{"pch"},
		ShardsNum:            1,
		CreateTime:           1,
		StartPositions:       startPositions,
		ConsistencyLevel:     commonpb.ConsistencyLevel_Strong,
		Partitions: []*Partition{
			{
				PartitionID:               partID,
				PartitionName:             partName,
				PartitionCreatedTimestamp: 1,
			},
		},
	}

	deprecatedColPb = &pb.CollectionInfo{
		ID: colID,
		Schema: &schemapb.CollectionSchema{
			Name:        colName,
			Description: "none",
			AutoID:      false,
			Fields:      []*schemapb.FieldSchema{filedSchemaPb},
		},
		CreateTime:                 1,
		PartitionIDs:               []int64{partID},
		PartitionNames:             []string{partName},
		PartitionCreatedTimestamps: []uint64{1},
		FieldIndexes: []*pb.FieldIndexInfo{
			{
				FiledID: fieldID,
				IndexID: indexID,
			},
		},
		VirtualChannelNames:  []string{"vch"},
		PhysicalChannelNames: []string{"pch"},
		ShardsNum:            1,
		StartPositions:       startPositions,
		ConsistencyLevel:     commonpb.ConsistencyLevel_Strong,
	}
)

func TestUnmarshalCollectionModel(t *testing.T) {
	ret := UnmarshalCollectionModel(deprecatedColPb)
	ret.TenantID = tenantID
	assert.Equal(t, ret, colModel)

	assert.Nil(t, UnmarshalCollectionModel(nil))
}

func TestMarshalCollectionModel(t *testing.T) {
	assert.Nil(t, MarshalCollectionModel(nil))
}
