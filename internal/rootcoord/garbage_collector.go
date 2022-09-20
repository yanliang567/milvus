package rootcoord

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/api/commonpb"
	ms "github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"go.uber.org/zap"
)

type GarbageCollector interface {
	ReDropCollection(collMeta *model.Collection, ts Timestamp)
	RemoveCreatingCollection(collMeta *model.Collection)
	ReDropPartition(pChannels []string, partition *model.Partition, ts Timestamp)
	GcCollectionData(ctx context.Context, coll *model.Collection, ts typeutil.Timestamp) error
	GcPartitionData(ctx context.Context, pChannels []string, partition *model.Partition, ts typeutil.Timestamp) error
}

type GarbageCollectorCtx struct {
	s *Core
}

func newGarbageCollectorCtx(s *Core) *GarbageCollectorCtx {
	return &GarbageCollectorCtx{s: s}
}

func (c *GarbageCollectorCtx) ReDropCollection(collMeta *model.Collection, ts Timestamp) {
	// TODO: remove this after data gc can be notified by rpc.
	c.s.chanTimeTick.addDmlChannels(collMeta.PhysicalChannelNames...)
	defer c.s.chanTimeTick.removeDmlChannels(collMeta.PhysicalChannelNames...)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	if err := c.s.broker.ReleaseCollection(ctx, collMeta.CollectionID); err != nil {
		log.Error("failed to release collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
		return
	}

	if err := c.s.broker.DropCollectionIndex(ctx, collMeta.CollectionID); err != nil {
		log.Error("failed to drop collection index when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
		return
	}

	if err := c.GcCollectionData(ctx, collMeta, ts); err != nil {
		log.Error("failed to notify datacoord to gc collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
		return
	}

	if err := c.s.meta.RemoveCollection(ctx, collMeta.CollectionID, ts); err != nil {
		log.Error("failed to remove collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
	}
}

func (c *GarbageCollectorCtx) RemoveCreatingCollection(collMeta *model.Collection) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	if err := c.s.broker.UnwatchChannels(ctx, &watchInfo{collectionID: collMeta.CollectionID, vChannels: collMeta.VirtualChannelNames}); err != nil {
		log.Error("failed to unwatch channels when recovery",
			zap.Error(err),
			zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID),
			zap.Strings("vchans", collMeta.VirtualChannelNames), zap.Strings("pchans", collMeta.PhysicalChannelNames))
		return
	}

	if err := c.s.meta.RemoveCollection(ctx, collMeta.CollectionID, collMeta.CreateTime); err != nil {
		log.Error("failed to remove collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
	}
}

func (c *GarbageCollectorCtx) ReDropPartition(pChannels []string, partition *model.Partition, ts Timestamp) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	// TODO: release partition when query coord is ready.

	// TODO: remove this after data gc can be notified by rpc.
	c.s.chanTimeTick.addDmlChannels(pChannels...)
	defer c.s.chanTimeTick.removeDmlChannels(pChannels...)

	if err := c.GcPartitionData(ctx, pChannels, partition, ts); err != nil {
		log.Error("failed to notify datanodes to gc partition", zap.Error(err))
		return
	}

	if err := c.s.meta.RemovePartition(ctx, partition.CollectionID, partition.PartitionID, ts); err != nil {
		log.Error("failed to remove partition when recovery", zap.Error(err))
	}
}

func (c *GarbageCollectorCtx) GcCollectionData(ctx context.Context, coll *model.Collection, ts typeutil.Timestamp) error {
	msgPack := ms.MsgPack{}
	baseMsg := ms.BaseMsg{
		Ctx:            ctx,
		BeginTimestamp: ts,
		EndTimestamp:   ts,
		HashValues:     []uint32{0},
	}
	msg := &ms.DropCollectionMsg{
		BaseMsg: baseMsg,
		DropCollectionRequest: internalpb.DropCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_DropCollection,
				Timestamp: ts,
				SourceID:  c.s.session.ServerID,
			},
			CollectionName: coll.Name,
			CollectionID:   coll.CollectionID,
		},
	}
	msgPack.Msgs = append(msgPack.Msgs, msg)
	if err := c.s.chanTimeTick.broadcastDmlChannels(coll.PhysicalChannelNames, &msgPack); err != nil {
		return err
	}

	// TODO: remove this after gc can be notified by rpc. Without this tt, DropCollectionMsg cannot be seen by
	// 		datanodes.
	return c.s.chanTimeTick.sendTimeTickToChannel(coll.PhysicalChannelNames, ts)
}

func (c *GarbageCollectorCtx) GcPartitionData(ctx context.Context, pChannels []string, partition *model.Partition, ts typeutil.Timestamp) error {
	msgPack := ms.MsgPack{}
	baseMsg := ms.BaseMsg{
		Ctx:            ctx,
		BeginTimestamp: ts,
		EndTimestamp:   ts,
		HashValues:     []uint32{0},
	}
	msg := &ms.DropPartitionMsg{
		BaseMsg: baseMsg,
		DropPartitionRequest: internalpb.DropPartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_DropPartition,
				Timestamp: ts,
				SourceID:  c.s.session.ServerID,
			},
			PartitionName: partition.PartitionName,
			CollectionID:  partition.CollectionID,
			PartitionID:   partition.PartitionID,
		},
	}
	msgPack.Msgs = append(msgPack.Msgs, msg)
	if err := c.s.chanTimeTick.broadcastDmlChannels(pChannels, &msgPack); err != nil {
		return err
	}

	// TODO: remove this after gc can be notified by rpc. Without this tt, DropCollectionMsg cannot be seen by
	// 		datanodes.
	return c.s.chanTimeTick.sendTimeTickToChannel(pChannels, ts)
}
