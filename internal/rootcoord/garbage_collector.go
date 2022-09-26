package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/api/commonpb"
	ms "github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/metastore/model"
)

//go:generate mockery --name=GarbageCollector --outpkg=mockrootcoord
type GarbageCollector interface {
	ReDropCollection(collMeta *model.Collection, ts Timestamp)
	RemoveCreatingCollection(collMeta *model.Collection)
	ReDropPartition(pChannels []string, partition *model.Partition, ts Timestamp)
	GcCollectionData(ctx context.Context, coll *model.Collection) (ddlTs Timestamp, err error)
	GcPartitionData(ctx context.Context, pChannels []string, partition *model.Partition) (ddlTs Timestamp, err error)
}

type bgGarbageCollector struct {
	s *Core
}

func newBgGarbageCollector(s *Core) *bgGarbageCollector {
	return &bgGarbageCollector{s: s}
}

func (c *bgGarbageCollector) ReDropCollection(collMeta *model.Collection, ts Timestamp) {
	// TODO: remove this after data gc can be notified by rpc.
	c.s.chanTimeTick.addDmlChannels(collMeta.PhysicalChannelNames...)

	redo := newBaseRedoTask(c.s.stepExecutor)
	redo.AddAsyncStep(&releaseCollectionStep{
		baseStep:     baseStep{core: c.s},
		collectionID: collMeta.CollectionID,
	})
	redo.AddAsyncStep(&dropIndexStep{
		baseStep: baseStep{core: c.s},
		collID:   collMeta.CollectionID,
		partIDs:  nil,
	})
	redo.AddAsyncStep(&deleteCollectionDataStep{
		baseStep: baseStep{core: c.s},
		coll:     collMeta,
	})
	redo.AddAsyncStep(&removeDmlChannelsStep{
		baseStep:  baseStep{core: c.s},
		pChannels: collMeta.PhysicalChannelNames,
	})
	redo.AddAsyncStep(&deleteCollectionMetaStep{
		baseStep:     baseStep{core: c.s},
		collectionID: collMeta.CollectionID,
		// This ts is less than the ts when we notify data nodes to drop collection, but it's OK since we have already
		// marked this collection as deleted. If we want to make this ts greater than the notification's ts, we should
		// wrap a step who will have these three children and connect them with ts.
		ts: ts,
	})

	// err is ignored since no sync steps will be executed.
	_ = redo.Execute(context.Background())
}

func (c *bgGarbageCollector) RemoveCreatingCollection(collMeta *model.Collection) {
	// TODO: remove this after data gc can be notified by rpc.
	c.s.chanTimeTick.addDmlChannels(collMeta.PhysicalChannelNames...)

	redo := newBaseRedoTask(c.s.stepExecutor)

	redo.AddAsyncStep(&unwatchChannelsStep{
		baseStep:     baseStep{core: c.s},
		collectionID: collMeta.CollectionID,
		channels: collectionChannels{
			virtualChannels:  collMeta.VirtualChannelNames,
			physicalChannels: collMeta.PhysicalChannelNames,
		},
	})
	redo.AddAsyncStep(&removeDmlChannelsStep{
		baseStep:  baseStep{core: c.s},
		pChannels: collMeta.PhysicalChannelNames,
	})
	redo.AddAsyncStep(&deleteCollectionMetaStep{
		baseStep:     baseStep{core: c.s},
		collectionID: collMeta.CollectionID,
		// When we undo createCollectionTask, this ts may be less than the ts when unwatch channels.
		ts: collMeta.CreateTime,
	})
	// err is ignored since no sync steps will be executed.
	_ = redo.Execute(context.Background())
}

func (c *bgGarbageCollector) ReDropPartition(pChannels []string, partition *model.Partition, ts Timestamp) {
	// TODO: remove this after data gc can be notified by rpc.
	c.s.chanTimeTick.addDmlChannels(pChannels...)

	redo := newBaseRedoTask(c.s.stepExecutor)
	redo.AddAsyncStep(&deletePartitionDataStep{
		baseStep:  baseStep{core: c.s},
		pchans:    pChannels,
		partition: partition,
	})
	redo.AddAsyncStep(&removeDmlChannelsStep{
		baseStep:  baseStep{core: c.s},
		pChannels: pChannels,
	})
	redo.AddAsyncStep(&dropIndexStep{
		baseStep: baseStep{core: c.s},
		collID:   partition.CollectionID,
		partIDs:  []UniqueID{partition.PartitionID},
	})
	redo.AddAsyncStep(&removePartitionMetaStep{
		baseStep:     baseStep{core: c.s},
		collectionID: partition.CollectionID,
		partitionID:  partition.PartitionID,
		// This ts is less than the ts when we notify data nodes to drop partition, but it's OK since we have already
		// marked this partition as deleted. If we want to make this ts greater than the notification's ts, we should
		// wrap a step who will have these children and connect them with ts.
		ts: ts,
	})

	// err is ignored since no sync steps will be executed.
	_ = redo.Execute(context.Background())
}

func (c *bgGarbageCollector) notifyCollectionGc(ctx context.Context, coll *model.Collection) (ddlTs Timestamp, err error) {
	ts, err := c.s.tsoAllocator.GenerateTSO(1)
	if err != nil {
		return 0, err
	}

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
		return 0, err
	}

	return ts, nil
}

func (c *bgGarbageCollector) notifyPartitionGc(ctx context.Context, pChannels []string, partition *model.Partition) (ddlTs Timestamp, err error) {
	ts, err := c.s.tsoAllocator.GenerateTSO(1)
	if err != nil {
		return 0, err
	}

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
		return 0, err
	}

	return ts, nil
}

func (c *bgGarbageCollector) GcCollectionData(ctx context.Context, coll *model.Collection) (ddlTs Timestamp, err error) {
	c.s.ddlTsLockManager.Lock()
	c.s.ddlTsLockManager.AddRefCnt(1)
	defer c.s.ddlTsLockManager.AddRefCnt(-1)
	defer c.s.ddlTsLockManager.Unlock()

	ddlTs, err = c.notifyCollectionGc(ctx, coll)
	if err != nil {
		return 0, err
	}
	c.s.ddlTsLockManager.UpdateLastTs(ddlTs)
	return ddlTs, nil
}

func (c *bgGarbageCollector) GcPartitionData(ctx context.Context, pChannels []string, partition *model.Partition) (ddlTs Timestamp, err error) {
	c.s.ddlTsLockManager.Lock()
	c.s.ddlTsLockManager.AddRefCnt(1)
	defer c.s.ddlTsLockManager.AddRefCnt(-1)
	defer c.s.ddlTsLockManager.Unlock()

	ddlTs, err = c.notifyPartitionGc(ctx, pChannels, partition)
	if err != nil {
		return 0, err
	}
	c.s.ddlTsLockManager.UpdateLastTs(ddlTs)
	return ddlTs, nil
}
