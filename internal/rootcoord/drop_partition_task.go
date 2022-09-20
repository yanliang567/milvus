package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/api/commonpb"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/api/milvuspb"
)

type dropPartitionTask struct {
	baseTaskV2
	Req      *milvuspb.DropPartitionRequest
	collMeta *model.Collection
}

func (t *dropPartitionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_DropPartition); err != nil {
		return err
	}
	if t.Req.GetPartitionName() == Params.CommonCfg.DefaultPartitionName {
		return fmt.Errorf("default partition cannot be deleted")
	}
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.GetTs())
	if err != nil {
		// Is this idempotent?
		return err
	}
	t.collMeta = collMeta
	return nil
}

func (t *dropPartitionTask) Execute(ctx context.Context) error {
	partID := common.InvalidPartitionID
	for _, partition := range t.collMeta.Partitions {
		if partition.PartitionName == t.Req.GetPartitionName() {
			partID = partition.PartitionID
			break
		}
	}
	if partID == common.InvalidPartitionID {
		log.Warn("drop an non-existent partition", zap.String("collection", t.Req.GetCollectionName()), zap.String("partition", t.Req.GetPartitionName()))
		// make dropping partition idempotent.
		return nil
	}

	redoTask := newBaseRedoTask()
	redoTask.AddSyncStep(&ExpireCacheStep{
		baseStep:        baseStep{core: t.core},
		collectionNames: []string{t.Req.GetCollectionName()},
		collectionID:    t.collMeta.CollectionID,
		ts:              t.GetTs(),
	})
	redoTask.AddSyncStep(&ChangePartitionStateStep{
		baseStep:     baseStep{core: t.core},
		collectionID: t.collMeta.CollectionID,
		partitionID:  partID,
		state:        pb.PartitionState_PartitionDropping,
		ts:           t.GetTs(),
	})

	// TODO: release partition when query coord is ready.
	redoTask.AddAsyncStep(&DeletePartitionDataStep{
		baseStep: baseStep{core: t.core},
		pchans:   t.collMeta.PhysicalChannelNames,
		partition: &model.Partition{
			PartitionID:   partID,
			PartitionName: t.Req.GetPartitionName(),
			CollectionID:  t.collMeta.CollectionID,
		},
		ts: t.GetTs(),
	})
	redoTask.AddAsyncStep(&RemovePartitionMetaStep{
		baseStep:     baseStep{core: t.core},
		collectionID: t.collMeta.CollectionID,
		partitionID:  partID,
		ts:           t.GetTs(),
	})

	return redoTask.Execute(ctx)
}
