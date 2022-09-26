package querycoord

import (
	"context"
	"errors"
	"fmt"
	"path"

	"github.com/milvus-io/milvus/internal/util/retry"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/api/schemapb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/funcutil"
)

type globalMetaBroker struct {
	ctx    context.Context
	cancel context.CancelFunc

	rootCoord  types.RootCoord
	dataCoord  types.DataCoord
	indexCoord types.IndexCoord

	cm storage.ChunkManager
}

func newGlobalMetaBroker(ctx context.Context, rootCoord types.RootCoord, dataCoord types.DataCoord, indexCoord types.IndexCoord, cm storage.ChunkManager) (*globalMetaBroker, error) {
	childCtx, cancel := context.WithCancel(ctx)
	parser := &globalMetaBroker{
		ctx:        childCtx,
		cancel:     cancel,
		rootCoord:  rootCoord,
		dataCoord:  dataCoord,
		indexCoord: indexCoord,
		cm:         cm,
	}
	return parser, nil
}

// invalidateCollectionMetaCache notifies RootCoord to remove all the collection meta cache with the specified collectionID in Proxies
func (broker *globalMetaBroker) invalidateCollectionMetaCache(ctx context.Context, collectionID UniqueID) error {
	ctx1, cancel1 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel1()
	req := &proxypb.InvalidateCollMetaCacheRequest{
		Base: &commonpb.MsgBase{
			MsgType: 0, // TODO: msg type?
		},
		CollectionID: collectionID,
	}

	res, err := broker.rootCoord.InvalidateCollectionMetaCache(ctx1, req)
	if err != nil {
		log.Error("InvalidateCollMetaCacheRequest failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return err
	}
	if res.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(res.Reason)
		log.Error("InvalidateCollMetaCacheRequest failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return err
	}
	log.Info("InvalidateCollMetaCacheRequest successfully", zap.Int64("collectionID", collectionID))

	return nil
}

func (broker *globalMetaBroker) describeCollection(ctx context.Context, collectionID UniqueID) (*schemapb.CollectionSchema, error) {
	req := &milvuspb.DescribeCollectionRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_DescribeCollection,
		},
		CollectionID: collectionID,
	}

	resp, err := broker.rootCoord.DescribeCollection(ctx, req)
	if err != nil {
		log.Warn("failed to describe collection schema", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}

	return resp.GetSchema(), nil
}

func (broker *globalMetaBroker) showPartitionIDs(ctx context.Context, collectionID UniqueID) ([]UniqueID, error) {
	ctx2, cancel2 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel2()
	showPartitionRequest := &milvuspb.ShowPartitionsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_ShowPartitions,
		},
		CollectionID: collectionID,
	}
	showPartitionResponse, err := broker.rootCoord.ShowPartitions(ctx2, showPartitionRequest)
	if err != nil {
		log.Error("showPartition failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}

	if showPartitionResponse.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(showPartitionResponse.Status.Reason)
		log.Error("showPartition failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}
	log.Info("show partition successfully", zap.Int64("collectionID", collectionID), zap.Int64s("partitionIDs", showPartitionResponse.PartitionIDs))

	return showPartitionResponse.PartitionIDs, nil
}

func (broker *globalMetaBroker) getRecoveryInfo(ctx context.Context, collectionID UniqueID, partitionID UniqueID) ([]*datapb.VchannelInfo, []*datapb.SegmentBinlogs, error) {
	ctx2, cancel2 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel2()
	getRecoveryInfoRequest := &datapb.GetRecoveryInfoRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_GetRecoveryInfo,
		},
		CollectionID: collectionID,
		PartitionID:  partitionID,
	}
	recoveryInfo, err := broker.dataCoord.GetRecoveryInfo(ctx2, getRecoveryInfoRequest)
	if err != nil {
		log.Error("get recovery info failed", zap.Int64("collectionID", collectionID), zap.Int64("partitionID", partitionID), zap.Error(err))
		return nil, nil, err
	}

	if recoveryInfo.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(recoveryInfo.Status.Reason)
		log.Error("get recovery info failed", zap.Int64("collectionID", collectionID), zap.Int64("partitionID", partitionID), zap.Error(err))
		return nil, nil, err
	}
	log.Info("get recovery info successfully",
		zap.Int64("collectionID", collectionID),
		zap.Int64("partitionID", partitionID),
		zap.Int("num channels", len(recoveryInfo.Channels)),
		zap.Int("num segments", len(recoveryInfo.Binlogs)))

	return recoveryInfo.Channels, recoveryInfo.Binlogs, nil
}

func (broker *globalMetaBroker) getIndexFilePaths(ctx context.Context, collID UniqueID, indexName string, segmentIDs []int64) (*indexpb.GetIndexInfoResponse, error) {
	indexFilePathRequest := &indexpb.GetIndexInfoRequest{
		CollectionID: collID,
		SegmentIDs:   segmentIDs,
		IndexName:    indexName,
	}

	ctx3, cancel3 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel3()
	pathResponse, err := broker.indexCoord.GetIndexInfos(ctx3, indexFilePathRequest)
	if err != nil {
		log.Error("get index info from indexCoord failed", zap.Int64s("segmentIDs", segmentIDs),
			zap.String("indexName", indexName), zap.Error(err))
		return nil, err
	}

	if pathResponse.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = fmt.Errorf("get index info from indexCoord failed, reason = %s", pathResponse.Status.Reason)
		log.Error(err.Error())
		return nil, err
	}
	log.Info("get index info from indexCoord successfully", zap.Int64s("segmentIDs", segmentIDs))

	return pathResponse, nil
}

// Better to let index params key appear in the file paths first.
func (broker *globalMetaBroker) loadIndexExtraInfo(ctx context.Context, fieldPathInfo *indexpb.IndexFilePathInfo) (*extraIndexInfo, error) {
	indexCodec := storage.NewIndexFileBinlogCodec()
	for _, indexFilePath := range fieldPathInfo.IndexFilePaths {
		// get index params when detecting indexParamPrefix
		if path.Base(indexFilePath) == storage.IndexParamsKey {
			content, err := broker.cm.MultiRead([]string{indexFilePath})
			if err != nil {
				return nil, err
			}

			if len(content) <= 0 {
				return nil, fmt.Errorf("failed to read index file binlog, path: %s", indexFilePath)
			}

			indexPiece := content[0]
			_, indexParams, indexName, _, err := indexCodec.Deserialize([]*storage.Blob{{Key: storage.IndexParamsKey, Value: indexPiece}})
			if err != nil {
				return nil, err
			}

			return &extraIndexInfo{
				indexName:   indexName,
				indexParams: funcutil.Map2KeyValuePair(indexParams),
			}, nil
		}
	}
	return nil, errors.New("failed to load index extra info")
}

// return: segment_id -> segment_index_infos
func (broker *globalMetaBroker) getFullIndexInfos(ctx context.Context, collectionID UniqueID, segmentIDs []UniqueID) (map[UniqueID][]*querypb.FieldIndexInfo, error) {
	resp, err := broker.getIndexFilePaths(ctx, collectionID, "", segmentIDs)
	if err != nil {
		log.Warn("failed to get index file paths", zap.Int64("collection", collectionID),
			zap.Int64s("segmentIDs", segmentIDs), zap.Error(err))
		return nil, err
	}

	ret := make(map[UniqueID][]*querypb.FieldIndexInfo)
	for _, segmentID := range segmentIDs {
		infos, ok := resp.GetSegmentInfo()[segmentID]
		if !ok {
			log.Warn("segment not found",
				zap.Int64("collection", collectionID),
				zap.Int64("segment", segmentID))
			return nil, fmt.Errorf("segment not found, collection: %d, segment: %d", collectionID, segmentID)
		}

		if _, ok := ret[segmentID]; !ok {
			ret[segmentID] = make([]*querypb.FieldIndexInfo, 0, len(infos.IndexInfos))
		}
		indexInfo := &querypb.FieldIndexInfo{
			EnableIndex: infos.EnableIndex,
		}

		if !infos.EnableIndex {
			ret[segmentID] = append(ret[segmentID], indexInfo)
			continue
		}

		for _, info := range infos.IndexInfos {
			//extraInfo, ok := infos.GetExtraIndexInfos()[info.IndexID]
			indexInfo := &querypb.FieldIndexInfo{
				FieldID:        info.FieldID,
				EnableIndex:    infos.EnableIndex,
				IndexName:      info.IndexName,
				IndexID:        info.IndexID,
				BuildID:        info.BuildID,
				IndexParams:    info.IndexParams,
				IndexFilePaths: info.IndexFilePaths,
				IndexSize:      int64(info.SerializedSize),
				IndexVersion:   info.IndexVersion,
			}

			ret[segmentID] = append(ret[segmentID], indexInfo)
		}
	}

	return ret, nil
}

func (broker *globalMetaBroker) getIndexInfo(ctx context.Context, collectionID UniqueID, segmentID UniqueID, schema *schemapb.CollectionSchema) ([]*querypb.FieldIndexInfo, error) {
	segmentIndexInfos, err := broker.getFullIndexInfos(ctx, collectionID, []UniqueID{segmentID})
	if err != nil {
		return nil, err
	}
	if infos, ok := segmentIndexInfos[segmentID]; ok {
		return infos, nil
	}
	return nil, fmt.Errorf("failed to get segment index infos, collection: %d, segment: %d", collectionID, segmentID)
}

func (broker *globalMetaBroker) generateSegmentLoadInfo(ctx context.Context,
	collectionID UniqueID,
	partitionID UniqueID,
	segmentBinlog *datapb.SegmentBinlogs,
	setIndex bool,
	schema *schemapb.CollectionSchema) *querypb.SegmentLoadInfo {
	segmentID := segmentBinlog.SegmentID
	segmentLoadInfo := &querypb.SegmentLoadInfo{
		SegmentID:     segmentID,
		PartitionID:   partitionID,
		CollectionID:  collectionID,
		BinlogPaths:   segmentBinlog.FieldBinlogs,
		NumOfRows:     segmentBinlog.NumOfRows,
		Statslogs:     segmentBinlog.Statslogs,
		Deltalogs:     segmentBinlog.Deltalogs,
		InsertChannel: segmentBinlog.InsertChannel,
	}
	if setIndex {
		// if index not exist, load binlog to query node
		indexInfo, err := broker.getIndexInfo(ctx, collectionID, segmentID, schema)
		if err == nil {
			segmentLoadInfo.IndexInfos = indexInfo
		}
		log.Warn("querycoord debug generateSegmentLoadInfo", zap.Any("indexInfo", indexInfo))
	}

	// set the estimate segment size to segmentLoadInfo
	segmentLoadInfo.SegmentSize = estimateSegmentSize(segmentLoadInfo)

	return segmentLoadInfo
}

func (broker *globalMetaBroker) getSegmentStates(ctx context.Context, segmentID UniqueID) (*datapb.SegmentStateInfo, error) {
	ctx2, cancel2 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel2()

	req := &datapb.GetSegmentStatesRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_GetSegmentState,
		},
		SegmentIDs: []UniqueID{segmentID},
	}
	resp, err := broker.dataCoord.GetSegmentStates(ctx2, req)
	if err != nil {
		log.Error("get segment states failed from dataCoord,", zap.Int64("segmentID", segmentID), zap.Error(err))
		return nil, err
	}

	if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(resp.Status.Reason)
		log.Error("get segment states failed from dataCoord,", zap.Int64("segmentID", segmentID), zap.Error(err))
		return nil, err
	}

	if len(resp.States) != 1 {
		err = fmt.Errorf("the length of segmentStates result should be 1, segmentID = %d", segmentID)
		log.Error(err.Error())
		return nil, err
	}

	return resp.States[0], nil
}

func (broker *globalMetaBroker) acquireSegmentsReferLock(ctx context.Context, taskID int64, segmentIDs []UniqueID) error {
	ctx, cancel := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel()
	acquireSegLockReq := &datapb.AcquireSegmentLockRequest{
		TaskID:     taskID,
		SegmentIDs: segmentIDs,
		NodeID:     Params.QueryCoordCfg.GetNodeID(),
	}
	status, err := broker.dataCoord.AcquireSegmentLock(ctx, acquireSegLockReq)
	if err != nil {
		log.Error("QueryCoord acquire the segment reference lock error", zap.Int64s("segIDs", segmentIDs),
			zap.Error(err))
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		log.Error("QueryCoord acquire the segment reference lock error", zap.Int64s("segIDs", segmentIDs),
			zap.String("failed reason", status.Reason))
		return fmt.Errorf(status.Reason)
	}

	return nil
}

func (broker *globalMetaBroker) releaseSegmentReferLock(ctx context.Context, taskID int64, segmentIDs []UniqueID) error {
	ctx, cancel := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel()

	releaseSegReferLockReq := &datapb.ReleaseSegmentLockRequest{
		TaskID:     taskID,
		NodeID:     Params.QueryCoordCfg.GetNodeID(),
		SegmentIDs: segmentIDs,
	}

	if err := retry.Do(ctx, func() error {
		status, err := broker.dataCoord.ReleaseSegmentLock(ctx, releaseSegReferLockReq)
		if err != nil {
			log.Error("QueryCoord release reference lock on segments failed", zap.Int64s("segmentIDs", segmentIDs),
				zap.Error(err))
			return err
		}

		if status.ErrorCode != commonpb.ErrorCode_Success {
			log.Error("QueryCoord release reference lock on segments failed", zap.Int64s("segmentIDs", segmentIDs),
				zap.String("failed reason", status.Reason))
			return errors.New(status.Reason)
		}
		return nil
	}, retry.Attempts(100)); err != nil {
		return err
	}

	return nil
}

// getDataSegmentInfosByIDs return the SegmentInfo details according to the given ids through RPC to datacoord
func (broker *globalMetaBroker) getDataSegmentInfosByIDs(ctx context.Context, segmentIds []int64) ([]*datapb.SegmentInfo, error) {
	var segmentInfos []*datapb.SegmentInfo
	infoResp, err := broker.dataCoord.GetSegmentInfo(ctx, &datapb.GetSegmentInfoRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_SegmentInfo,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  Params.ProxyCfg.GetNodeID(),
		},
		SegmentIDs:       segmentIds,
		IncludeUnHealthy: true,
	})
	if err != nil {
		log.Error("Fail to get datapb.SegmentInfo by ids from datacoord", zap.Error(err))
		return nil, err
	}
	if infoResp.GetStatus().ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(infoResp.GetStatus().Reason)
		log.Error("Fail to get datapb.SegmentInfo by ids from datacoord", zap.Error(err))
		return nil, err
	}
	segmentInfos = infoResp.Infos
	return segmentInfos, nil
}
