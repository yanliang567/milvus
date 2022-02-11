package querycoord

import (
	"context"
	"errors"
	"fmt"
	"path"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	minioKV "github.com/milvus-io/milvus/internal/kv/minio"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
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

	dataKV kv.DataKV
}

func newGlobalMetaBroker(ctx context.Context, rootCoord types.RootCoord, dataCoord types.DataCoord, indexCoord types.IndexCoord) (*globalMetaBroker, error) {
	childCtx, cancel := context.WithCancel(ctx)
	parser := &globalMetaBroker{
		ctx:        childCtx,
		cancel:     cancel,
		rootCoord:  rootCoord,
		dataCoord:  dataCoord,
		indexCoord: indexCoord,
	}
	option := &minioKV.Option{
		Address:           Params.MinioCfg.Address,
		AccessKeyID:       Params.MinioCfg.AccessKeyID,
		SecretAccessKeyID: Params.MinioCfg.SecretAccessKey,
		UseSSL:            Params.MinioCfg.UseSSL,
		CreateBucket:      true,
		BucketName:        Params.MinioCfg.BucketName,
	}

	dataKV, err := minioKV.NewMinIOKV(childCtx, option)
	if err != nil {
		return nil, err
	}
	parser.dataKV = dataKV
	return parser, nil
}

func (broker *globalMetaBroker) releaseDQLMessageStream(ctx context.Context, collectionID UniqueID) error {
	ctx2, cancel2 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel2()
	releaseDQLMessageStreamReq := &proxypb.ReleaseDQLMessageStreamRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_RemoveQueryChannels,
		},
		CollectionID: collectionID,
	}
	res, err := broker.rootCoord.ReleaseDQLMessageStream(ctx2, releaseDQLMessageStreamReq)
	if err != nil {
		log.Error("releaseDQLMessageStream occur error", zap.Int64("collectionID", collectionID), zap.Error(err))
		return err
	}
	if res.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(res.Reason)
		log.Error("releaseDQLMessageStream occur error", zap.Int64("collectionID", collectionID), zap.Error(err))
		return err
	}
	log.Debug("releaseDQLMessageStream successfully", zap.Int64("collectionID", collectionID))

	return nil
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
	log.Debug("show partition successfully", zap.Int64("collectionID", collectionID), zap.Int64s("partitionIDs", showPartitionResponse.PartitionIDs))

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
	log.Debug("get recovery info successfully",
		zap.Int64("collectionID", collectionID),
		zap.Int64("partitionID", partitionID),
		zap.Int("num channels", len(recoveryInfo.Channels)),
		zap.Int("num segments", len(recoveryInfo.Binlogs)))

	return recoveryInfo.Channels, recoveryInfo.Binlogs, nil
}

func (broker *globalMetaBroker) getIndexBuildID(ctx context.Context, collectionID UniqueID, segmentID UniqueID) (bool, int64, error) {
	req := &milvuspb.DescribeSegmentRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_DescribeSegment,
		},
		CollectionID: collectionID,
		SegmentID:    segmentID,
	}
	ctx2, cancel2 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel2()
	response, err := broker.rootCoord.DescribeSegment(ctx2, req)
	if err != nil {
		log.Error("describe segment from rootCoord failed",
			zap.Int64("collectionID", collectionID),
			zap.Int64("segmentID", segmentID),
			zap.Error(err))
		return false, 0, err
	}
	if response.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = errors.New(response.Status.Reason)
		log.Error("describe segment from rootCoord failed",
			zap.Int64("collectionID", collectionID),
			zap.Int64("segmentID", segmentID),
			zap.Error(err))
		return false, 0, err
	}

	if !response.EnableIndex {
		log.Debug("describe segment from rootCoord successfully",
			zap.Int64("collectionID", collectionID),
			zap.Int64("segmentID", segmentID),
			zap.Bool("enableIndex", false))
		return false, 0, nil
	}

	log.Debug("describe segment from rootCoord successfully",
		zap.Int64("collectionID", collectionID),
		zap.Int64("segmentID", segmentID),
		zap.Bool("enableIndex", true),
		zap.Int64("buildID", response.BuildID))
	return true, response.BuildID, nil
}

func (broker *globalMetaBroker) getIndexFilePaths(ctx context.Context, buildID int64) ([]*indexpb.IndexFilePathInfo, error) {
	indexFilePathRequest := &indexpb.GetIndexFilePathsRequest{
		IndexBuildIDs: []UniqueID{buildID},
	}
	ctx3, cancel3 := context.WithTimeout(ctx, timeoutForRPC)
	defer cancel3()
	pathResponse, err := broker.indexCoord.GetIndexFilePaths(ctx3, indexFilePathRequest)
	if err != nil {
		log.Error("get index info from indexCoord failed",
			zap.Int64("indexBuildID", buildID),
			zap.Error(err))
		return nil, err
	}

	if pathResponse.Status.ErrorCode != commonpb.ErrorCode_Success {
		err = fmt.Errorf("get index info from indexCoord failed, buildID = %d, reason = %s", buildID, pathResponse.Status.Reason)
		log.Error(err.Error())
		return nil, err
	}
	log.Debug("get index info from indexCoord successfully", zap.Int64("buildID", buildID))

	return pathResponse.FilePaths, nil
}

func (broker *globalMetaBroker) parseIndexInfo(ctx context.Context, segmentID UniqueID, indexInfo *querypb.VecFieldIndexInfo) error {
	if !indexInfo.EnableIndex {
		log.Debug(fmt.Sprintf("fieldID %d of segment %d don't has index", indexInfo.FieldID, segmentID))
		return nil
	}
	buildID := indexInfo.BuildID
	indexFilePathInfos, err := broker.getIndexFilePaths(ctx, buildID)
	if err != nil {
		return err
	}

	if len(indexFilePathInfos) != 1 {
		err = fmt.Errorf("illegal index file paths, there should be only one vector column,  segmentID = %d, fieldID = %d, buildID = %d", segmentID, indexInfo.FieldID, buildID)
		log.Error(err.Error())
		return err
	}

	fieldPathInfo := indexFilePathInfos[0]
	if len(fieldPathInfo.IndexFilePaths) == 0 {
		err = fmt.Errorf("empty index paths, segmentID = %d, fieldID = %d, buildID = %d", segmentID, indexInfo.FieldID, buildID)
		log.Error(err.Error())
		return err
	}

	indexInfo.IndexFilePaths = fieldPathInfo.IndexFilePaths
	indexInfo.IndexSize = int64(fieldPathInfo.SerializedSize)

	log.Debug("get indexFilePath info from indexCoord success", zap.Int64("segmentID", segmentID), zap.Int64("fieldID", indexInfo.FieldID), zap.Int64("buildID", buildID), zap.Strings("indexPaths", fieldPathInfo.IndexFilePaths))

	indexCodec := storage.NewIndexFileBinlogCodec()
	for _, indexFilePath := range fieldPathInfo.IndexFilePaths {
		// get index params when detecting indexParamPrefix
		if path.Base(indexFilePath) == storage.IndexParamsKey {
			indexPiece, err := broker.dataKV.Load(indexFilePath)
			if err != nil {
				log.Error("load index params file failed",
					zap.Int64("segmentID", segmentID),
					zap.Int64("fieldID", indexInfo.FieldID),
					zap.Int64("indexBuildID", buildID),
					zap.String("index params filePath", indexFilePath),
					zap.Error(err))
				return err
			}
			_, indexParams, indexName, indexID, err := indexCodec.Deserialize([]*storage.Blob{{Key: storage.IndexParamsKey, Value: []byte(indexPiece)}})
			if err != nil {
				log.Error("deserialize index params file failed",
					zap.Int64("segmentID", segmentID),
					zap.Int64("fieldID", indexInfo.FieldID),
					zap.Int64("indexBuildID", buildID),
					zap.String("index params filePath", indexFilePath),
					zap.Error(err))
				return err
			}
			if len(indexParams) <= 0 {
				err = fmt.Errorf("cannot find index param, segmentID = %d, fieldID = %d, buildID = %d, indexFilePath = %s", segmentID, indexInfo.FieldID, buildID, indexFilePath)
				log.Error(err.Error())
				return err
			}
			indexInfo.IndexName = indexName
			indexInfo.IndexID = indexID
			indexInfo.IndexParams = funcutil.Map2KeyValuePair(indexParams)
			break
		}
	}

	if len(indexInfo.IndexParams) == 0 {
		err = fmt.Errorf("no index params in Index file, segmentID = %d, fieldID = %d, buildID = %d, indexPaths = %v", segmentID, indexInfo.FieldID, buildID, fieldPathInfo.IndexFilePaths)
		log.Error(err.Error())
		return err
	}

	log.Debug("set index info  success", zap.Int64("segmentID", segmentID), zap.Int64("fieldID", indexInfo.FieldID), zap.Int64("buildID", buildID))

	return nil
}

func (broker *globalMetaBroker) getIndexInfo(ctx context.Context, collectionID UniqueID, segmentID UniqueID, schema *schemapb.CollectionSchema) ([]*querypb.VecFieldIndexInfo, error) {
	// TODO:: collection has multi vec field, and build index for every vec field, get indexInfo by fieldID
	// Currently, each collection can only have one vector field
	vecFieldIDs := funcutil.GetVecFieldIDs(schema)
	if len(vecFieldIDs) != 1 {
		err := fmt.Errorf("collection %d has multi vec field, num of vec fields = %d", collectionID, len(vecFieldIDs))
		log.Error("get index info failed",
			zap.Int64("collectionID", collectionID),
			zap.Int64("segmentID", segmentID),
			zap.Error(err))
		return nil, err
	}
	indexInfo := &querypb.VecFieldIndexInfo{
		FieldID: vecFieldIDs[0],
	}
	// check the buildID of the segment's index whether exist on rootCoord
	enableIndex, buildID, err := broker.getIndexBuildID(ctx, collectionID, segmentID)
	if err != nil {
		return nil, err
	}

	// if the segment.EnableIndex == false, then load the segment immediately
	if !enableIndex {
		indexInfo.EnableIndex = false
	} else {
		indexInfo.BuildID = buildID
		indexInfo.EnableIndex = true
		err = broker.parseIndexInfo(ctx, segmentID, indexInfo)
		if err != nil {
			return nil, err
		}
	}
	log.Debug("get index info success", zap.Int64("collectionID", collectionID), zap.Int64("segmentID", segmentID), zap.Bool("enableIndex", enableIndex))

	return []*querypb.VecFieldIndexInfo{indexInfo}, nil
}

func (broker *globalMetaBroker) generateSegmentLoadInfo(ctx context.Context,
	collectionID UniqueID,
	partitionID UniqueID,
	segmentBinlog *datapb.SegmentBinlogs,
	setIndex bool,
	schema *schemapb.CollectionSchema) *querypb.SegmentLoadInfo {
	segmentID := segmentBinlog.SegmentID
	segmentLoadInfo := &querypb.SegmentLoadInfo{
		SegmentID:    segmentID,
		PartitionID:  partitionID,
		CollectionID: collectionID,
		BinlogPaths:  segmentBinlog.FieldBinlogs,
		NumOfRows:    segmentBinlog.NumOfRows,
		Statslogs:    segmentBinlog.Statslogs,
		Deltalogs:    segmentBinlog.Deltalogs,
	}
	if setIndex {
		// if index not exist, load binlog to query node
		indexInfo, err := broker.getIndexInfo(ctx, collectionID, segmentID, schema)
		if err == nil {
			segmentLoadInfo.IndexInfos = indexInfo
		}
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
