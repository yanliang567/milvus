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
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type queryShard struct {
	ctx    context.Context
	cancel context.CancelFunc

	collectionID UniqueID
	channel      Channel
	deltaChannel Channel
	replicaID    int64

	clusterService *ShardClusterService
	historical     *historical
	streaming      *streaming

	dmTSafeWatcher    *tSafeWatcher
	deltaTSafeWatcher *tSafeWatcher
	watcherCond       *sync.Cond
	serviceDmTs       atomic.Uint64
	serviceDeltaTs    atomic.Uint64
	startTickerOnce   sync.Once
	ticker            *time.Ticker // timed ticker for trigger timeout check

	localChunkManager  storage.ChunkManager
	remoteChunkManager storage.ChunkManager
	vectorChunkManager *storage.VectorChunkManager
	localCacheEnabled  bool
	localCacheSize     int64
}

func newQueryShard(
	ctx context.Context,
	collectionID UniqueID,
	channel Channel,
	replicaID int64,
	clusterService *ShardClusterService,
	historical *historical,
	streaming *streaming,
	localChunkManager storage.ChunkManager,
	remoteChunkManager storage.ChunkManager,
	localCacheEnabled bool,
) *queryShard {
	ctx, cancel := context.WithCancel(ctx)
	qs := &queryShard{
		ctx:                ctx,
		cancel:             cancel,
		collectionID:       collectionID,
		channel:            channel,
		replicaID:          replicaID,
		clusterService:     clusterService,
		historical:         historical,
		streaming:          streaming,
		localChunkManager:  localChunkManager,
		remoteChunkManager: remoteChunkManager,
		localCacheEnabled:  localCacheEnabled,
		localCacheSize:     Params.QueryNodeCfg.CacheMemoryLimit,

		watcherCond: sync.NewCond(&sync.Mutex{}),
	}
	deltaChannel, err := funcutil.ConvertChannelName(channel, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta)
	if err != nil {
		log.Warn("failed to convert dm channel to delta", zap.String("channel", channel), zap.Error(err))
	}
	qs.deltaChannel = deltaChannel

	return qs
}

// Close cleans query shard
func (q *queryShard) Close() {
	q.cancel()
}

func (q *queryShard) watchDMLTSafe() error {
	q.dmTSafeWatcher = newTSafeWatcher()
	err := q.streaming.tSafeReplica.registerTSafeWatcher(q.channel, q.dmTSafeWatcher)
	if err != nil {
		log.Warn("failed to register dml tsafe watcher", zap.String("channel", q.channel), zap.Error(err))
		return err
	}
	go q.watchTs(q.dmTSafeWatcher.watcherChan(), q.dmTSafeWatcher.closeCh, tsTypeDML)

	q.startTsTicker()
	return nil
}

func (q *queryShard) watchDeltaTSafe() error {
	q.deltaTSafeWatcher = newTSafeWatcher()
	err := q.streaming.tSafeReplica.registerTSafeWatcher(q.deltaChannel, q.deltaTSafeWatcher)
	if err != nil {
		log.Warn("failed to register delta tsafe watcher", zap.String("channel", q.deltaChannel), zap.Error(err))
		return err
	}

	go q.watchTs(q.deltaTSafeWatcher.watcherChan(), q.deltaTSafeWatcher.closeCh, tsTypeDelta)
	q.startTsTicker()

	return nil
}

func (q *queryShard) startTsTicker() {
	q.startTickerOnce.Do(func() {
		go func() {
			q.ticker = time.NewTicker(time.Millisecond * 50) // check timeout every 50 milliseconds, need not to be to frequent
			defer q.ticker.Stop()
			for {
				select {
				case <-q.ticker.C:
					q.watcherCond.L.Lock()
					q.watcherCond.Broadcast()
					q.watcherCond.L.Unlock()
				case <-q.ctx.Done():
					return
				}
			}
		}()
	})
}

type tsType int32

const (
	tsTypeDML   tsType = 1
	tsTypeDelta tsType = 2
)

func (tp tsType) String() string {
	switch tp {
	case tsTypeDML:
		return "DML tSafe"
	case tsTypeDelta:
		return "Delta tSafe"
	}
	return ""
}

func (q *queryShard) watchTs(channel <-chan bool, closeCh <-chan struct{}, tp tsType) {
	for {
		select {
		case <-q.ctx.Done():
			log.Debug("stop queryShard watcher due to ctx done", zap.Int64("collectionID", q.collectionID), zap.String("vChannel", q.channel))
			return
		case <-closeCh:
			log.Debug("stop queryShard watcher due to watcher closed", zap.Int64("collectionID", q.collectionID), zap.String("vChannel", q.channel))
			return
		case _, ok := <-channel:
			if !ok {
				log.Warn("tsafe watcher channel closed", zap.Int64("collectionID", q.collectionID), zap.String("vChannel", q.channel))
				return
			}

			ts, err := q.getNewTSafe(tp)
			if err == nil {
				q.watcherCond.L.Lock()
				q.setServiceableTime(ts, tp)
				q.watcherCond.Broadcast()
				q.watcherCond.L.Unlock()
			}
		}
	}
}

func (q *queryShard) getNewTSafe(tp tsType) (Timestamp, error) {
	var channel string
	switch tp {
	case tsTypeDML:
		channel = q.channel
	case tsTypeDelta:
		channel = q.deltaChannel
	default:
		return 0, errors.New("invalid ts type")
	}
	t := Timestamp(math.MaxInt64)
	ts, err := q.streaming.tSafeReplica.getTSafe(channel)
	if err != nil {
		return 0, err
	}
	if ts <= t {
		t = ts
	}
	return t, nil
}

func (q *queryShard) waitUntilServiceable(ctx context.Context, guaranteeTs Timestamp, tp tsType) {
	q.watcherCond.L.Lock()
	defer q.watcherCond.L.Unlock()
	st := q.getServiceableTime(tp)
	for guaranteeTs > st {
		log.Debug("serviceable ts before guarantee ts", zap.Uint64("serviceable ts", st), zap.Uint64("guarantee ts", guaranteeTs), zap.String("channel", q.channel))
		q.watcherCond.Wait()
		if err := ctx.Err(); err != nil {
			log.Warn("waitUntialServiceable timeout", zap.Uint64("serviceable ts", st), zap.Uint64("guarantee ts", guaranteeTs), zap.String("channel", q.channel))
			return
		}
		st = q.getServiceableTime(tp)
	}
	log.Debug("wait serviceable ts done", zap.String("tsType", tp.String()), zap.Uint64("guarantee ts", guaranteeTs), zap.Uint64("serviceable ts", st), zap.String("channel", q.channel))
}

func (q *queryShard) getServiceableTime(tp tsType) Timestamp {
	gracefulTimeInMilliSecond := Params.QueryNodeCfg.GracefulTime
	gracefulTime := typeutil.ZeroTimestamp
	if gracefulTimeInMilliSecond > 0 {
		gracefulTime = tsoutil.ComposeTS(gracefulTimeInMilliSecond, 0)
	}
	var serviceTs Timestamp
	switch tp {
	case tsTypeDML: // use min value of dml & delta
		serviceTs = q.serviceDmTs.Load()
	case tsTypeDelta: // check delta ts only
		serviceTs = q.serviceDeltaTs.Load()
	}
	return serviceTs + gracefulTime
}

func (q *queryShard) setServiceableTime(t Timestamp, tp tsType) {
	switch tp {
	case tsTypeDML:
		ts := q.serviceDmTs.Load()
		if t < ts {
			return
		}
		for !q.serviceDmTs.CAS(ts, t) {
			ts = q.serviceDmTs.Load()
			if t < ts {
				return
			}
		}
	case tsTypeDelta:
		ts := q.serviceDeltaTs.Load()
		if t < ts {
			return
		}
		for !q.serviceDeltaTs.CAS(ts, t) {
			ts = q.serviceDeltaTs.Load()
			if t < ts {
				return
			}
		}
	}
}

func (q *queryShard) search(ctx context.Context, req *querypb.SearchRequest) (*internalpb.SearchResults, error) {
	collectionID := req.Req.CollectionID
	segmentIDs := req.SegmentIDs
	timestamp := req.Req.TravelTimestamp

	// check ctx timeout
	if !funcutil.CheckCtxValid(ctx) {
		return nil, errors.New("search context timeout")
	}

	// check if collection has been released
	collection, err := q.historical.replica.getCollectionByID(collectionID)
	if err != nil {
		return nil, err
	}
	if req.GetReq().GetGuaranteeTimestamp() >= collection.getReleaseTime() {
		log.Warn("collection release before search", zap.Int64("collectionID", collectionID))
		return nil, fmt.Errorf("retrieve failed, collection has been released, collectionID = %d", collectionID)
	}

	// deserialize query plan

	var plan *SearchPlan
	if req.Req.GetDslType() == commonpb.DslType_BoolExprV1 {
		expr := req.Req.SerializedExprPlan
		plan, err = createSearchPlanByExpr(collection, expr)
		if err != nil {
			return nil, err
		}
	} else {
		dsl := req.Req.Dsl
		plan, err = createSearchPlan(collection, dsl)
		if err != nil {
			return nil, err
		}
	}
	defer plan.delete()

	schemaHelper, err := typeutil.CreateSchemaHelper(collection.schema)
	if err != nil {
		return nil, err
	}

	// validate top-k
	topK := plan.getTopK()
	if topK <= 0 || topK >= 16385 {
		return nil, fmt.Errorf("limit should be in range [1, 16385], but got %d", topK)
	}

	// parse plan to search request
	searchReq, err := parseSearchRequest(plan, req.Req.PlaceholderGroup)
	if err != nil {
		return nil, err
	}
	defer searchReq.delete()
	queryNum := searchReq.getNumOfQuery()
	searchRequests := []*searchRequest{searchReq}

	if len(segmentIDs) == 0 {
		// segmentIDs not specified, searching as shard leader
		return q.searchLeader(ctx, req, searchRequests, collectionID, schemaHelper, plan, topK, queryNum, timestamp)
	}

	// segmentIDs specified search as shard follower
	return q.searchFollower(ctx, req, searchRequests, collectionID, schemaHelper, plan, topK, queryNum, timestamp)
}

func (q *queryShard) searchLeader(ctx context.Context, req *querypb.SearchRequest, searchRequests []*searchRequest, collectionID UniqueID,
	schemaHelper *typeutil.SchemaHelper, plan *SearchPlan, topK int64, queryNum int64, timestamp Timestamp) (*internalpb.SearchResults, error) {
	q.streaming.replica.queryRLock()
	defer q.streaming.replica.queryRUnlock()
	cluster, ok := q.clusterService.getShardCluster(req.GetDmlChannel())
	if !ok {
		return nil, fmt.Errorf("channel %s leader is not here", req.GetDmlChannel())
	}

	searchCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var results []*internalpb.SearchResults
	var streamingResults []*SearchResult
	var err error
	var mut sync.Mutex
	var wg sync.WaitGroup

	wg.Add(2) // search cluster and search streaming

	go func() {
		defer wg.Done()
		// shard leader dispatches request to its shard cluster
		cResults, cErr := cluster.Search(searchCtx, req)
		mut.Lock()
		defer mut.Unlock()
		if cErr != nil {
			log.Warn("search cluster failed", zap.Int64("collectionID", q.collectionID), zap.Error(cErr))
			err = cErr
			cancel()
			return
		}

		results = cResults
	}()

	go func() {
		defer wg.Done()
		// hold request until guarantee timestamp >= service timestamp
		guaranteeTs := req.GetReq().GetGuaranteeTimestamp()
		q.waitUntilServiceable(ctx, guaranteeTs, tsTypeDML)
		// shard leader queries its own streaming data
		// TODO add context
		sResults, _, _, sErr := q.streaming.search(searchRequests, collectionID, req.Req.PartitionIDs, req.DmlChannel, plan, timestamp)
		mut.Lock()
		defer mut.Unlock()
		if sErr != nil {
			log.Warn("failed to search streaming data", zap.Int64("collectionID", q.collectionID), zap.Error(sErr))
			err = sErr
			cancel()
			return
		}
		streamingResults = sResults
	}()

	wg.Wait()
	if err != nil {
		return nil, err
	}

	defer deleteSearchResults(streamingResults)

	results = append(results, &internalpb.SearchResults{
		Status:         &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		MetricType:     plan.getMetricType(),
		NumQueries:     queryNum,
		TopK:           topK,
		SlicedBlob:     nil,
		SlicedOffset:   1,
		SlicedNumCount: 1,
	})

	if len(streamingResults) > 0 {
		// reduce search results
		numSegment := int64(len(streamingResults))
		err = reduceSearchResultsAndFillData(plan, streamingResults, numSegment)
		if err != nil {
			return nil, err
		}

		nq := searchRequests[0].getNumOfQuery()
		nqOfReqs := []int64{nq}
		nqPerSlice := nq
		reqSlices, err := getReqSlices(nqOfReqs, nqPerSlice)
		if err != nil {
			log.Warn("getReqSlices for streaming results error", zap.Error(err))
			return nil, err
		}

		blobs, err := marshal(collectionID, 0, streamingResults, int(numSegment), reqSlices)
		defer deleteSearchResultDataBlobs(blobs)
		if err != nil {
			log.Warn("marshal for streaming results error", zap.Error(err))
			return nil, err
		}

		// assume only one blob will be sent back
		blob, err := getSearchResultDataBlob(blobs, 0)
		if err != nil {
			log.Warn("getSearchResultDataBlob for streaming results error", zap.Error(err))
		}

		results[len(results)-1].SlicedBlob = blob
	}

	// reduce shard search results: unmarshal -> reduce -> marshal
	log.Debug("shard leader get search results", zap.Int("numbers", len(results)))
	searchResultData, err := decodeSearchResults(results)
	if err != nil {
		log.Warn("shard leader decode search results errors", zap.Error(err))
		return nil, err
	}
	log.Debug("shard leader get valid search results", zap.Int("numbers", len(searchResultData)))

	for i, sData := range searchResultData {
		log.Debug("reduceSearchResultData",
			zap.Int("result No.", i),
			zap.Int64("nq", sData.NumQueries),
			zap.Int64("topk", sData.TopK),
			zap.String("ids", sData.Ids.String()),
			zap.Any("len(FieldsData)", len(sData.FieldsData)))
	}

	reducedResultData, err := reduceSearchResultData(searchResultData, queryNum, plan.getTopK(), plan.getMetricType())
	if err != nil {
		log.Warn("shard leader reduce errors", zap.Error(err))
		return nil, err
	}
	searchResults, err := encodeSearchResultData(reducedResultData, queryNum, plan.getTopK(), plan.getMetricType())
	if err != nil {
		log.Warn("shard leader encode search result errors", zap.Error(err))
		return nil, err
	}
	if searchResults.SlicedBlob == nil {
		log.Debug("shard leader send nil results to proxy",
			zap.String("shard", q.channel))
	} else {
		log.Debug("shard leader send non-nil results to proxy",
			zap.String("shard", q.channel),
			zap.String("ids", reducedResultData.Ids.String()))
		// printSearchResultData(reducedResultData, q.channel)
	}
	return searchResults, nil
}

func (q *queryShard) searchFollower(ctx context.Context, req *querypb.SearchRequest, searchRequests []*searchRequest, collectionID UniqueID,
	schemaHelper *typeutil.SchemaHelper, plan *SearchPlan, topK int64, queryNum int64, timestamp Timestamp) (*internalpb.SearchResults, error) {
	q.historical.replica.queryRLock()
	defer q.historical.replica.queryRUnlock()
	segmentIDs := req.GetSegmentIDs()
	// hold request until guarantee timestamp >= service timestamp
	guaranteeTs := req.GetReq().GetGuaranteeTimestamp()
	q.waitUntilServiceable(ctx, guaranteeTs, tsTypeDelta)
	// search each segments by segment IDs in request
	historicalResults, _, err := q.historical.searchSegments(segmentIDs, searchRequests, plan, timestamp)
	if err != nil {
		return nil, err
	}
	defer deleteSearchResults(historicalResults)

	// reduce search results
	numSegment := int64(len(historicalResults))
	err = reduceSearchResultsAndFillData(plan, historicalResults, numSegment)
	if err != nil {
		return nil, err
	}

	nq := searchRequests[0].getNumOfQuery()
	nqOfReqs := []int64{nq}
	nqPerSlice := nq
	reqSlices, err := getReqSlices(nqOfReqs, nqPerSlice)
	if err != nil {
		log.Warn("getReqSlices for historical results error", zap.Error(err))
		return nil, err
	}

	blobs, err := marshal(collectionID, 0, historicalResults, int(numSegment), reqSlices)
	defer deleteSearchResultDataBlobs(blobs)
	if err != nil {
		log.Warn("marshal for historical results error", zap.Error(err))
		return nil, err
	}

	// assume only one blob will be sent back
	blob, err := getSearchResultDataBlob(blobs, 0)
	if err != nil {
		log.Warn("getSearchResultDataBlob for historical results error", zap.Error(err))
	}
	bs := make([]byte, len(blob))
	copy(bs, blob)

	resp := &internalpb.SearchResults{
		Status:         &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		MetricType:     plan.getMetricType(),
		NumQueries:     queryNum,
		TopK:           topK,
		SlicedBlob:     bs,
		SlicedOffset:   1,
		SlicedNumCount: 1,
	}
	log.Debug("shard follower send search result to leader")
	return resp, nil
}

func reduceSearchResultData(searchResultData []*schemapb.SearchResultData, nq int64, topk int64, metricType string) (*schemapb.SearchResultData, error) {
	if len(searchResultData) == 0 {
		return &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topk,
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     make([]float32, 0),
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: make([]int64, 0),
					},
				},
			},
			Topks: make([]int64, 0),
		}, nil
	}
	ret := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: make([]*schemapb.FieldData, len(searchResultData[0].FieldsData)),
		Scores:     make([]float32, 0),
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: make([]int64, 0),
				},
			},
		},
		Topks: make([]int64, 0),
	}

	var skipDupCnt int64
	var dummyCnt int64
	// var realTopK int64 = -1
	for i := int64(0); i < nq; i++ {
		offsets := make([]int64, len(searchResultData))

		var idSet = make(map[int64]struct{})
		var j int64
		for j = 0; j < topk; {
			sel := selectSearchResultData(searchResultData, offsets, topk, i)
			if sel == -1 {
				break
			}
			idx := i*topk + offsets[sel]

			id := searchResultData[sel].Ids.GetIntId().Data[idx]
			score := searchResultData[sel].Scores[idx]
			// ignore invalid search result
			if id == -1 {
				continue
			}

			// remove duplicates
			if _, ok := idSet[id]; !ok {
				typeutil.AppendFieldData(ret.FieldsData, searchResultData[sel].FieldsData, idx)
				ret.Ids.GetIntId().Data = append(ret.Ids.GetIntId().Data, id)
				ret.Scores = append(ret.Scores, score)
				idSet[id] = struct{}{}
				j++
			} else {
				// skip entity with same id
				skipDupCnt++
			}
			offsets[sel]++
		}
		// add empty data
		for j < topk {
			typeutil.AppendFieldData(ret.FieldsData, searchResultData[0].FieldsData, 0)
			ret.Ids.GetIntId().Data = append(ret.Ids.GetIntId().Data, -1)
			ret.Scores = append(ret.Scores, -1*float32(math.MaxFloat32))
			j++
			dummyCnt++
		}

		// if realTopK != -1 && realTopK != j {
		// 	log.Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
		// 	// return nil, errors.New("the length (topk) between all result of query is different")
		// }
		// realTopK = j
		// ret.Topks = append(ret.Topks, realTopK)
	}
	log.Debug("skip duplicated search result", zap.Int64("count", skipDupCnt))
	log.Debug("add dummy data in search result", zap.Int64("count", dummyCnt))
	// ret.TopK = realTopK

	// if !distance.PositivelyRelated(metricType) {
	// 	for k := range ret.Scores {
	// 		ret.Scores[k] *= -1
	// 	}
	// }

	return ret, nil
}

func selectSearchResultData(dataArray []*schemapb.SearchResultData, offsets []int64, topk int64, qi int64) int {
	sel := -1
	maxDistance := -1 * float32(math.MaxFloat32)
	for i, offset := range offsets { // query num, the number of ways to merge
		if offset >= topk {
			continue
		}
		idx := qi*topk + offset
		id := dataArray[i].Ids.GetIntId().Data[idx]
		if id != -1 {
			distance := dataArray[i].Scores[idx]
			if distance > maxDistance {
				sel = i
				maxDistance = distance
			}
		}
	}
	return sel
}

func decodeSearchResults(searchResults []*internalpb.SearchResults) ([]*schemapb.SearchResultData, error) {
	results := make([]*schemapb.SearchResultData, 0)
	for _, partialSearchResult := range searchResults {
		if partialSearchResult.SlicedBlob == nil {
			continue
		}

		var partialResultData schemapb.SearchResultData
		err := proto.Unmarshal(partialSearchResult.SlicedBlob, &partialResultData)
		if err != nil {
			return nil, err
		}

		results = append(results, &partialResultData)
	}
	return results, nil
}

func encodeSearchResultData(searchResultData *schemapb.SearchResultData, nq int64, topk int64, metricType string) (searchResults *internalpb.SearchResults, err error) {
	searchResults = &internalpb.SearchResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		NumQueries: nq,
		TopK:       topk,
		MetricType: metricType,
		SlicedBlob: nil,
	}
	slicedBlob, err := proto.Marshal(searchResultData)
	if err != nil {
		return nil, err
	}
	if searchResultData != nil && searchResultData.Ids != nil && len(searchResultData.Ids.GetIntId().Data) != 0 {
		searchResults.SlicedBlob = slicedBlob
	}
	return
}

func (q *queryShard) query(ctx context.Context, req *querypb.QueryRequest) (*internalpb.RetrieveResults, error) {
	collectionID := req.Req.CollectionID
	segmentIDs := req.SegmentIDs
	partitionIDs := req.Req.PartitionIDs
	expr := req.Req.SerializedExprPlan
	timestamp := req.Req.TravelTimestamp

	// check ctx timeout
	if !funcutil.CheckCtxValid(ctx) {
		return nil, errors.New("search context timeout")
	}

	// check if collection has been released
	collection, err := q.streaming.replica.getCollectionByID(collectionID)
	if err != nil {
		return nil, err
	}

	if req.GetReq().GetGuaranteeTimestamp() >= collection.getReleaseTime() {
		log.Warn("collection release before query", zap.Int64("collectionID", collectionID))
		return nil, fmt.Errorf("retrieve failed, collection has been released, collectionID = %d", collectionID)
	}
	// deserialize query plan
	plan, err := createRetrievePlanByExpr(collection, expr, timestamp)
	if err != nil {
		return nil, err
	}
	defer plan.delete()

	// TODO: init vector chunk manager at most once
	if q.vectorChunkManager == nil {
		if q.localChunkManager == nil {
			return nil, fmt.Errorf("can not create vector chunk manager for local chunk manager is nil")
		}
		if q.remoteChunkManager == nil {
			return nil, fmt.Errorf("can not create vector chunk manager for remote chunk manager is nil")
		}
		q.vectorChunkManager, err = storage.NewVectorChunkManager(q.localChunkManager, q.remoteChunkManager,
			&etcdpb.CollectionMeta{
				ID:     collection.id,
				Schema: collection.schema,
			}, q.localCacheSize, q.localCacheEnabled)
		if err != nil {
			return nil, err
		}
	}

	// check if shard leader b.c only leader receives request with no segment specified
	if len(req.GetSegmentIDs()) == 0 {
		q.streaming.replica.queryRLock()
		defer q.streaming.replica.queryRUnlock()
		cluster, ok := q.clusterService.getShardCluster(req.GetDmlChannel())
		if !ok {
			return nil, fmt.Errorf("channel %s leader is not here", req.GetDmlChannel())
		}

		// add cancel when error occurs
		queryCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		var results []*internalpb.RetrieveResults
		var streamingResults []*segcorepb.RetrieveResults
		var err error
		var mut sync.Mutex
		var wg sync.WaitGroup

		wg.Add(2)

		go func() {
			defer wg.Done()
			// shard leader dispatches request to its shard cluster
			cResults, cErr := cluster.Query(queryCtx, req)
			mut.Lock()
			defer mut.Unlock()
			if cErr != nil {
				err = cErr
				log.Warn("failed to query cluster", zap.Int64("collectionID", q.collectionID), zap.Error(cErr))
				cancel()
				return
			}
			results = cResults
		}()

		go func() {
			defer wg.Done()
			// hold request until guarantee timestamp >= service timestamp
			guaranteeTs := req.GetReq().GetGuaranteeTimestamp()
			q.waitUntilServiceable(ctx, guaranteeTs, tsTypeDML)
			// shard leader queries its own streaming data
			// TODO add context
			sResults, _, _, sErr := q.streaming.retrieve(collectionID, partitionIDs, plan, func(segment *Segment) bool { return segment.vChannelID == q.channel })
			mut.Lock()
			defer mut.Unlock()
			if sErr != nil {
				err = sErr
				log.Warn("failed to query streaming", zap.Int64("collectionID", q.collectionID), zap.Error(err))
				cancel()
				return
			}
			streamingResults = sResults
		}()

		wg.Wait()
		if err != nil {
			return nil, err
		}

		streamingResult, err := mergeRetrieveResults(streamingResults)
		if err != nil {
			return nil, err
		}

		// complete results with merged streaming result
		results = append(results, &internalpb.RetrieveResults{
			Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Ids:        streamingResult.Ids,
			FieldsData: streamingResult.FieldsData,
		})
		// merge shard query results
		mergedResults, err := mergeInternalRetrieveResults(results)
		if err != nil {
			return nil, err
		}
		log.Debug("leader retrieve result", zap.String("channel", req.DmlChannel), zap.String("ids", mergedResults.Ids.String()))
		return mergedResults, nil
	}
	q.historical.replica.queryRLock()
	defer q.historical.replica.queryRUnlock()
	// hold request until guarantee timestamp >= service timestamp
	guaranteeTs := req.GetReq().GetGuaranteeTimestamp()
	q.waitUntilServiceable(ctx, guaranteeTs, tsTypeDelta)
	// shard follower considers solely historical segments
	retrieveResults, err := q.historical.retrieveBySegmentIDs(collectionID, segmentIDs, q.vectorChunkManager, plan)
	if err != nil {
		return nil, err
	}
	mergedResult, err := mergeRetrieveResults(retrieveResults)
	if err != nil {
		return nil, err
	}

	log.Debug("follower retrieve result", zap.String("ids", mergedResult.Ids.String()))
	RetrieveResults := &internalpb.RetrieveResults{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Ids:        mergedResult.Ids,
		FieldsData: mergedResult.FieldsData,
	}
	return RetrieveResults, nil
}

// TODO: largely based on function mergeRetrieveResults, need rewriting
func mergeInternalRetrieveResults(retrieveResults []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	var ret *internalpb.RetrieveResults
	var skipDupCnt int64
	var idSet = make(map[int64]struct{})

	// merge results and remove duplicates
	for _, rr := range retrieveResults {
		// skip if fields data is empty
		if len(rr.FieldsData) == 0 {
			continue
		}

		if ret == nil {
			ret = &internalpb.RetrieveResults{
				Ids: &schemapb.IDs{
					IdField: &schemapb.IDs_IntId{
						IntId: &schemapb.LongArray{
							Data: []int64{},
						},
					},
				},
				FieldsData: make([]*schemapb.FieldData, len(rr.FieldsData)),
			}
		}

		if len(ret.FieldsData) != len(rr.FieldsData) {
			log.Warn("mismatch FieldData in RetrieveResults")
			return nil, fmt.Errorf("mismatch FieldData in RetrieveResults")
		}

		dstIds := ret.Ids.GetIntId()
		for i, id := range rr.Ids.GetIntId().GetData() {
			if _, ok := idSet[id]; !ok {
				dstIds.Data = append(dstIds.Data, id)
				typeutil.AppendFieldData(ret.FieldsData, rr.FieldsData, int64(i))
				idSet[id] = struct{}{}
			} else {
				// primary keys duplicate
				skipDupCnt++
			}
		}
	}

	// not found, return default values indicating not result found
	if ret == nil {
		ret = &internalpb.RetrieveResults{
			Ids:        &schemapb.IDs{},
			FieldsData: []*schemapb.FieldData{},
		}
	}

	return ret, nil
}

// func printSearchResultData(data *schemapb.SearchResultData, header string) {
// 	size := len(data.Ids.GetIntId().Data)
// 	if size != len(data.Scores) {
// 		log.Error("SearchResultData length mis-match")
// 	}
// 	log.Debug("==== SearchResultData ====",
// 		zap.String("header", header), zap.Int64("nq", data.NumQueries), zap.Int64("topk", data.TopK))
// 	for i := 0; i < size; i++ {
// 		log.Debug("", zap.Int("i", i), zap.Int64("id", data.Ids.GetIntId().Data[i]), zap.Float32("score", data.Scores[i]))
// 	}
// }
