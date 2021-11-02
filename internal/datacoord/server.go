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

package datacoord

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	datanodeclient "github.com/milvus-io/milvus/internal/distributed/datanode/client"
	rootcoordclient "github.com/milvus-io/milvus/internal/distributed/rootcoord/client"
	"github.com/milvus-io/milvus/internal/logutil"
	"github.com/milvus-io/milvus/internal/rootcoord"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/mqclient"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"go.uber.org/zap"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

const connEtcdMaxRetryTime = 100000

var (
	// TODO: sunby put to config
	enableTtChecker           = true
	ttCheckerName             = "dataTtChecker"
	ttMaxInterval             = 3 * time.Minute
	ttCheckerWarnMsg          = fmt.Sprintf("we haven't received tt for %f minutes", ttMaxInterval.Minutes())
	segmentTimedFlushDuration = 10.0
)

type (
	// UniqueID shortcut for typeutil.UniqueID
	UniqueID = typeutil.UniqueID
	// Timestamp shortcurt for typeutil.Timestamp
	Timestamp = typeutil.Timestamp
)

// ServerState type alias, presents datacoord Server State
type ServerState = int64

const (
	// ServerStateStopped state stands for just created or stopped `Server` instance
	ServerStateStopped ServerState = 0
	// ServerStateInitializing state stands initializing `Server` instance
	ServerStateInitializing ServerState = 1
	// ServerStateHealthy state stands for healthy `Server` instance
	ServerStateHealthy ServerState = 2
)

type dataNodeCreatorFunc func(ctx context.Context, addr string) (types.DataNode, error)
type rootCoordCreatorFunc func(ctx context.Context, metaRootPath string, etcdEndpoints []string) (types.RootCoord, error)

// makes sure Server implements `DataCoord`
var _ types.DataCoord = (*Server)(nil)

// makes sure Server implements `positionProvider`
var _ positionProvider = (*Server)(nil)

// Server implements `types.Datacoord`
// handles Data Cooridinator related jobs
type Server struct {
	ctx              context.Context
	serverLoopCtx    context.Context
	serverLoopCancel context.CancelFunc
	serverLoopWg     sync.WaitGroup
	isServing        ServerState
	helper           ServerHelper

	kvClient        *etcdkv.EtcdKV
	meta            *meta
	segmentManager  Manager
	allocator       allocator
	cluster         *Cluster
	channelManager  *ChannelManager
	rootCoordClient types.RootCoord

	metricsCacheManager *metricsinfo.MetricsCacheManager

	flushCh   chan UniqueID
	msFactory msgstream.Factory

	session *sessionutil.Session
	eventCh <-chan *sessionutil.SessionEvent

	dataNodeCreator        dataNodeCreatorFunc
	rootCoordClientCreator rootCoordCreatorFunc
}

// ServerHelper datacoord server injection helper
type ServerHelper struct {
	eventAfterHandleDataNodeTt func()
}

func defaultServerHelper() ServerHelper {
	return ServerHelper{
		eventAfterHandleDataNodeTt: func() {},
	}
}

// Option utility function signature to set DataCoord server attributes
type Option func(svr *Server)

// SetRootCoordCreator returns an `Option` setting RootCoord creator with provided parameter
func SetRootCoordCreator(creator rootCoordCreatorFunc) Option {
	return func(svr *Server) {
		svr.rootCoordClientCreator = creator
	}
}

// SetServerHelper returns an `Option` setting ServerHelp with provided parameter
func SetServerHelper(helper ServerHelper) Option {
	return func(svr *Server) {
		svr.helper = helper
	}
}

// SetCluster returns an `Option` setting Cluster with provided parameter
func SetCluster(cluster *Cluster) Option {
	return func(svr *Server) {
		svr.cluster = cluster
	}
}

// SetDataNodeCreator returns an `Option` setting DataNode create function
func SetDataNodeCreator(creator dataNodeCreatorFunc) Option {
	return func(svr *Server) {
		svr.dataNodeCreator = creator
	}
}

// CreateServer create `Server` instance
func CreateServer(ctx context.Context, factory msgstream.Factory, opts ...Option) (*Server, error) {
	rand.Seed(time.Now().UnixNano())
	s := &Server{
		ctx:                    ctx,
		msFactory:              factory,
		flushCh:                make(chan UniqueID, 1024),
		dataNodeCreator:        defaultDataNodeCreatorFunc,
		rootCoordClientCreator: defaultRootCoordCreatorFunc,
		helper:                 defaultServerHelper(),

		metricsCacheManager: metricsinfo.NewMetricsCacheManager(),
	}

	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

func defaultDataNodeCreatorFunc(ctx context.Context, addr string) (types.DataNode, error) {
	return datanodeclient.NewClient(ctx, addr)
}

func defaultRootCoordCreatorFunc(ctx context.Context, metaRootPath string, etcdEndpoints []string) (types.RootCoord, error) {
	return rootcoordclient.NewClient(ctx, metaRootPath, etcdEndpoints)
}

// Register register data service at etcd
func (s *Server) Register() error {
	s.session = sessionutil.NewSession(s.ctx, Params.MetaRootPath, Params.EtcdEndpoints)
	if s.session == nil {
		return errors.New("failed to initialize session")
	}
	s.session.Init(typeutil.DataCoordRole, Params.IP, true)
	Params.NodeID = s.session.ServerID
	Params.SetLogger(typeutil.UniqueID(-1))
	return nil
}

// Init change server state to Initializing
func (s *Server) Init() error {
	atomic.StoreInt64(&s.isServing, ServerStateInitializing)
	return nil
}

// Start initialize `Server` members and start loops, follow steps are taken:
// 1. initialize message factory parameters
// 2. initialize root coord client, meta, datanode cluster, segment info channel,
//		allocator, segment manager
// 3. start service discovery and server loops, which includes message stream handler (segment statistics,datanode tt)
//		datanodes etcd watch, etcd alive check and flush completed status check
// 4. set server state to Healthy
func (s *Server) Start() error {
	var err error
	m := map[string]interface{}{
		"PulsarAddress":  Params.PulsarAddress,
		"ReceiveBufSize": 1024,
		"PulsarBufSize":  1024}
	err = s.msFactory.SetParams(m)
	if err != nil {
		return err
	}
	if err = s.initRootCoordClient(); err != nil {
		return err
	}

	if err = s.initMeta(); err != nil {
		return err
	}

	if err = s.initCluster(); err != nil {
		return err
	}

	s.allocator = newRootCoordAllocator(s.rootCoordClient)

	s.startSegmentManager()
	if err = s.initServiceDiscovery(); err != nil {
		return err
	}

	s.startServerLoop()
	Params.CreatedTime = time.Now()
	Params.UpdatedTime = time.Now()
	atomic.StoreInt64(&s.isServing, ServerStateHealthy)
	log.Debug("dataCoordinator startup success")

	return nil
}

func (s *Server) initCluster() error {
	if s.cluster != nil {
		return nil
	}

	var err error
	s.channelManager, err = NewChannelManager(s.kvClient, s)
	if err != nil {
		return err
	}
	sessionManager := NewSessionManager(withSessionCreator(s.dataNodeCreator))
	s.cluster = NewCluster(sessionManager, s.channelManager)
	return nil
}

func (s *Server) initServiceDiscovery() error {
	sessions, rev, err := s.session.GetSessions(typeutil.DataNodeRole)
	if err != nil {
		log.Debug("dataCoord initServiceDiscovery failed", zap.Error(err))
		return err
	}
	log.Debug("registered sessions", zap.Any("sessions", sessions))

	datanodes := make([]*NodeInfo, 0, len(sessions))
	for _, session := range sessions {
		info := &NodeInfo{
			NodeID:  session.ServerID,
			Address: session.Address,
		}
		datanodes = append(datanodes, info)
	}

	s.cluster.Startup(datanodes)

	s.eventCh = s.session.WatchServices(typeutil.DataNodeRole, rev+1)
	return nil
}

func (s *Server) startSegmentManager() {
	s.segmentManager = newSegmentManager(s.meta, s.allocator)
}

func (s *Server) initMeta() error {
	connectEtcdFn := func() error {
		etcdKV, err := etcdkv.NewEtcdKV(Params.EtcdEndpoints, Params.MetaRootPath)
		if err != nil {
			return err
		}

		s.kvClient = etcdKV
		s.meta, err = newMeta(s.kvClient)
		if err != nil {
			return err
		}
		return nil
	}
	return retry.Do(s.ctx, connectEtcdFn, retry.Attempts(connEtcdMaxRetryTime))
}

func (s *Server) startServerLoop() {
	s.serverLoopCtx, s.serverLoopCancel = context.WithCancel(s.ctx)
	s.serverLoopWg.Add(4)
	s.startStatsChannel(s.serverLoopCtx)
	s.startDataNodeTtLoop(s.serverLoopCtx)
	s.startWatchService(s.serverLoopCtx)
	s.startFlushLoop(s.serverLoopCtx)
	go s.session.LivenessCheck(s.serverLoopCtx, func() {
		log.Error("Data Coord disconnected from etcd, process will exit", zap.Int64("Server Id", s.session.ServerID))
		if err := s.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
	})
}

func (s *Server) startStatsChannel(ctx context.Context) {
	statsStream, _ := s.msFactory.NewMsgStream(ctx)
	statsStream.AsConsumer([]string{Params.StatisticsChannelName}, Params.DataCoordSubscriptionName)
	log.Debug("dataCoord create stats channel consumer",
		zap.String("channelName", Params.StatisticsChannelName),
		zap.String("descriptionName", Params.DataCoordSubscriptionName))
	statsStream.Start()
	go func() {
		defer logutil.LogPanic()
		defer s.serverLoopWg.Done()
		defer statsStream.Close()
		for {
			select {
			case <-ctx.Done():
				log.Debug("stats channel shutdown")
				return
			default:
			}
			msgPack := statsStream.Consume()
			if msgPack == nil {
				log.Debug("receive nil stats msg, shutdown stats channel")
				return
			}
			for _, msg := range msgPack.Msgs {
				if msg.Type() != commonpb.MsgType_SegmentStatistics {
					log.Warn("receive unknown msg from segment statistics channel",
						zap.Stringer("msgType", msg.Type()))
					continue
				}
				ssMsg := msg.(*msgstream.SegmentStatisticsMsg)
				for _, stat := range ssMsg.SegStats {
					s.meta.SetCurrentRows(stat.GetSegmentID(), stat.GetNumRows())
				}
			}
		}
	}()
}

func (s *Server) startDataNodeTtLoop(ctx context.Context) {
	ttMsgStream, err := s.msFactory.NewMsgStream(ctx)
	if err != nil {
		log.Error("new msg stream failed", zap.Error(err))
		return
	}
	ttMsgStream.AsConsumerWithPosition([]string{Params.TimeTickChannelName},
		Params.DataCoordSubscriptionName, mqclient.SubscriptionPositionLatest)
	log.Debug("dataCoord create time tick channel consumer",
		zap.String("timeTickChannelName", Params.TimeTickChannelName),
		zap.String("subscriptionName", Params.DataCoordSubscriptionName))
	ttMsgStream.Start()

	go func() {
		var checker *LongTermChecker
		if enableTtChecker {
			checker = NewLongTermChecker(ctx, ttCheckerName, ttMaxInterval, ttCheckerWarnMsg)
			checker.Start()
			defer checker.Stop()
		}
		defer logutil.LogPanic()
		defer s.serverLoopWg.Done()
		defer ttMsgStream.Close()
		for {
			select {
			case <-ctx.Done():
				log.Debug("data node tt loop shutdown")
				return
			default:
			}
			msgPack := ttMsgStream.Consume()
			if msgPack == nil {
				log.Debug("receive nil tt msg, shutdown tt channel")
				return
			}
			for _, msg := range msgPack.Msgs {
				if msg.Type() != commonpb.MsgType_DataNodeTt {
					log.Warn("receive unexpected msg type from tt channel",
						zap.Stringer("msgType", msg.Type()))
					continue
				}
				ttMsg := msg.(*msgstream.DataNodeTtMsg)
				if enableTtChecker {
					checker.Check()
				}

				ch := ttMsg.ChannelName
				ts := ttMsg.Timestamp
				if err := s.segmentManager.ExpireAllocations(ch, ts); err != nil {
					log.Warn("failed to expire allocations", zap.Error(err))
					continue
				}
				physical, _ := tsoutil.ParseTS(ts)
				if time.Since(physical).Minutes() > 1 {
					// if lag behind, log every 1 mins about
					log.RatedWarn(60.0, "Time tick lag behind for more than 1 minutes", zap.String("channel", ch), zap.Time("tt", physical))
				}
				segments, err := s.segmentManager.GetFlushableSegments(ctx, ch, ts)
				if err != nil {
					log.Warn("get flushable segments failed", zap.Error(err))
					continue
				}

				staleSegments := s.meta.SelectSegments(func(info *SegmentInfo) bool {
					return info.GetInsertChannel() == ch &&
						!info.lastFlushTime.IsZero() &&
						time.Since(info.lastFlushTime).Minutes() >= segmentTimedFlushDuration
				})

				if len(segments)+len(staleSegments) == 0 {
					continue
				}
				log.Debug("flush segments", zap.Int64s("segmentIDs", segments), zap.Int("markSegments count", len(staleSegments)))
				segmentInfos := make([]*datapb.SegmentInfo, 0, len(segments))
				for _, id := range segments {
					sInfo := s.meta.GetSegment(id)
					if sInfo == nil {
						log.Error("get segment from meta error", zap.Int64("id", id),
							zap.Error(err))
						continue
					}
					segmentInfos = append(segmentInfos, sInfo.SegmentInfo)
					s.meta.SetLastFlushTime(id, time.Now())
				}
				markSegments := make([]*datapb.SegmentInfo, 0, len(staleSegments))
				for _, segment := range staleSegments {
					for _, fSeg := range segmentInfos {
						// check segment needs flush first
						if segment.GetID() == fSeg.GetID() {
							continue
						}
					}
					markSegments = append(markSegments, segment.SegmentInfo)
					s.meta.SetLastFlushTime(segment.GetID(), time.Now())
				}
				if len(segmentInfos)+len(markSegments) > 0 {
					s.cluster.Flush(s.ctx, segmentInfos, markSegments)
				}
			}
			s.helper.eventAfterHandleDataNodeTt()
		}
	}()
}

// start a goroutine wto watch services
func (s *Server) startWatchService(ctx context.Context) {
	go s.watchService(ctx)
}

// watchService watchs services
func (s *Server) watchService(ctx context.Context) {
	defer logutil.LogPanic()
	defer s.serverLoopWg.Done()
	for {
		select {
		case <-ctx.Done():
			log.Debug("watch service shutdown")
			return
		case event, ok := <-s.eventCh:
			if !ok {
				//TODO add retry logic
				return
			}
			if err := s.handleSessionEvent(ctx, event); err != nil {
				go func() {
					if err := s.Stop(); err != nil {
						log.Warn("datacoord server stop error", zap.Error(err))
					}
				}()
				return
			}
		}
	}

}

// handles session events - DataNodes Add/Del
func (s *Server) handleSessionEvent(ctx context.Context, event *sessionutil.SessionEvent) error {
	if event == nil {
		return nil
	}
	info := &datapb.DataNodeInfo{
		Address:  event.Session.Address,
		Version:  event.Session.ServerID,
		Channels: []*datapb.ChannelStatus{},
	}
	node := &NodeInfo{
		NodeID:  event.Session.ServerID,
		Address: event.Session.Address,
	}
	switch event.EventType {
	case sessionutil.SessionAddEvent:
		log.Info("received datanode register",
			zap.String("address", info.Address),
			zap.Int64("serverID", info.Version))
		if err := s.cluster.Register(node); err != nil {
			log.Warn("failed to regisger node", zap.Int64("id", node.NodeID), zap.String("address", node.Address), zap.Error(err))
			return err
		}
		s.metricsCacheManager.InvalidateSystemInfoMetrics()
	case sessionutil.SessionDelEvent:
		log.Info("received datanode unregister",
			zap.String("address", info.Address),
			zap.Int64("serverID", info.Version))
		if err := s.cluster.UnRegister(node); err != nil {
			log.Warn("failed to deregisger node", zap.Int64("id", node.NodeID), zap.String("address", node.Address), zap.Error(err))
			return err
		}
		s.metricsCacheManager.InvalidateSystemInfoMetrics()
	default:
		log.Warn("receive unknown service event type",
			zap.Any("type", event.EventType))
	}
	return nil
}

func (s *Server) startFlushLoop(ctx context.Context) {
	go func() {
		defer logutil.LogPanic()
		defer s.serverLoopWg.Done()
		ctx2, cancel := context.WithCancel(ctx)
		defer cancel()
		// send `Flushing` segments
		go s.handleFlushingSegments(ctx2)
		for {
			select {
			case <-ctx.Done():
				log.Debug("flush loop shutdown")
				return
			case segmentID := <-s.flushCh:
				//Ignore return error
				_ = s.postFlush(ctx, segmentID)
			}
		}
	}()
}

// post function after flush is done
// 1. check segment id is valid
// 2. notify RootCoord segment is flushed
// 3. change segment state to `Flushed` in meta
func (s *Server) postFlush(ctx context.Context, segmentID UniqueID) error {
	segment := s.meta.GetSegment(segmentID)
	if segment == nil {
		log.Warn("failed to get flused segment", zap.Int64("id", segmentID))
		return errors.New("segment not found")
	}
	// Notify RootCoord segment is flushed
	req := &datapb.SegmentFlushCompletedMsg{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_SegmentFlushDone,
		},
		Segment: segment.SegmentInfo,
	}
	resp, err := s.rootCoordClient.SegmentFlushCompleted(ctx, req)
	if err = VerifyResponse(resp, err); err != nil {
		log.Warn("failed to call SegmentFlushComplete", zap.Int64("segmentID", segmentID), zap.Error(err))
		return err
	}
	// set segment to SegmentState_Flushed
	if err = s.meta.SetState(segmentID, commonpb.SegmentState_Flushed); err != nil {
		log.Error("flush segment complete failed", zap.Error(err))
		return err
	}
	log.Debug("flush segment complete", zap.Int64("id", segmentID))
	return nil
}

// recovery logic, fetch all Segment in `Flushing` state and do Flush notification logic
func (s *Server) handleFlushingSegments(ctx context.Context) {
	segments := s.meta.GetFlushingSegments()
	for _, segment := range segments {
		select {
		case <-ctx.Done():
			return
		case s.flushCh <- segment.ID:
		}
	}
}

func (s *Server) initRootCoordClient() error {
	var err error
	if s.rootCoordClient, err = s.rootCoordClientCreator(s.ctx, Params.MetaRootPath, Params.EtcdEndpoints); err != nil {
		return err
	}
	if err = s.rootCoordClient.Init(); err != nil {
		return err
	}
	return s.rootCoordClient.Start()
}

// Stop do the Server finalize processes
// it checks the server status is healthy, if not, just quit
// if Server is healthy, set server state to stopped, release etcd session,
//	stop message stream client and stop server loops
func (s *Server) Stop() error {
	if !atomic.CompareAndSwapInt64(&s.isServing, ServerStateHealthy, ServerStateStopped) {
		return nil
	}
	log.Debug("dataCoord server shutdown")
	s.cluster.Close()
	s.stopServerLoop()
	return nil
}

// CleanMeta only for test
func (s *Server) CleanMeta() error {
	log.Debug("clean meta", zap.Any("kv", s.kvClient))
	return s.kvClient.RemoveWithPrefix("")
}

func (s *Server) stopServerLoop() {
	s.serverLoopCancel()
	s.serverLoopWg.Wait()
}

//func (s *Server) validateAllocRequest(collID UniqueID, partID UniqueID, channelName string) error {
//	if !s.meta.HasCollection(collID) {
//		return fmt.Errorf("can not find collection %d", collID)
//	}
//	if !s.meta.HasPartition(collID, partID) {
//		return fmt.Errorf("can not find partition %d", partID)
//	}
//	for _, name := range s.insertChannels {
//		if name == channelName {
//			return nil
//		}
//	}
//	return fmt.Errorf("can not find channel %s", channelName)
//}

func (s *Server) loadCollectionFromRootCoord(ctx context.Context, collectionID int64) error {
	resp, err := s.rootCoordClient.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{
		Base: &commonpb.MsgBase{
			MsgType:  commonpb.MsgType_DescribeCollection,
			SourceID: Params.NodeID,
		},
		DbName:       "",
		CollectionID: collectionID,
	})
	if err = VerifyResponse(resp, err); err != nil {
		return err
	}
	presp, err := s.rootCoordClient.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_ShowPartitions,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  Params.NodeID,
		},
		DbName:         "",
		CollectionName: resp.Schema.Name,
		CollectionID:   resp.CollectionID,
	})
	if err = VerifyResponse(presp, err); err != nil {
		log.Error("show partitions error", zap.String("collectionName", resp.Schema.Name),
			zap.Int64("collectionID", resp.CollectionID), zap.Error(err))
		return err
	}
	collInfo := &datapb.CollectionInfo{
		ID:             resp.CollectionID,
		Schema:         resp.Schema,
		Partitions:     presp.PartitionIDs,
		StartPositions: resp.GetStartPositions(),
	}
	s.meta.AddCollection(collInfo)
	return nil
}

// GetVChanPositions get vchannel latest postitions with provided dml channel names
func (s *Server) GetVChanPositions(channel string, collectionID UniqueID, seekFromStartPosition bool) *datapb.VchannelInfo {
	segments := s.meta.GetSegmentsByChannel(channel)
	flushed := make([]*datapb.SegmentInfo, 0)
	unflushed := make([]*datapb.SegmentInfo, 0)
	var seekPosition *internalpb.MsgPosition
	for _, s := range segments {
		if s.State == commonpb.SegmentState_Flushing || s.State == commonpb.SegmentState_Flushed {
			flushed = append(flushed, trimSegmentInfo(s.SegmentInfo))
			if seekPosition == nil || (s.DmlPosition.Timestamp < seekPosition.Timestamp) {
				seekPosition = s.DmlPosition
			}
			continue
		}

		if s.DmlPosition == nil { // segment position all nil
			continue
		}

		unflushed = append(unflushed, trimSegmentInfo(s.SegmentInfo))

		segmentPosition := s.DmlPosition
		if seekFromStartPosition {
			// need to use start position when load collection/partition, querynode does not support seek from checkpoint yet
			// TODO silverxia remove seek from start logic after checkpoint supported in querynode
			segmentPosition = s.StartPosition
		}

		if seekPosition == nil || segmentPosition.Timestamp < seekPosition.Timestamp {
			seekPosition = segmentPosition
		}
	}
	// use collection start position when segment position is not found
	if seekPosition == nil {
		coll := s.meta.GetCollection(collectionID)
		if coll != nil {
			for _, sp := range coll.GetStartPositions() {
				if sp.GetKey() == rootcoord.ToPhysicalChannel(channel) {
					seekPosition = &internalpb.MsgPosition{
						ChannelName: channel,
						MsgID:       sp.GetData(),
					}
				}
			}
		}
	}

	return &datapb.VchannelInfo{
		CollectionID:      collectionID,
		ChannelName:       channel,
		SeekPosition:      seekPosition,
		FlushedSegments:   flushed,
		UnflushedSegments: unflushed,
	}
}

// trimSegmentInfo returns a shallow copy of datapb.SegmentInfo and sets ALL binlog info to nil
func trimSegmentInfo(info *datapb.SegmentInfo) *datapb.SegmentInfo {
	return &datapb.SegmentInfo{
		ID:             info.ID,
		CollectionID:   info.CollectionID,
		PartitionID:    info.PartitionID,
		InsertChannel:  info.InsertChannel,
		NumOfRows:      info.NumOfRows,
		State:          info.State,
		MaxRowNum:      info.MaxRowNum,
		LastExpireTime: info.LastExpireTime,
		StartPosition:  info.StartPosition,
		DmlPosition:    info.DmlPosition,
	}
}
