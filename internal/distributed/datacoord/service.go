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

// Package grpcdatacoord implements grpc server for datacoord
package grpcdatacoord

import (
	"context"
	"io"
	"net"
	"strconv"
	"sync"

	"github.com/milvus-io/milvus/internal/logutil"
	"github.com/milvus-io/milvus/internal/types"

	"go.uber.org/zap"

	"google.golang.org/grpc"

	grpc_opentracing "github.com/grpc-ecosystem/go-grpc-middleware/tracing/opentracing"
	grpc_prometheus "github.com/grpc-ecosystem/go-grpc-prometheus"
	"github.com/milvus-io/milvus/internal/datacoord"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/trace"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

// Server is the grpc server of datacoord
type Server struct {
	ctx    context.Context
	cancel context.CancelFunc

	wg        sync.WaitGroup
	dataCoord types.DataCoord

	grpcErrChan chan error
	grpcServer  *grpc.Server
	closer      io.Closer
}

// NewServer new data service grpc server
func NewServer(ctx context.Context, factory msgstream.Factory, opts ...datacoord.Option) (*Server, error) {
	var err error
	ctx1, cancel := context.WithCancel(ctx)

	s := &Server{
		ctx:         ctx1,
		cancel:      cancel,
		grpcErrChan: make(chan error),
	}
	s.dataCoord, err = datacoord.CreateServer(s.ctx, factory, opts...)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Server) init() error {
	Params.Init()

	closer := trace.InitTracing("datacoord")
	s.closer = closer

	datacoord.Params.InitOnce()
	datacoord.Params.IP = Params.IP
	datacoord.Params.Port = Params.Port

	err := s.dataCoord.Register()
	if err != nil {
		log.Debug("DataCoord Register etcd failed", zap.Error(err))
		return err
	}
	log.Debug("DataCoord Register etcd success")

	err = s.startGrpc()
	if err != nil {
		log.Debug("DataCoord startGrpc failed", zap.Error(err))
		return err
	}

	if err := s.dataCoord.Init(); err != nil {
		log.Error("dataCoord init error", zap.Error(err))
		return err
	}
	return nil
}

func (s *Server) startGrpc() error {
	s.wg.Add(1)
	go s.startGrpcLoop(Params.Port)
	// wait for grpc server loop start
	err := <-s.grpcErrChan
	return err
}

func (s *Server) startGrpcLoop(grpcPort int) {
	defer logutil.LogPanic()
	defer s.wg.Done()

	log.Debug("network port", zap.Int("port", grpcPort))
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(grpcPort))
	if err != nil {
		log.Error("grpc server failed to listen error", zap.Error(err))
		s.grpcErrChan <- err
		return
	}

	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	opts := trace.GetInterceptorOpts()
	s.grpcServer = grpc.NewServer(
		grpc.MaxRecvMsgSize(Params.ServerMaxRecvSize),
		grpc.MaxSendMsgSize(Params.ServerMaxSendSize),
		grpc.UnaryInterceptor(
			grpc_opentracing.UnaryServerInterceptor(opts...)),
		grpc.StreamInterceptor(
			grpc_opentracing.StreamServerInterceptor(opts...)))
	//grpc.UnaryInterceptor(grpc_prometheus.UnaryServerInterceptor))
	datapb.RegisterDataCoordServer(s.grpcServer, s)
	grpc_prometheus.Register(s.grpcServer)
	go funcutil.CheckGrpcReady(ctx, s.grpcErrChan)
	if err := s.grpcServer.Serve(lis); err != nil {
		s.grpcErrChan <- err
	}
}

func (s *Server) start() error {
	return s.dataCoord.Start()
}

// Stop stops the DataCoord server gracefully.
// Need to call the GracefulStop interface of grpc server and call the stop method of the inner DataCoord object.
func (s *Server) Stop() error {
	var err error
	if s.closer != nil {
		if err = s.closer.Close(); err != nil {
			return err
		}
	}
	s.cancel()

	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}

	err = s.dataCoord.Stop()
	if err != nil {
		return err
	}

	s.wg.Wait()

	return nil
}

// Run starts the Server. Need to call inner init and start method.
func (s *Server) Run() error {
	if err := s.init(); err != nil {
		return err
	}
	log.Debug("DataCoord init done ...")

	if err := s.start(); err != nil {
		return err
	}
	log.Debug("DataCoord start done ...")
	return nil
}

// GetComponentStates gets states of datacoord and datanodes
func (s *Server) GetComponentStates(ctx context.Context, req *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error) {
	return s.dataCoord.GetComponentStates(ctx)
}

// GetTimeTickChannel gets timetick channel
func (s *Server) GetTimeTickChannel(ctx context.Context, req *internalpb.GetTimeTickChannelRequest) (*milvuspb.StringResponse, error) {
	return s.dataCoord.GetTimeTickChannel(ctx)
}

// GetStatisticsChannel gets statistics channel
func (s *Server) GetStatisticsChannel(ctx context.Context, req *internalpb.GetStatisticsChannelRequest) (*milvuspb.StringResponse, error) {
	return s.dataCoord.GetStatisticsChannel(ctx)
}

// GetSegmentInfo gets segment information according to segment id
func (s *Server) GetSegmentInfo(ctx context.Context, req *datapb.GetSegmentInfoRequest) (*datapb.GetSegmentInfoResponse, error) {
	return s.dataCoord.GetSegmentInfo(ctx, req)
}

// Flush flushes a collection's data
func (s *Server) Flush(ctx context.Context, req *datapb.FlushRequest) (*datapb.FlushResponse, error) {
	return s.dataCoord.Flush(ctx, req)
}

// AssignSegmentID requests to allocate segment space for insert
func (s *Server) AssignSegmentID(ctx context.Context, req *datapb.AssignSegmentIDRequest) (*datapb.AssignSegmentIDResponse, error) {
	return s.dataCoord.AssignSegmentID(ctx, req)
}

// GetSegmentStates gets states of segments
func (s *Server) GetSegmentStates(ctx context.Context, req *datapb.GetSegmentStatesRequest) (*datapb.GetSegmentStatesResponse, error) {
	return s.dataCoord.GetSegmentStates(ctx, req)
}

// GetInsertBinlogPaths gets insert binlog paths of a segment
func (s *Server) GetInsertBinlogPaths(ctx context.Context, req *datapb.GetInsertBinlogPathsRequest) (*datapb.GetInsertBinlogPathsResponse, error) {
	return s.dataCoord.GetInsertBinlogPaths(ctx, req)
}

// GetCollectionStatistics gets statistics of a collection
func (s *Server) GetCollectionStatistics(ctx context.Context, req *datapb.GetCollectionStatisticsRequest) (*datapb.GetCollectionStatisticsResponse, error) {
	return s.dataCoord.GetCollectionStatistics(ctx, req)
}

// GetPartitionStatistics gets statistics of a partition
func (s *Server) GetPartitionStatistics(ctx context.Context, req *datapb.GetPartitionStatisticsRequest) (*datapb.GetPartitionStatisticsResponse, error) {
	return s.dataCoord.GetPartitionStatistics(ctx, req)
}

// GetSegmentInfoChannel gets channel to which datacoord sends segment information
func (s *Server) GetSegmentInfoChannel(ctx context.Context, req *datapb.GetSegmentInfoChannelRequest) (*milvuspb.StringResponse, error) {
	return s.dataCoord.GetSegmentInfoChannel(ctx)
}

// SaveBinlogPaths implement DataCoordServer, saves segment, collection binlog according to datanode request
func (s *Server) SaveBinlogPaths(ctx context.Context, req *datapb.SaveBinlogPathsRequest) (*commonpb.Status, error) {
	return s.dataCoord.SaveBinlogPaths(ctx, req)
}

// GetRecoveryInfo gets information for recovering channels
func (s *Server) GetRecoveryInfo(ctx context.Context, req *datapb.GetRecoveryInfoRequest) (*datapb.GetRecoveryInfoResponse, error) {
	return s.dataCoord.GetRecoveryInfo(ctx, req)
}

// GetFlushedSegments get all flushed segments of a partition
func (s *Server) GetFlushedSegments(ctx context.Context, req *datapb.GetFlushedSegmentsRequest) (*datapb.GetFlushedSegmentsResponse, error) {
	return s.dataCoord.GetFlushedSegments(ctx, req)
}

// GetMetrics gets metrics of data coordinator and datanodes
func (s *Server) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	return s.dataCoord.GetMetrics(ctx, req)
}

func (s *Server) CompleteCompaction(ctx context.Context, req *datapb.CompactionResult) (*commonpb.Status, error) {
	return s.dataCoord.CompleteCompaction(ctx, req)
}

func (s *Server) ManualCompaction(ctx context.Context, req *datapb.ManualCompactionRequest) (*datapb.ManualCompactionResponse, error) {
	return s.dataCoord.ManualCompaction(ctx, req)
}

func (s *Server) GetCompactionState(ctx context.Context, req *datapb.GetCompactionStateRequest) (*datapb.GetCompactionStateResponse, error) {
	return s.dataCoord.GetCompactionState(ctx, req)
}
