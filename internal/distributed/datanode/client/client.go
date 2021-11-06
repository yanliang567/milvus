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

package grpcdatanodeclient

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/retry"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware"
	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	grpc_opentracing "github.com/grpc-ecosystem/go-grpc-middleware/tracing/opentracing"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/util/trace"
	"google.golang.org/grpc/codes"

	"go.uber.org/zap"
	"google.golang.org/grpc"
)

// Client is the grpc client for DataNode
type Client struct {
	ctx    context.Context
	cancel context.CancelFunc

	grpc    datapb.DataNodeClient
	conn    *grpc.ClientConn
	grpcMtx sync.RWMutex

	addr string

	retryOptions []retry.Option

	getGrpcClient func() (datapb.DataNodeClient, error)
}

func (c *Client) setGetGrpcClientFunc() {
	c.getGrpcClient = c.getGrpcClientFunc
}

func (c *Client) getGrpcClientFunc() (datapb.DataNodeClient, error) {
	c.grpcMtx.RLock()
	if c.grpc != nil {
		defer c.grpcMtx.RUnlock()
		return c.grpc, nil
	}
	c.grpcMtx.RUnlock()

	c.grpcMtx.Lock()
	defer c.grpcMtx.Unlock()

	if c.grpc != nil {
		return c.grpc, nil
	}

	// FIXME(dragondriver): how to handle error here?
	// if we return nil here, then we should check if client is nil outside,
	err := c.connect(retry.Attempts(20))
	if err != nil {
		return nil, err
	}

	return c.grpc, nil
}

func (c *Client) resetConnection() {
	c.grpcMtx.Lock()
	defer c.grpcMtx.Unlock()

	if c.conn != nil {
		_ = c.conn.Close()
	}
	c.conn = nil
	c.grpc = nil
}

// NewClient creates a client for DataNode.
func NewClient(ctx context.Context, addr string, retryOptions ...retry.Option) (*Client, error) {
	if addr == "" {
		return nil, fmt.Errorf("address is empty")
	}

	ctx, cancel := context.WithCancel(ctx)
	client := &Client{
		ctx:          ctx,
		cancel:       cancel,
		addr:         addr,
		retryOptions: retryOptions,
	}

	client.setGetGrpcClientFunc()
	return client, nil
}

// Init initializes the client.
func (c *Client) Init() error {
	Params.Init()
	return nil
}

func (c *Client) connect(retryOptions ...retry.Option) error {
	connectGrpcFunc := func() error {
		opts := trace.GetInterceptorOpts()
		log.Debug("DataNode connect ", zap.String("address", c.addr))
		ctx, cancel := context.WithTimeout(c.ctx, 15*time.Second)
		defer cancel()
		conn, err := grpc.DialContext(ctx, c.addr,
			grpc.WithInsecure(), grpc.WithBlock(),
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(Params.ClientMaxRecvSize),
				grpc.MaxCallSendMsgSize(Params.ClientMaxSendSize)),
			grpc.WithDisableRetry(),
			grpc.WithUnaryInterceptor(
				grpc_middleware.ChainUnaryClient(
					grpc_retry.UnaryClientInterceptor(
						grpc_retry.WithMax(3),
						grpc_retry.WithCodes(codes.Aborted, codes.Unavailable),
					),
					grpc_opentracing.UnaryClientInterceptor(opts...),
				)),
			grpc.WithStreamInterceptor(
				grpc_middleware.ChainStreamClient(
					grpc_retry.StreamClientInterceptor(
						grpc_retry.WithMax(3),
						grpc_retry.WithCodes(codes.Aborted, codes.Unavailable),
					),
					grpc_opentracing.StreamClientInterceptor(opts...),
				)),
		)
		if err != nil {
			return err
		}
		if c.conn != nil {
			_ = c.conn.Close()
		}
		c.conn = conn
		return nil
	}

	err := retry.Do(c.ctx, connectGrpcFunc, retryOptions...)
	if err != nil {
		log.Debug("DataNodeClient try connect failed", zap.Error(err))
		return err
	}
	log.Debug("DataNodeClient connect success")
	c.grpc = datapb.NewDataNodeClient(c.conn)
	return nil
}

func (c *Client) recall(caller func() (interface{}, error)) (interface{}, error) {
	ret, err := caller()
	if err == nil {
		return ret, nil
	}
	log.Debug("DataNode Client grpc error", zap.Error(err))

	c.resetConnection()

	ret, err = caller()
	if err == nil {
		return ret, nil
	}
	return ret, err
}

// Start starts the client.
// Currently, it does nothing.
func (c *Client) Start() error {
	return nil
}

// Stop stops the client.
// Currently, it closes the grpc connection with the DataNode.
func (c *Client) Stop() error {
	c.cancel()
	c.grpcMtx.Lock()
	defer c.grpcMtx.Unlock()
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Register does nothing.
func (c *Client) Register() error {
	return nil
}

func (c *Client) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.GetComponentStates(ctx, &internalpb.GetComponentStatesRequest{})
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*internalpb.ComponentStates), err
}

func (c *Client) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.GetStatisticsChannel(ctx, &internalpb.GetStatisticsChannelRequest{})
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.StringResponse), err
}

func (c *Client) WatchDmChannels(ctx context.Context, req *datapb.WatchDmChannelsRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.WatchDmChannels(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// FlushSegments notifies DataNode to flush the segments req provids. The flush tasks are async to this
//  rpc, DataNode will flush the segments in the background.
//
// Return UnexpectedError code in status:
//     If DataNode isn't in HEALTHY: states not HEALTHY or dynamic checks not HEALTHY
//     If DataNode doesn't find the correspounding segmentID in its memeory replica
// Return Success code in status and trigers background flush:
//     Log an info log if a segment is under flushing
func (c *Client) FlushSegments(ctx context.Context, req *datapb.FlushSegmentsRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.FlushSegments(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.GetMetrics(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.GetMetricsResponse), err
}

func (c *Client) Compaction(ctx context.Context, req *datapb.CompactionPlan) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		client, err := c.getGrpcClient()
		if err != nil {
			return nil, err
		}

		return client.Compaction(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}
