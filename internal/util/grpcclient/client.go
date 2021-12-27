package grpcclient

import (
	"context"
	"fmt"
	"sync"
	"time"

	grpcopentracing "github.com/grpc-ecosystem/go-grpc-middleware/tracing/opentracing"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/trace"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/keepalive"
)

const (
	dialTimeout      = 5 * time.Second
	keepAliveTime    = 10 * time.Second
	keepAliveTimeout = 3 * time.Second
)

// GrpcClient abstracts client of grpc
type GrpcClient interface {
	SetRole(string)
	GetRole() string
	SetGetAddrFunc(func() (string, error))
	SetNewGrpcClientFunc(func(cc *grpc.ClientConn) interface{})
	GetGrpcClient(ctx context.Context) (interface{}, error)
	ReCall(ctx context.Context, caller func(client interface{}) (interface{}, error)) (interface{}, error)
	Call(ctx context.Context, caller func(client interface{}) (interface{}, error)) (interface{}, error)
	Close() error
}

// ClientBase is a base of grpc client
type ClientBase struct {
	getAddrFunc   func() (string, error)
	newGrpcClient func(cc *grpc.ClientConn) interface{}

	grpcClient        interface{}
	conn              *grpc.ClientConn
	grpcClientMtx     sync.RWMutex
	role              string
	ClientMaxSendSize int
	ClientMaxRecvSize int
}

// SetRole sets role of client
func (c *ClientBase) SetRole(role string) {
	c.role = role
}

// GetRole returns role of client
func (c *ClientBase) GetRole() string {
	return c.role
}

// SetGetAddrFunc sets getAddrFunc of client
func (c *ClientBase) SetGetAddrFunc(f func() (string, error)) {
	c.getAddrFunc = f
}

// SetNewGrpcClientFunc sets newGrpcClient of client
func (c *ClientBase) SetNewGrpcClientFunc(f func(cc *grpc.ClientConn) interface{}) {
	c.newGrpcClient = f
}

// GetGrpcClient returns grpc client
func (c *ClientBase) GetGrpcClient(ctx context.Context) (interface{}, error) {
	c.grpcClientMtx.RLock()

	if c.grpcClient != nil {
		defer c.grpcClientMtx.RUnlock()
		return c.grpcClient, nil
	}
	c.grpcClientMtx.RUnlock()

	c.grpcClientMtx.Lock()
	defer c.grpcClientMtx.Unlock()

	if c.grpcClient != nil {
		return c.grpcClient, nil
	}

	err := c.connect(ctx)
	if err != nil {
		return nil, err
	}

	return c.grpcClient, nil
}

func (c *ClientBase) resetConnection(client interface{}) {
	c.grpcClientMtx.Lock()
	defer c.grpcClientMtx.Unlock()
	if c.grpcClient == nil {
		return
	}

	if client != c.grpcClient {
		return
	}
	if c.conn != nil {
		_ = c.conn.Close()
	}
	c.conn = nil
	c.grpcClient = nil
}

func (c *ClientBase) connect(ctx context.Context) error {
	addr, err := c.getAddrFunc()
	if err != nil {
		log.Error("failed to get cclient address", zap.Error(err))
		return err
	}

	opts := trace.GetInterceptorOpts()
	dialContext, cancel := context.WithTimeout(ctx, dialTimeout)

	retryPolicy := `{
		"methodConfig": [{
		  "name": [{}],
		  "waitForReady": true,
		  "retryPolicy": {
			  "MaxAttempts": 4,
			  "InitialBackoff": ".1s",
			  "MaxBackoff": ".4s",
			  "BackoffMultiplier": 1.6,
			  "RetryableStatusCodes": [ "UNAVAILABLE" ]
		  }
		}]}`

	conn, err := grpc.DialContext(
		dialContext,
		addr,
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(c.ClientMaxRecvSize),
			grpc.MaxCallSendMsgSize(c.ClientMaxSendSize),
		),
		grpc.WithUnaryInterceptor(grpcopentracing.UnaryClientInterceptor(opts...)),
		grpc.WithStreamInterceptor(grpcopentracing.StreamClientInterceptor(opts...)),
		grpc.WithDefaultServiceConfig(retryPolicy),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                keepAliveTime,    // send pings every 60 seconds if there is no activity
			Timeout:             keepAliveTimeout, // wait 6 second for ping ack before considering the connection dead
			PermitWithoutStream: true,             // send pings even without active streams
		}),
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff: backoff.Config{
				BaseDelay:  100 * time.Millisecond,
				Multiplier: 1.6,
				Jitter:     0.2,
				MaxDelay:   3 * time.Second,
			},
			MinConnectTimeout: dialTimeout,
		}),
	)
	cancel()
	if err != nil {
		return err
	}
	if c.conn != nil {
		_ = c.conn.Close()
	}

	c.conn = conn
	c.grpcClient = c.newGrpcClient(c.conn)
	return nil
}

func (c *ClientBase) callOnce(ctx context.Context, caller func(client interface{}) (interface{}, error)) (interface{}, error) {
	client, err := c.GetGrpcClient(ctx)
	if err != nil {
		return nil, err
	}

	ret, err2 := caller(client)
	if err2 == nil {
		return ret, nil
	}
	if err2 == context.Canceled || err2 == context.DeadlineExceeded {
		return nil, err2
	}

	log.Debug(c.GetRole()+" ClientBase grpc error, start to reset connection", zap.Error(err2))

	c.resetConnection(client)
	return ret, err2
}

// Call does a grpc call
func (c *ClientBase) Call(ctx context.Context, caller func(client interface{}) (interface{}, error)) (interface{}, error) {
	if !funcutil.CheckCtxValid(ctx) {
		return nil, ctx.Err()
	}

	ret, err := c.callOnce(ctx, caller)
	if err != nil {
		traceErr := fmt.Errorf("err: %s\n, %s", err.Error(), trace.StackTrace())
		log.Error(c.GetRole()+" ClientBase Call grpc first call get error ", zap.Error(traceErr))
		return nil, traceErr
	}
	return ret, err
}

func (c *ClientBase) ReCall(ctx context.Context, caller func(client interface{}) (interface{}, error)) (interface{}, error) {
	if !funcutil.CheckCtxValid(ctx) {
		return nil, ctx.Err()
	}

	ret, err := c.callOnce(ctx, caller)
	if err == nil {
		return ret, nil
	}

	traceErr := fmt.Errorf("err: %s\n, %s", err.Error(), trace.StackTrace())
	log.Warn(c.GetRole()+" ClientBase ReCall grpc first call get error ", zap.Error(traceErr))

	if !funcutil.CheckCtxValid(ctx) {
		return nil, ctx.Err()
	}

	ret, err = c.callOnce(ctx, caller)
	if err != nil {
		traceErr = fmt.Errorf("err: %s\n, %s", err.Error(), trace.StackTrace())
		log.Error(c.GetRole()+" ClientBase ReCall grpc second call get error ", zap.Error(traceErr))
		return nil, traceErr
	}
	return ret, err
}

// Close close the client connection
func (c *ClientBase) Close() error {
	c.grpcClientMtx.Lock()
	defer c.grpcClientMtx.Unlock()
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
