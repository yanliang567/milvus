package interceptor

import (
	"context"
	"strings"

	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/internal/proto/streamingpb"
	"github.com/milvus-io/milvus/internal/util/streamingutil/status"
)

// NewStreamingServiceUnaryServerInterceptor returns a new unary server interceptor for error handling, metric...
func NewStreamingServiceUnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		resp, err := handler(ctx, req)
		if err == nil {
			return resp, err
		}
		// Streaming Service Method should be overwrite the response error code.
		if strings.HasPrefix(info.FullMethod, streamingpb.ServiceMethodPrefix) {
			err := status.AsStreamingError(err)
			if err == nil {
				// return no error if StreamingError is ok.
				return resp, nil
			}
			return resp, status.NewGRPCStatusFromStreamingError(err).Err()
		}
		return resp, err
	}
}

// NewStreamingServiceStreamServerInterceptor returns a new stream server interceptor for error handling, metric...
func NewStreamingServiceStreamServerInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		err := handler(srv, ss)
		if err == nil {
			return err
		}

		// Streaming Service Method should be overwrite the response error code.
		if strings.HasPrefix(info.FullMethod, streamingpb.ServiceMethodPrefix) {
			err := status.AsStreamingError(err)
			if err == nil {
				// return no error if StreamingError is ok.
				return nil
			}
			return status.NewGRPCStatusFromStreamingError(err).Err()
		}
		return err
	}
}
