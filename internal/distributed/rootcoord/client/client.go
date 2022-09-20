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

package grpcrootcoordclient

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/grpcclient"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

var ClientParams paramtable.GrpcClientConfig

// Client grpc client
type Client struct {
	grpcClient grpcclient.GrpcClient
	sess       *sessionutil.Session
}

// NewClient create root coordinator client with specified etcd info and timeout
// ctx execution control context
// metaRoot is the path in etcd for root coordinator registration
// etcdEndpoints are the address list for etcd end points
// timeout is default setting for each grpc call
func NewClient(ctx context.Context, metaRoot string, etcdCli *clientv3.Client) (*Client, error) {
	sess := sessionutil.NewSession(ctx, metaRoot, etcdCli)
	if sess == nil {
		err := fmt.Errorf("new session error, maybe can not connect to etcd")
		log.Debug("QueryCoordClient NewClient failed", zap.Error(err))
		return nil, err
	}
	ClientParams.InitOnce(typeutil.RootCoordRole)
	client := &Client{
		grpcClient: &grpcclient.ClientBase{
			ClientMaxRecvSize:      ClientParams.ClientMaxRecvSize,
			ClientMaxSendSize:      ClientParams.ClientMaxSendSize,
			DialTimeout:            ClientParams.DialTimeout,
			KeepAliveTime:          ClientParams.KeepAliveTime,
			KeepAliveTimeout:       ClientParams.KeepAliveTimeout,
			RetryServiceNameConfig: "milvus.proto.rootcoord.RootCoord",
			MaxAttempts:            ClientParams.MaxAttempts,
			InitialBackoff:         ClientParams.InitialBackoff,
			MaxBackoff:             ClientParams.MaxBackoff,
			BackoffMultiplier:      ClientParams.BackoffMultiplier,
		},
		sess: sess,
	}
	client.grpcClient.SetRole(typeutil.RootCoordRole)
	client.grpcClient.SetGetAddrFunc(client.getRootCoordAddr)
	client.grpcClient.SetNewGrpcClientFunc(client.newGrpcClient)

	return client, nil
}

// Init initialize grpc parameters
func (c *Client) Init() error {
	return nil
}

func (c *Client) newGrpcClient(cc *grpc.ClientConn) interface{} {
	return rootcoordpb.NewRootCoordClient(cc)
}

func (c *Client) getRootCoordAddr() (string, error) {
	key := c.grpcClient.GetRole()
	msess, _, err := c.sess.GetSessions(key)
	if err != nil {
		log.Debug("RootCoordClient GetSessions failed", zap.Any("key", key))
		return "", err
	}
	ms, ok := msess[key]
	if !ok {
		log.Warn("RootCoordClient mess key not exist", zap.Any("key", key))
		return "", fmt.Errorf("find no available rootcoord, check rootcoord state")
	}
	log.Debug("RootCoordClient GetSessions success", zap.String("address", ms.Address))
	return ms.Address, nil
}

// Start dummy
func (c *Client) Start() error {
	return nil
}

// Stop terminate grpc connection
func (c *Client) Stop() error {
	return c.grpcClient.Close()
}

// Register dummy
func (c *Client) Register() error {
	return nil
}

// GetComponentStates TODO: timeout need to be propagated through ctx
func (c *Client) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetComponentStates(ctx, &internalpb.GetComponentStatesRequest{})
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*internalpb.ComponentStates), err
}

// GetTimeTickChannel get timetick channel name
func (c *Client) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetTimeTickChannel(ctx, &internalpb.GetTimeTickChannelRequest{})
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.StringResponse), err
}

// GetStatisticsChannel just define a channel, not used currently
func (c *Client) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetStatisticsChannel(ctx, &internalpb.GetStatisticsChannelRequest{})
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.StringResponse), err
}

// CreateCollection create collection
func (c *Client) CreateCollection(ctx context.Context, in *milvuspb.CreateCollectionRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).CreateCollection(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// DropCollection drop collection
func (c *Client) DropCollection(ctx context.Context, in *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DropCollection(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// HasCollection check collection existence
func (c *Client) HasCollection(ctx context.Context, in *milvuspb.HasCollectionRequest) (*milvuspb.BoolResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).HasCollection(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.BoolResponse), err
}

// DescribeCollection return collection info
func (c *Client) DescribeCollection(ctx context.Context, in *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DescribeCollection(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.DescribeCollectionResponse), err
}

// ShowCollections list all collection names
func (c *Client) ShowCollections(ctx context.Context, in *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ShowCollections(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ShowCollectionsResponse), err
}

// CreatePartition create partition
func (c *Client) CreatePartition(ctx context.Context, in *milvuspb.CreatePartitionRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).CreatePartition(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// DropPartition drop partition
func (c *Client) DropPartition(ctx context.Context, in *milvuspb.DropPartitionRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DropPartition(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// HasPartition check partition existence
func (c *Client) HasPartition(ctx context.Context, in *milvuspb.HasPartitionRequest) (*milvuspb.BoolResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).HasPartition(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.BoolResponse), err
}

// ShowPartitions list all partitions in collection
func (c *Client) ShowPartitions(ctx context.Context, in *milvuspb.ShowPartitionsRequest) (*milvuspb.ShowPartitionsResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ShowPartitions(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ShowPartitionsResponse), err
}

// AllocTimestamp global timestamp allocator
func (c *Client) AllocTimestamp(ctx context.Context, in *rootcoordpb.AllocTimestampRequest) (*rootcoordpb.AllocTimestampResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).AllocTimestamp(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*rootcoordpb.AllocTimestampResponse), err
}

// AllocID global ID allocator
func (c *Client) AllocID(ctx context.Context, in *rootcoordpb.AllocIDRequest) (*rootcoordpb.AllocIDResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).AllocID(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*rootcoordpb.AllocIDResponse), err
}

// UpdateChannelTimeTick used to handle ChannelTimeTickMsg
func (c *Client) UpdateChannelTimeTick(ctx context.Context, in *internalpb.ChannelTimeTickMsg) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).UpdateChannelTimeTick(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// ShowSegments list all segments
func (c *Client) ShowSegments(ctx context.Context, in *milvuspb.ShowSegmentsRequest) (*milvuspb.ShowSegmentsResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ShowSegments(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ShowSegmentsResponse), err
}

// InvalidateCollectionMetaCache notifies RootCoord to release the collection cache in Proxies.
func (c *Client) InvalidateCollectionMetaCache(ctx context.Context, in *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).InvalidateCollectionMetaCache(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// ShowConfigurations gets specified configurations para of RootCoord
func (c *Client) ShowConfigurations(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ShowConfigurations(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}

	return ret.(*internalpb.ShowConfigurationsResponse), err
}

// GetMetrics get metrics
func (c *Client) GetMetrics(ctx context.Context, in *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetMetrics(ctx, in)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.GetMetricsResponse), err
}

// CreateAlias create collection alias
func (c *Client) CreateAlias(ctx context.Context, req *milvuspb.CreateAliasRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).CreateAlias(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// DropAlias drop collection alias
func (c *Client) DropAlias(ctx context.Context, req *milvuspb.DropAliasRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DropAlias(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// AlterAlias alter collection alias
func (c *Client) AlterAlias(ctx context.Context, req *milvuspb.AlterAliasRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).AlterAlias(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

// Import data files(json, numpy, etc.) on MinIO/S3 storage, read and parse them into sealed segments
func (c *Client) Import(ctx context.Context, req *milvuspb.ImportRequest) (*milvuspb.ImportResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).Import(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ImportResponse), err
}

// Check import task state from datanode
func (c *Client) GetImportState(ctx context.Context, req *milvuspb.GetImportStateRequest) (*milvuspb.GetImportStateResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetImportState(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.GetImportStateResponse), err
}

// List id array of all import tasks
func (c *Client) ListImportTasks(ctx context.Context, req *milvuspb.ListImportTasksRequest) (*milvuspb.ListImportTasksResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ListImportTasks(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ListImportTasksResponse), err
}

// Report impot task state to rootcoord
func (c *Client) ReportImport(ctx context.Context, req *rootcoordpb.ImportResult) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ReportImport(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) CreateCredential(ctx context.Context, req *internalpb.CredentialInfo) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).CreateCredential(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) GetCredential(ctx context.Context, req *rootcoordpb.GetCredentialRequest) (*rootcoordpb.GetCredentialResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).GetCredential(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*rootcoordpb.GetCredentialResponse), err
}

func (c *Client) UpdateCredential(ctx context.Context, req *internalpb.CredentialInfo) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).UpdateCredential(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) DeleteCredential(ctx context.Context, req *milvuspb.DeleteCredentialRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DeleteCredential(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) ListCredUsers(ctx context.Context, req *milvuspb.ListCredUsersRequest) (*milvuspb.ListCredUsersResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ListCredUsers(ctx, req)
	})
	if err != nil || ret == nil {
		return nil, err
	}
	return ret.(*milvuspb.ListCredUsersResponse), err
}

func (c *Client) CreateRole(ctx context.Context, req *milvuspb.CreateRoleRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).CreateRole(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) DropRole(ctx context.Context, req *milvuspb.DropRoleRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).DropRole(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) OperateUserRole(ctx context.Context, req *milvuspb.OperateUserRoleRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).OperateUserRole(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) SelectRole(ctx context.Context, req *milvuspb.SelectRoleRequest) (*milvuspb.SelectRoleResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).SelectRole(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*milvuspb.SelectRoleResponse), err
}

func (c *Client) SelectUser(ctx context.Context, req *milvuspb.SelectUserRequest) (*milvuspb.SelectUserResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).SelectUser(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*milvuspb.SelectUserResponse), err
}

func (c *Client) OperatePrivilege(ctx context.Context, req *milvuspb.OperatePrivilegeRequest) (*commonpb.Status, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).OperatePrivilege(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*commonpb.Status), err
}

func (c *Client) SelectGrant(ctx context.Context, req *milvuspb.SelectGrantRequest) (*milvuspb.SelectGrantResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).SelectGrant(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*milvuspb.SelectGrantResponse), err
}

func (c *Client) ListPolicy(ctx context.Context, req *internalpb.ListPolicyRequest) (*internalpb.ListPolicyResponse, error) {
	ret, err := c.grpcClient.ReCall(ctx, func(client interface{}) (interface{}, error) {
		if !funcutil.CheckCtxValid(ctx) {
			return nil, ctx.Err()
		}
		return client.(rootcoordpb.RootCoordClient).ListPolicy(ctx, req)
	})
	if err != nil {
		return nil, err
	}
	return ret.(*internalpb.ListPolicyResponse), err
}
