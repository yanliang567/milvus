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

package components

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/milvuspb"

	grpcindexcoord "github.com/milvus-io/milvus/internal/distributed/indexcoord"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// IndexCoord implements IndexCoord grpc server
type IndexCoord struct {
	svr *grpcindexcoord.Server
}

// NewIndexCoord creates a new IndexCoord
func NewIndexCoord(ctx context.Context, factory dependency.Factory) (*IndexCoord, error) {
	var err error
	s := &IndexCoord{}
	svr, err := grpcindexcoord.NewServer(ctx, factory)

	if err != nil {
		return nil, err
	}
	s.svr = svr
	return s, nil
}

// Run starts service
func (s *IndexCoord) Run() error {
	if err := s.svr.Run(); err != nil {
		return err
	}
	log.Debug("IndexCoord successfully started")
	return nil
}

// Stop terminates service
func (s *IndexCoord) Stop() error {
	if err := s.svr.Stop(); err != nil {
		return err
	}
	return nil
}

// GetComponentStates returns indexnode's states
func (s *IndexCoord) Health(ctx context.Context) commonpb.StateCode {
	resp, err := s.svr.GetComponentStates(ctx, &milvuspb.GetComponentStatesRequest{})
	if err != nil {
		return commonpb.StateCode_Abnormal
	}
	return resp.State.GetStateCode()
}

func (s *IndexCoord) GetName() string {
	return typeutil.IndexCoordRole
}
