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

package roles

import (
	"context"
	"fmt"
	"net/http"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/healthz"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/internalpb"
)

func unhealthyHandler(w http.ResponseWriter, r *http.Request, reason string) {
	w.WriteHeader(http.StatusInternalServerError)
	w.Header().Set(healthz.ContentTypeHeader, healthz.ContentTypeText)
	_, err := fmt.Fprint(w, reason)
	if err != nil {
		log.Warn("failed to send response",
			zap.Error(err))
	}
}

func healthyHandler(w http.ResponseWriter, r *http.Request) {
	var err error

	w.WriteHeader(http.StatusOK)
	w.Header().Set(healthz.ContentTypeHeader, healthz.ContentTypeText)
	_, err = fmt.Fprint(w, "OK")
	if err != nil {
		log.Warn("failed to send response",
			zap.Error(err))
	}
}

// GetComponentStatesInterface defines the interface that get states from component.
type GetComponentStatesInterface interface {
	// GetComponentStates returns the states of component.
	GetComponentStates(ctx context.Context, request *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error)
}

type componentsHealthzHandler struct {
	component GetComponentStatesInterface
}

func (handler *componentsHealthzHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	states, err := handler.component.GetComponentStates(context.Background(), &internalpb.GetComponentStatesRequest{})

	if err != nil {
		log.Warn("failed to get component states", zap.Error(err))
		unhealthyHandler(w, r, err.Error())
		return
	}

	if states == nil {
		log.Warn("failed to get component states, states is nil")
		unhealthyHandler(w, r, "failed to get states")
		return
	}

	if states.Status == nil {
		log.Warn("failed to get component states, states.Status is nil")
		unhealthyHandler(w, r, "failed to get status")
		return
	}

	if states.Status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to get component states",
			zap.String("ErrorCode", states.Status.ErrorCode.String()),
			zap.String("Reason", states.Status.Reason))
		unhealthyHandler(w, r, states.Status.Reason)
		return
	}

	if states.State == nil {
		log.Warn("failed to get component states, states.State is nil")
		unhealthyHandler(w, r, "failed to get state")
		return
	}

	if states.State.StateCode != internalpb.StateCode_Healthy {
		log.Warn("component is unhealthy", zap.String("state", states.State.StateCode.String()))
		unhealthyHandler(w, r, fmt.Sprintf("state: %s", states.State.StateCode.String()))
		return
	}

	healthyHandler(w, r)
}
