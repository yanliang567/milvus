// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package querynode

import (
	"context"
	"os"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func getSystemInfoMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest, node *QueryNode) (*milvuspb.GetMetricsResponse, error) {
	usedMem, err := getUsedMemory()
	if err != nil {
		return nil, err
	}
	totalMem, err := getTotalMemory()
	if err != nil {
		return nil, err
	}
	nodeInfos := metricsinfo.QueryNodeInfos{
		BaseComponentInfos: metricsinfo.BaseComponentInfos{
			Name: metricsinfo.ConstructComponentName(typeutil.QueryNodeRole, Params.QueryNodeID),
			HardwareInfos: metricsinfo.HardwareMetrics{
				IP:           node.session.Address,
				CPUCoreCount: metricsinfo.GetCPUCoreCount(false),
				CPUCoreUsage: metricsinfo.GetCPUUsage(),
				Memory:       totalMem,
				MemoryUsage:  usedMem,
				Disk:         metricsinfo.GetDiskCount(),
				DiskUsage:    metricsinfo.GetDiskUsage(),
			},
			SystemInfo: metricsinfo.DeployMetrics{
				SystemVersion: os.Getenv(metricsinfo.GitCommitEnvKey),
				DeployMode:    os.Getenv(metricsinfo.DeployModeEnvKey),
			},
			CreatedTime: Params.CreatedTime.String(),
			UpdatedTime: Params.UpdatedTime.String(),
			Type:        typeutil.QueryNodeRole,
			ID:          node.session.ServerID,
		},
		SystemConfigurations: metricsinfo.QueryNodeConfiguration{
			SearchReceiveBufSize:         Params.SearchReceiveBufSize,
			SearchPulsarBufSize:          Params.SearchPulsarBufSize,
			SearchResultReceiveBufSize:   Params.SearchResultReceiveBufSize,
			RetrieveReceiveBufSize:       Params.RetrieveReceiveBufSize,
			RetrievePulsarBufSize:        Params.RetrievePulsarBufSize,
			RetrieveResultReceiveBufSize: Params.RetrieveResultReceiveBufSize,

			SimdType: Params.SimdType,
		},
	}
	resp, err := metricsinfo.MarshalComponentInfos(nodeInfos)
	if err != nil {
		return &milvuspb.GetMetricsResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
			Response:      "",
			ComponentName: metricsinfo.ConstructComponentName(typeutil.QueryNodeRole, Params.QueryNodeID),
		}, nil
	}

	return &milvuspb.GetMetricsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Response:      resp,
		ComponentName: metricsinfo.ConstructComponentName(typeutil.QueryNodeRole, Params.QueryNodeID),
	}, nil
}

func getUsedMemory() (uint64, error) {
	if Params.InContainer {
		return metricsinfo.GetContainerMemUsed()
	}
	return metricsinfo.GetUsedMemoryCount(), nil
}

func getTotalMemory() (uint64, error) {
	if Params.InContainer {
		return metricsinfo.GetContainerMemLimit()
	}
	return metricsinfo.GetMemoryCount(), nil
}
