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

package etcd

import (
	"time"

	"github.com/milvus-io/milvus/internal/util/paramtable"
	clientv3 "go.etcd.io/etcd/client/v3"
)

// GetEtcdClient returns etcd client
func GetEtcdClient(cfg *paramtable.EtcdConfig) (*clientv3.Client, error) {
	if cfg.UseEmbedEtcd {
		return GetEmbedEtcdClient()
	}
	return GetRemoteEtcdClient(cfg.Endpoints)
}

// GetRemoteEtcdClient returns client of remote etcd by given endpoints
func GetRemoteEtcdClient(endpoints []string) (*clientv3.Client, error) {
	return clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
}
