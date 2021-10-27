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

package indexcoord

import (
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/util/paramtable"
)

// ParamTable is used to record configuration items.
type ParamTable struct {
	paramtable.BaseTable

	Address string
	Port    int

	EtcdEndpoints []string
	KvRootPath    string
	MetaRootPath  string
	IndexRootPath string

	MinIOAddress         string
	MinIOAccessKeyID     string
	MinIOSecretAccessKey string
	MinIOUseSSL          bool
	MinioBucketName      string

	CreatedTime time.Time
	UpdatedTime time.Time
}

// Params is an alias for ParamTable.
var Params ParamTable
var once sync.Once

// Init is used to initialize configuration items.
func (pt *ParamTable) Init() {
	pt.BaseTable.Init()
	// TODO, load index_node.yaml
	/*err := pt.LoadYaml("advanced/index_coord.yaml")
	if err != nil {
		panic(err)
	}*/

	pt.initEtcdEndpoints()
	pt.initMetaRootPath()
	pt.initKvRootPath()
	pt.initMinIOAddress()
	pt.initMinIOAccessKeyID()
	pt.initMinIOSecretAccessKey()
	pt.initMinIOUseSSL()
	pt.initMinioBucketName()
	pt.initIndexRootPath()
	pt.initRoleName()
}

// InitOnce is used to initialize configuration items, and it will only be called once.
func (pt *ParamTable) InitOnce() {
	once.Do(func() {
		pt.Init()
	})
}

// initEtcdEndpoints initializes the etcd address of configuration items.
func (pt *ParamTable) initEtcdEndpoints() {
	endpoints, err := pt.Load("_EtcdEndpoints")
	if err != nil {
		panic(err)
	}
	pt.EtcdEndpoints = strings.Split(endpoints, ",")
}

// initMetaRootPath initializes the root path of etcd.
func (pt *ParamTable) initMetaRootPath() {
	rootPath, err := pt.Load("etcd.rootPath")
	if err != nil {
		panic(err)
	}
	subPath, err := pt.Load("etcd.metaSubPath")
	if err != nil {
		panic(err)
	}
	pt.MetaRootPath = rootPath + "/" + subPath
}

func (pt *ParamTable) initKvRootPath() {
	rootPath, err := pt.Load("etcd.rootPath")
	if err != nil {
		panic(err)
	}
	subPath, err := pt.Load("etcd.kvSubPath")
	if err != nil {
		panic(err)
	}
	pt.KvRootPath = rootPath + "/" + subPath
}

// initMinIOAddress initializes init the minio address of configuration items.
func (pt *ParamTable) initMinIOAddress() {
	ret, err := pt.Load("_MinioAddress")
	if err != nil {
		panic(err)
	}
	pt.MinIOAddress = ret
}

// initMinIOAccessKeyID initializes the minio access key of configuration items.
func (pt *ParamTable) initMinIOAccessKeyID() {
	ret, err := pt.Load("minio.accessKeyID")
	if err != nil {
		panic(err)
	}
	pt.MinIOAccessKeyID = ret
}

// initMinIOSecretAccessKey initializes the minio secret access key.
func (pt *ParamTable) initMinIOSecretAccessKey() {
	ret, err := pt.Load("minio.secretAccessKey")
	if err != nil {
		panic(err)
	}
	pt.MinIOSecretAccessKey = ret
}

// initMinIOUseSSL initializes the minio use SSL of configuration items.
func (pt *ParamTable) initMinIOUseSSL() {
	ret, err := pt.Load("minio.useSSL")
	if err != nil {
		panic(err)
	}
	pt.MinIOUseSSL, err = strconv.ParseBool(ret)
	if err != nil {
		panic(err)
	}
}

// initMinioBucketName initializes the minio bucket name of configuration items.
func (pt *ParamTable) initMinioBucketName() {
	bucketName, err := pt.Load("minio.bucketName")
	if err != nil {
		panic(err)
	}
	pt.MinioBucketName = bucketName
}

// initIndexRootPath initializes the root path of index files.
func (pt *ParamTable) initIndexRootPath() {
	rootPath, err := pt.Load("minio.rootPath")
	if err != nil {
		panic(err)
	}
	pt.IndexRootPath = path.Join(rootPath, "index_files")
}

func (pt *ParamTable) initRoleName() {
	pt.RoleName = "indexcoord"
}
