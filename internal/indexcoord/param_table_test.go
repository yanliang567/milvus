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
	"testing"
	"time"
)

func TestParamTable(t *testing.T) {
	Params.Init()

	t.Run("Address", func(t *testing.T) {
		t.Logf("Address: %v", Params.Address)
	})

	t.Run("Port", func(t *testing.T) {
		t.Logf("Port: %v", Params.Port)
	})

	t.Run("EtcdEndpoints", func(t *testing.T) {
		t.Logf("EtcdEndpoints: %v", Params.EtcdEndpoints)
	})

	t.Run("KvRootPath", func(t *testing.T) {
		t.Logf("KvRootPath: %v", Params.KvRootPath)
	})

	t.Run("MetaRootPath", func(t *testing.T) {
		t.Logf("MetaRootPath: %v", Params.MetaRootPath)
	})

	t.Run("MinIOAddress", func(t *testing.T) {
		t.Logf("MinIOAddress: %v", Params.MinIOAddress)
	})

	t.Run("MinIOAccessKeyID", func(t *testing.T) {
		t.Logf("MinIOAccessKeyID: %v", Params.MinIOAccessKeyID)
	})

	t.Run("MinIOSecretAccessKey", func(t *testing.T) {
		t.Logf("MinIOSecretAccessKey: %v", Params.MinIOSecretAccessKey)
	})

	t.Run("MinIOUseSSL", func(t *testing.T) {
		t.Logf("MinIOUseSSL: %v", Params.MinIOUseSSL)
	})

	t.Run("MinioBucketName", func(t *testing.T) {
		t.Logf("MinioBucketName: %v", Params.MinioBucketName)
	})

	t.Run("CreatedTime", func(t *testing.T) {
		Params.CreatedTime = time.Now()
		t.Logf("CreatedTime: %v", Params.CreatedTime)
	})

	t.Run("UpdatedTime", func(t *testing.T) {
		Params.UpdatedTime = time.Now()
		t.Logf("UpdatedTime: %v", Params.UpdatedTime)
	})

	t.Run("initIndexRootPath", func(t *testing.T) {
		t.Logf("IndexRootPath: %v", Params.IndexRootPath)
	})
}

//TODO: Params Load should be return error when key does not exist.
//func shouldPanic(t *testing.T, name string, f func()) {
//	defer func() {
//		if r := recover(); r != nil {
//			t.Logf("%v recover: %v", name, r)
//		}
//	}()
//	f()
//}
//
//func TestParamTable_Panic(t *testing.T) {
//	Params.Init()
//
//
//	t.Run("initEtcdEndpoints", func(t *testing.T) {
//		err := Params.Remove("_EtcdEndpoints")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initEtcdEndpoints", func() {
//			Params.initEtcdEndpoints()
//		})
//	})
//
//	t.Run("initMetaRootPath", func(t *testing.T) {
//		err := Params.Remove("etcd.rootPath")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMetaRootPath", func() {
//			Params.initMetaRootPath()
//		})
//	})
//
//	t.Run("initKvRootPath", func(t *testing.T) {
//		err := Params.Remove("etcd.rootPath")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initKvRootPath", func() {
//			Params.initKvRootPath()
//		})
//
//		err = Params.Save("etcd.rootPath", "test")
//		assert.Nil(t, err)
//
//		err = Params.Remove("etcd.kvSubPath")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initKvRootPath", func() {
//			Params.initKvRootPath()
//		})
//	})
//
//	t.Run("initMinIOAddress", func(t *testing.T) {
//		err := Params.Remove("_MinioAddress")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMinIOAddress", func() {
//			Params.initMinIOAddress()
//		})
//	})
//
//	t.Run("initMinIOAccessKeyID", func(t *testing.T) {
//		err := Params.Remove("minio.accessKeyID")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMinIOAccessKeyID", func() {
//			Params.initMinIOAccessKeyID()
//		})
//	})
//
//	t.Run("initMinIOSecretAccessKey", func(t *testing.T) {
//		err := Params.Remove("minio.secretAccessKey")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMinIOSecretAccessKey", func() {
//			Params.initMinIOSecretAccessKey()
//		})
//	})
//
//	t.Run("initMinIOUseSSL", func(t *testing.T) {
//		err := Params.Remove("minio.useSSL")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMinIOUseSSL", func() {
//			Params.initMinIOUseSSL()
//		})
//	})
//
//	t.Run("initMinioBucketName", func(t *testing.T) {
//		err := Params.Remove("minio.bucketName")
//		assert.Nil(t, err)
//
//		shouldPanic(t, "initMinioBucketName", func() {
//			Params.initMinioBucketName()
//		})
//	})
//}
