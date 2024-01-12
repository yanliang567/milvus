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

package config

import (
	"context"
	"os"
	"path"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"go.etcd.io/etcd/server/v3/embed"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
	"golang.org/x/sync/errgroup"
)

func TestAllConfigFromManager(t *testing.T) {
	mgr, _ := Init()
	all := mgr.GetConfigs()
	assert.Equal(t, 0, len(all))

	mgr, _ = Init(WithEnvSource(formatKey))
	all = mgr.GetConfigs()
	assert.Less(t, 0, len(all))
}

func TestConfigChangeEvent(t *testing.T) {
	dir, _ := os.MkdirTemp("", "milvus")
	os.WriteFile(path.Join(dir, "milvus.yaml"), []byte("a.b: 1\nc.d: 2"), 0o600)
	os.WriteFile(path.Join(dir, "user.yaml"), []byte("a.b: 3"), 0o600)

	fs := NewFileSource(&FileInfo{[]string{path.Join(dir, "milvus.yaml"), path.Join(dir, "user.yaml")}, 1})
	mgr, _ := Init()
	err := mgr.AddSource(fs)
	assert.NoError(t, err)
	res, err := mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, res, "3")
	os.WriteFile(path.Join(dir, "user.yaml"), []byte("a.b: 6"), 0o600)
	time.Sleep(3 * time.Second)
	res, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, res, "6")
}

func TestAllDupliateSource(t *testing.T) {
	mgr, _ := Init()
	err := mgr.AddSource(NewEnvSource(formatKey))
	assert.NoError(t, err)
	err = mgr.AddSource(NewEnvSource(formatKey))
	assert.Error(t, err)

	err = mgr.AddSource(ErrSource{})
	assert.Error(t, err, "error")

	err = mgr.pullSourceConfigs("ErrSource")
	assert.Error(t, err, "invalid source or source not added")
}

func TestBasic(t *testing.T) {
	mgr, _ := Init()

	// test set config
	mgr.SetConfig("a.b", "aaa")
	value, err := mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "aaa")
	_, err = mgr.GetConfig("a.a")
	assert.Error(t, err)

	// test delete config
	mgr.SetConfig("a.b", "aaa")
	mgr.DeleteConfig("a.b")
	assert.Error(t, err)

	// test reset config
	mgr.ResetConfig("a.b")
	assert.Error(t, err)

	// test forbid config
	envSource := NewEnvSource(formatKey)
	err = mgr.AddSource(envSource)
	assert.NoError(t, err)

	envSource.configs.Insert("ab", "aaa")
	mgr.OnEvent(&Event{
		EventSource: envSource.GetSourceName(),
		EventType:   CreateType,
		Key:         "ab",
		Value:       "aaa",
	})
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "aaa")

	mgr.ForbidUpdate("a.b")
	mgr.OnEvent(&Event{
		EventSource: envSource.GetSourceName(),
		EventType:   UpdateType,
		Key:         "a.b",
		Value:       "bbb",
	})
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "aaa")

	configs := mgr.FileConfigs()
	assert.Len(t, configs, 0)
}

func TestOnEvent(t *testing.T) {
	cfg, _ := embed.ConfigFromFile("../../configs/advanced/etcd.yaml")
	cfg.Dir = "/tmp/milvus/test"
	e, err := embed.StartEtcd(cfg)
	assert.NoError(t, err)
	defer e.Close()
	defer os.RemoveAll(cfg.Dir)

	client := v3client.New(e.Server)

	dir, _ := os.MkdirTemp("", "milvus")
	yamlFile := path.Join(dir, "milvus.yaml")
	mgr, _ := Init(WithEnvSource(formatKey),
		WithFilesSource(&FileInfo{
			Files:           []string{yamlFile},
			RefreshInterval: 10 * time.Millisecond,
		}),
		WithEtcdSource(&EtcdInfo{
			Endpoints:       []string{cfg.ACUrls[0].Host},
			KeyPrefix:       "test",
			RefreshInterval: 10 * time.Millisecond,
		}))

	os.WriteFile(yamlFile, []byte("a.b: aaa"), 0o600)
	time.Sleep(time.Second)
	value, err := mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "aaa")

	ctx := context.Background()
	client.KV.Put(ctx, "test/config/a/b", "bbb")
	time.Sleep(time.Second)
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "bbb")

	client.KV.Put(ctx, "test/config/a/b", "ccc")
	time.Sleep(time.Second)
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "ccc")

	os.WriteFile(yamlFile, []byte("a.b: ddd"), 0o600)
	time.Sleep(time.Second)
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "ccc")

	client.KV.Delete(ctx, "test/config/a/b")
	time.Sleep(time.Second)
	value, err = mgr.GetConfig("a.b")
	assert.NoError(t, err)
	assert.Equal(t, value, "ddd")
}

func TestDeadlock(t *testing.T) {
	mgr, _ := Init()

	// test concurrent lock and recursive rlock
	wg, _ := errgroup.WithContext(context.Background())
	wg.Go(func() error {
		for i := 0; i < 100; i++ {
			mgr.GetBy(WithPrefix("rootcoord."))
		}
		return nil
	})

	wg.Go(func() error {
		for i := 0; i < 100; i++ {
			mgr.SetConfig("rootcoord.xxx", "111")
		}
		return nil
	})

	wg.Wait()
}

type ErrSource struct{}

func (e ErrSource) Close() {
}

func (e ErrSource) GetConfigurationByKey(string) (string, error) {
	return "", errors.New("error")
}

// GetConfigurations implements Source
func (ErrSource) GetConfigurations() (map[string]string, error) {
	return nil, errors.New("error")
}

// GetPriority implements Source
func (ErrSource) GetPriority() int {
	return 2
}

// GetSourceName implements Source
func (ErrSource) GetSourceName() string {
	return "ErrSource"
}

func (e ErrSource) SetEventHandler(eh EventHandler) {
}

func (e ErrSource) UpdateOptions(opt Options) {
}
