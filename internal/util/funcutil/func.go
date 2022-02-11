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

package funcutil

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"time"

	"go.uber.org/zap"

	"github.com/go-basic/ipv4"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/retry"
)

// CheckGrpcReady wait for context timeout, or wait 100ms then send nil to targetCh
func CheckGrpcReady(ctx context.Context, targetCh chan error) {
	select {
	case <-time.After(100 * time.Millisecond):
		targetCh <- nil
	case <-ctx.Done():
		return
	}
}

// GetLocalIP return the local ip address
func GetLocalIP() string {
	return ipv4.LocalIP()
}

// WaitForComponentStates wait for component's state to be one of the specific states
func WaitForComponentStates(ctx context.Context, service types.Component, serviceName string, states []internalpb.StateCode, attempts uint, sleep time.Duration) error {
	checkFunc := func() error {
		resp, err := service.GetComponentStates(ctx)
		if err != nil {
			return err
		}

		if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
			return errors.New(resp.Status.Reason)
		}

		meet := false
		for _, state := range states {
			if resp.State.StateCode == state {
				meet = true
				break
			}
		}
		if !meet {
			return fmt.Errorf(
				"WaitForComponentStates, not meet, %s current state: %s",
				serviceName,
				resp.State.StateCode.String())
		}
		return nil
	}
	return retry.Do(ctx, checkFunc, retry.Attempts(attempts), retry.Sleep(sleep))
}

// WaitForComponentInitOrHealthy wait for component's state to be initializing or healthy
func WaitForComponentInitOrHealthy(ctx context.Context, service types.Component, serviceName string, attempts uint, sleep time.Duration) error {
	return WaitForComponentStates(ctx, service, serviceName, []internalpb.StateCode{internalpb.StateCode_Initializing, internalpb.StateCode_Healthy}, attempts, sleep)
}

// WaitForComponentInit wait for component's state to be initializing
func WaitForComponentInit(ctx context.Context, service types.Component, serviceName string, attempts uint, sleep time.Duration) error {
	return WaitForComponentStates(ctx, service, serviceName, []internalpb.StateCode{internalpb.StateCode_Initializing}, attempts, sleep)
}

// WaitForComponentHealthy wait for component's state to be healthy
func WaitForComponentHealthy(ctx context.Context, service types.Component, serviceName string, attempts uint, sleep time.Duration) error {
	return WaitForComponentStates(ctx, service, serviceName, []internalpb.StateCode{internalpb.StateCode_Healthy}, attempts, sleep)
}

// ParseIndexParamsMap parse the jsonic index parameters to map
func ParseIndexParamsMap(mStr string) (map[string]string, error) {
	buffer := make(map[string]interface{})
	err := json.Unmarshal([]byte(mStr), &buffer)
	if err != nil {
		return nil, errors.New("Unmarshal params failed")
	}
	ret := make(map[string]string)
	for key, value := range buffer {
		valueStr := fmt.Sprintf("%v", value)
		ret[key] = valueStr
	}
	return ret, nil
}

const (
	// PulsarMaxMessageSizeKey is the key of config item
	PulsarMaxMessageSizeKey = "maxMessageSize"
)

// GetPulsarConfig get pulsar configuration using pulsar admin api
func GetPulsarConfig(protocol, ip, port, url string, args ...int64) (map[string]interface{}, error) {
	var resp *http.Response
	var err error

	getResp := func() error {
		log.Debug("function util", zap.String("url", protocol+"://"+ip+":"+port+url))
		resp, err = http.Get(protocol + "://" + ip + ":" + port + url)
		return err
	}

	var attempt uint = 10
	var interval = time.Second
	if len(args) > 0 && args[0] > 0 {
		attempt = uint(args[0])
	}
	if len(args) > 1 && args[1] > 0 {
		interval = time.Duration(args[1])
	}

	err = retry.Do(context.TODO(), getResp, retry.Attempts(attempt), retry.Sleep(interval))
	if err != nil {
		log.Debug("failed to get config", zap.String("error", err.Error()))
		return nil, err
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	log.Debug("get config", zap.String("config", string(body)))
	if err != nil {
		return nil, err
	}

	ret := make(map[string]interface{})
	err = json.Unmarshal(body, &ret)
	if err != nil {
		return nil, err
	}

	return ret, nil
}

// GetAttrByKeyFromRepeatedKV return the value corresponding to key in kv pair
func GetAttrByKeyFromRepeatedKV(key string, kvs []*commonpb.KeyValuePair) (string, error) {
	for _, kv := range kvs {
		if kv.Key == key {
			return kv.Value, nil
		}
	}

	return "", errors.New("key " + key + " not found")
}

// CheckCtxValid check if the context is valid
func CheckCtxValid(ctx context.Context) bool {
	return ctx.Err() != context.DeadlineExceeded && ctx.Err() != context.Canceled
}

func GetVecFieldIDs(schema *schemapb.CollectionSchema) []int64 {
	var vecFieldIDs []int64
	for _, field := range schema.Fields {
		if field.DataType == schemapb.DataType_BinaryVector || field.DataType == schemapb.DataType_FloatVector {
			vecFieldIDs = append(vecFieldIDs, field.FieldID)
		}
	}

	return vecFieldIDs
}

func Map2KeyValuePair(datas map[string]string) []*commonpb.KeyValuePair {
	results := make([]*commonpb.KeyValuePair, len(datas))
	offset := 0
	for key, value := range datas {
		results[offset] = &commonpb.KeyValuePair{
			Key:   key,
			Value: value,
		}
		offset++
	}
	return results
}

func KeyValuePair2Map(datas []*commonpb.KeyValuePair) map[string]string {
	results := make(map[string]string)
	for _, pair := range datas {
		results[pair.Key] = pair.Value
	}

	return results
}

// GenChannelSubName generate subName to watch channel
func GenChannelSubName(prefix string, collectionID int64, nodeID int64) string {
	return fmt.Sprintf("%s-%d-%d", prefix, collectionID, nodeID)
}

// CheckPortAvailable check if a port is available to be listened on
func CheckPortAvailable(port int) bool {
	addr := ":" + strconv.Itoa(port)
	listener, err := net.Listen("tcp", addr)
	if listener != nil {
		listener.Close()
	}
	return err == nil
}

// GetAvailablePort return an available port that can be listened on
func GetAvailablePort() int {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		panic(err)
	}
	defer listener.Close()

	return listener.Addr().(*net.TCPAddr).Port
}
