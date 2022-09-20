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

package rootcoord

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/dependency"
)

func TestTimetickSync(t *testing.T) {
	ctx := context.Background()
	sourceID := int64(100)

	factory := dependency.NewDefaultFactory(true)

	//chanMap := map[typeutil.UniqueID][]string{
	//	int64(1): {"rootcoord-dml_0"},
	//}

	Params.RootCoordCfg.DmlChannelNum = 2
	Params.CommonCfg.RootCoordDml = "rootcoord-dml"
	Params.CommonCfg.RootCoordDelta = "rootcoord-delta"
	ttSync := newTimeTickSync(ctx, sourceID, factory, nil)

	var wg sync.WaitGroup
	wg.Add(1)
	t.Run("sendToChannel", func(t *testing.T) {
		defer wg.Done()
		ttSync.sendToChannel()

		ttSync.sess2ChanTsMap[1] = nil
		ttSync.sendToChannel()

		msg := &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_TimeTick,
			},
		}
		ttSync.sess2ChanTsMap[1] = newChanTsMsg(msg, 1)
		ttSync.sendToChannel()
	})

	wg.Add(1)
	t.Run("UpdateTimeTick", func(t *testing.T) {
		defer wg.Done()
		msg := &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_TimeTick,
				SourceID: int64(1),
			},
			DefaultTimestamp: 0,
		}

		err := ttSync.updateTimeTick(msg, "1")
		assert.Nil(t, err)

		msg.ChannelNames = append(msg.ChannelNames, "a")
		err = ttSync.updateTimeTick(msg, "1")
		assert.Error(t, err)

		msg.Timestamps = append(msg.Timestamps, uint64(2))
		msg.DefaultTimestamp = uint64(200)
		cttMsg := newChanTsMsg(msg, 1)
		ttSync.sess2ChanTsMap[msg.Base.SourceID] = cttMsg

		err = ttSync.updateTimeTick(msg, "1")
		assert.Nil(t, err)

		ttSync.sourceID = int64(1)
		err = ttSync.updateTimeTick(msg, "1")
		assert.Nil(t, err)
	})

	wg.Add(1)
	t.Run("minTimeTick", func(t *testing.T) {
		defer wg.Done()
		tts := make([]uint64, 2)
		tts[0] = uint64(5)
		tts[1] = uint64(3)

		ret := minTimeTick(tts...)
		assert.Equal(t, ret, tts[1])
	})
	wg.Wait()
}
