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

package querynode

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/mq/msgstream"
)

func TestFlowGraphMsg_insertMsg(t *testing.T) {
	schema := genTestCollectionSchema()
	msg, err := genSimpleInsertMsg(schema, defaultMsgLength)
	assert.NoError(t, err)
	timestampMax := Timestamp(1000)
	im := insertMsg{
		insertMessages: []*msgstream.InsertMsg{
			msg,
		},
		timeRange: TimeRange{
			timestampMin: 0,
			timestampMax: timestampMax,
		},
	}
	time := im.TimeTick()
	assert.Equal(t, timestampMax, time)
}

func TestFlowGraphMsg_serviceTimeMsg(t *testing.T) {
	timestampMax := Timestamp(1000)
	stm := serviceTimeMsg{
		timeRange: TimeRange{
			timestampMin: 0,
			timestampMax: timestampMax,
		},
	}
	time := stm.TimeTick()
	assert.Equal(t, timestampMax, time)
}
