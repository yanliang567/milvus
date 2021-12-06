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

package rocksmq

import "context"

type reader struct {
	c                       *client
	topic                   string
	name                    string
	startMessageID          UniqueID
	startMessageIDInclusive bool
	subscriptionRolePrefix  string
}

func newReader(c *client, readerOptions *ReaderOptions) (*reader, error) {
	if c == nil {
		return nil, newError(InvalidConfiguration, "client is nil")
	}
	if readerOptions == nil {
		return nil, newError(InvalidConfiguration, "options is nil")
	}
	if readerOptions.Topic == "" {
		return nil, newError(InvalidConfiguration, "topic is empty")
	}
	reader := &reader{
		c:                       c,
		topic:                   readerOptions.Topic,
		name:                    readerOptions.Name,
		startMessageID:          readerOptions.StartMessageID,
		startMessageIDInclusive: readerOptions.StartMessageIDInclusive,
		subscriptionRolePrefix:  readerOptions.SubscriptionRolePrefix,
	}
	if c.server == nil {
		return nil, newError(InvalidConfiguration, "rmq server in client is nil")
	}
	name, err := c.server.CreateReader(readerOptions.Topic, reader.startMessageID, reader.startMessageIDInclusive, reader.subscriptionRolePrefix)
	if err != nil {
		return nil, err
	}
	reader.name = name
	return reader, nil
}

func (r *reader) Topic() string {
	return r.topic
}

func (r *reader) Next(ctx context.Context) (Message, error) {
	cMsg, err := r.c.server.Next(ctx, r.topic, r.name, r.startMessageIDInclusive)
	if err != nil {
		return Message{}, err
	}
	msg := Message{
		MsgID:   cMsg.MsgID,
		Payload: cMsg.Payload,
		Topic:   r.topic,
	}
	return msg, nil
}

func (r *reader) HasNext() bool {
	return r.c.server.HasNext(r.topic, r.name, r.startMessageIDInclusive)
}

func (r *reader) Close() {
	r.c.server.CloseReader(r.topic, r.name)
}

func (r *reader) Seek(msgID UniqueID) error { //nolint:govet
	r.c.server.ReaderSeek(r.topic, r.name, msgID)
	return nil
}
