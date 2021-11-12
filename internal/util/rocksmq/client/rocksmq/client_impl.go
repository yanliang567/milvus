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

import (
	"reflect"
	"sync"

	"github.com/milvus-io/milvus/internal/log"
	server "github.com/milvus-io/milvus/internal/util/rocksmq/server/rocksmq"
	"go.uber.org/zap"
)

type client struct {
	server          RocksMQ
	producerOptions []ProducerOptions
	consumerOptions []ConsumerOptions
	wg              *sync.WaitGroup
	closeCh         chan struct{}
	closeOnce       sync.Once
}

func newClient(options ClientOptions) (*client, error) {
	if options.Server == nil {
		return nil, newError(InvalidConfiguration, "options.Server is nil")
	}

	c := &client{
		server:          options.Server,
		producerOptions: []ProducerOptions{},
		wg:              &sync.WaitGroup{},
		closeCh:         make(chan struct{}),
	}
	return c, nil
}

// CreateProducer create a rocksmq producer
func (c *client) CreateProducer(options ProducerOptions) (Producer, error) {
	// Create a producer
	producer, err := newProducer(c, options)
	if err != nil {
		return nil, err
	}

	if reflect.ValueOf(c.server).IsNil() {
		return nil, newError(0, "Rmq server is nil")
	}
	// Create a topic in rocksmq, ignore if topic exists
	err = c.server.CreateTopic(options.Topic)
	if err != nil {
		return nil, err
	}
	c.producerOptions = append(c.producerOptions, options)

	return producer, nil
}

// Subscribe create a rocksmq consumer and start consume in a goroutine
func (c *client) Subscribe(options ConsumerOptions) (Consumer, error) {
	// Create a consumer
	if reflect.ValueOf(c.server).IsNil() {
		return nil, newError(0, "Rmq server is nil")
	}
	if exist, con := c.server.ExistConsumerGroup(options.Topic, options.SubscriptionName); exist {
		log.Debug("ConsumerGroup already existed", zap.Any("topic", options.Topic), zap.Any("SubscriptionName", options.SubscriptionName))
		consumer, err := getExistedConsumer(c, options, con.MsgMutex)
		if err != nil {
			return nil, err
		}
		if options.SubscriptionInitialPosition == SubscriptionPositionLatest {
			err = c.server.SeekToLatest(options.Topic, options.SubscriptionName)
			if err != nil {
				return nil, err
			}
		}
		return consumer, nil
	}
	consumer, err := newConsumer(c, options)
	if err != nil {
		return nil, err
	}

	// Create a consumergroup in rocksmq, raise error if consumergroup exists
	err = c.server.CreateConsumerGroup(options.Topic, options.SubscriptionName)
	if err != nil {
		return nil, err
	}

	if options.SubscriptionInitialPosition == SubscriptionPositionLatest {
		err = c.server.SeekToLatest(options.Topic, options.SubscriptionName)
		if err != nil {
			return nil, err
		}
	}
	// Register self in rocksmq server
	cons := &server.Consumer{
		Topic:     consumer.topic,
		GroupName: consumer.consumerName,
		MsgMutex:  consumer.msgMutex,
	}
	c.server.RegisterConsumer(cons)

	// Take messages from RocksDB and put it into consumer.Chan(),
	// trigger by consumer.MsgMutex which trigger by producer
	c.consumerOptions = append(c.consumerOptions, options)

	return consumer, nil
}

func (c *client) consume(consumer *consumer) {
	defer c.wg.Done()
	for {
		select {
		case <-c.closeCh:
			return
		case _, ok := <-consumer.MsgMutex():
			if !ok {
				// consumer MsgMutex closed, goroutine exit
				log.Debug("Consumer MsgMutex closed")
				return
			}

			for {
				n := cap(consumer.messageCh) - len(consumer.messageCh)
				if n < 100 { // batch min size
					n = 100
				}
				msgs, err := consumer.client.server.Consume(consumer.topic, consumer.consumerName, n)
				if err != nil {
					log.Debug("Consumer's goroutine cannot consume from (" + consumer.topic +
						"," + consumer.consumerName + "): " + err.Error())
					break
				}

				// no more msgs
				if len(msgs) == 0 {
					break
				}
				for _, msg := range msgs {
					consumer.messageCh <- ConsumerMessage{
						MsgID:   msg.MsgID,
						Payload: msg.Payload,
						Topic:   consumer.Topic(),
					}
				}
			}
		}
	}
}

// Close close the channel to notify rocksmq to stop operation and close rocksmq server
func (c *client) Close() {
	// TODO(yukun): Should call server.close() here?
	c.closeOnce.Do(func() {
		close(c.closeCh)
		c.wg.Wait()
		if c.server != nil {
			c.server.Close()
		}
		// Wait all consume goroutines exit
		c.consumerOptions = nil
	})
}
