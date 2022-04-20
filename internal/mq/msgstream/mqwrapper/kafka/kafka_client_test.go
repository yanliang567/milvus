package kafka

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/mq/msgstream/mqwrapper"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

var Params paramtable.BaseTable

func TestMain(m *testing.M) {
	Params.Init()
	exitCode := m.Run()
	os.Exit(exitCode)
}
func IntToBytes(n int) []byte {
	tmp := int32(n)
	bytesBuffer := bytes.NewBuffer([]byte{})
	binary.Write(bytesBuffer, common.Endian, tmp)
	return bytesBuffer.Bytes()
}
func BytesToInt(b []byte) int {
	bytesBuffer := bytes.NewBuffer(b)
	var tmp int32
	binary.Read(bytesBuffer, common.Endian, &tmp)
	return int(tmp)
}

// Consume1 will consume random messages and record the last MessageID it received
func Consume1(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, c chan mqwrapper.MessageID, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	// get random number between 1 ~ 5
	rand.Seed(time.Now().UnixNano())
	cnt := 1 + rand.Int()%5

	log.Info("Consume1 start")
	var msg mqwrapper.Message
	for i := 0; i < cnt; i++ {
		select {
		case <-ctx.Done():
			log.Info("Consume1 channel closed")
			return
		case msg = <-consumer.Chan():
			if msg == nil {
				return
			}

			log.Info("Consume1 RECV", zap.Any("v", BytesToInt(msg.Payload())))
			consumer.Ack(msg)
			(*total)++
		}
	}

	c <- msg.ID()
	log.Info("Consume1 randomly RECV", zap.Any("number", cnt))
	log.Info("Consume1 done")
}

// Consume2 will consume messages from specified MessageID
func Consume2(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, msgID mqwrapper.MessageID, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	err = consumer.Seek(msgID, true)
	assert.Nil(t, err)

	mm := <-consumer.Chan()
	consumer.Ack(mm)
	log.Info("skip the last received message", zap.Any("skip msg", mm.ID()))

	log.Info("Consume2 start")
	for {
		select {
		case <-ctx.Done():
			log.Info("Consume2 channel closed")
			return
		case msg, ok := <-consumer.Chan():
			if msg == nil || !ok {
				return
			}

			log.Info("Consume2 RECV", zap.Any("v", BytesToInt(msg.Payload())))
			consumer.Ack(msg)
			(*total)++
		}
	}
}

func Consume3(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	log.Info("Consume3 start")
	for {
		select {
		case <-ctx.Done():
			log.Info("Consume3 channel closed")
			return
		case msg, ok := <-consumer.Chan():
			if msg == nil || !ok {
				return
			}

			consumer.Ack(msg)
			(*total)++
			log.Info("Consume3 RECV", zap.Any("v", BytesToInt(msg.Payload())))
		}
	}
}

func TestKafkaClient_ConsumeWithAck(t *testing.T) {
	kc := createKafkaClient(t)
	defer kc.Close()
	assert.NotNil(t, kc)

	rand.Seed(time.Now().UnixNano())
	topic := fmt.Sprintf("test-topic-%d", rand.Int())
	subName := fmt.Sprintf("test-subname-%d", rand.Int())
	arr := []int{111, 222, 333, 444, 555, 666, 777}
	c := make(chan mqwrapper.MessageID, 1)

	ctx, cancel := context.WithCancel(context.Background())

	var total1 int
	var total2 int
	var total3 int

	producer := createProducer(t, kc, topic)
	defer producer.Close()
	produceData(ctx, t, producer, arr)
	time.Sleep(100 * time.Millisecond)

	ctx1, cancel1 := context.WithTimeout(ctx, 5*time.Second)
	defer cancel1()
	Consume1(ctx1, t, kc, topic, subName, c, &total1)

	lastMsgID := <-c
	log.Info("lastMsgID", zap.Any("lastMsgID", lastMsgID.(*kafkaID).messageID))

	ctx2, cancel2 := context.WithTimeout(ctx, 3*time.Second)
	Consume2(ctx2, t, kc, topic, subName, lastMsgID, &total2)
	cancel2()

	time.Sleep(5 * time.Second)
	ctx3, cancel3 := context.WithTimeout(ctx, 3*time.Second)
	Consume3(ctx3, t, kc, topic, subName, &total3)
	cancel3()

	cancel()
	assert.Equal(t, len(arr), total1+total2)

	assert.Equal(t, len(arr), total3)
}

func ConsumeFromEarliestToRandomPosition(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, c chan mqwrapper.MessageID, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	// get random number between 1 ~ 5
	rand.Seed(time.Now().UnixNano())
	cnt := 1 + rand.Int()%5

	log.Info("Consume1 channel start")
	var msg mqwrapper.Message
	for i := 0; i < cnt; i++ {
		select {
		case <-ctx.Done():
			log.Info("Consume1 channel closed")
			return
		case msg = <-consumer.Chan():
			if msg == nil {
				continue
			}

			v := BytesToInt(msg.Payload())
			log.Info("Consume1 RECV", zap.Any("v", v))
			(*total)++
		}
	}

	c <- &kafkaID{messageID: msg.ID().(*kafkaID).messageID}

	log.Info("Consume1 randomly RECV", zap.Any("number", cnt))
	log.Info("Consume1 done")
}

// Consume2 will consume messages from specified MessageID
func consumeFromSpecifiedPositionToEnd(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, msgID mqwrapper.MessageID, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	err = consumer.Seek(msgID, false)
	assert.Nil(t, err)

	log.Info("Consume2 start")
	for {
		select {
		case <-ctx.Done():
			log.Info("Consume2 channel closed")
			return
		case msg, ok := <-consumer.Chan():
			if msg == nil || !ok {
				return
			}

			v := BytesToInt(msg.Payload())
			log.Info("Consume2 RECV", zap.Any("v", v))
			(*total)++
		}
	}

}

func ConsumeFromEarliestToEndPosition(ctx context.Context, t *testing.T, kc *kafkaClient, topic string, subName string, total *int) {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            subName,
		BufSize:                     1024,
		SubscriptionInitialPosition: mqwrapper.SubscriptionPositionEarliest,
	})
	assert.Nil(t, err)
	assert.NotNil(t, consumer)
	defer consumer.Close()

	log.Info("Consume3 start")
	for {
		select {
		case <-ctx.Done():
			log.Info("Consume3 channel closed")
			return
		case msg, ok := <-consumer.Chan():
			if msg == nil || !ok {
				return
			}
			v := BytesToInt(msg.Payload())
			log.Info("Consume3 RECV", zap.Any("v", v))
			(*total)++
		}
	}
}

func TestKafkaClient_ConsumeNoAck(t *testing.T) {
	kc := createKafkaClient(t)
	defer kc.Close()
	assert.NotNil(t, kc)

	rand.Seed(time.Now().UnixNano())
	topic := fmt.Sprintf("test-topic-%d", rand.Int())
	subName := fmt.Sprintf("test-subname-%d", rand.Int())

	var total1 int
	var total2 int
	var total3 int

	arr := []int{111, 222, 333, 444, 555, 666, 777}
	ctx, cancel := context.WithCancel(context.Background())

	producer := createProducer(t, kc, topic)
	defer producer.Close()
	produceData(ctx, t, producer, arr)
	time.Sleep(100 * time.Millisecond)

	ctx1, cancel1 := context.WithTimeout(ctx, 5*time.Second)
	defer cancel1()

	c := make(chan mqwrapper.MessageID, 1)
	ConsumeFromEarliestToRandomPosition(ctx1, t, kc, topic, subName, c, &total1)

	// record the last received message id
	lastMsgID := <-c
	log.Info("msg", zap.Any("lastMsgID", lastMsgID))

	ctx2, cancel2 := context.WithTimeout(ctx, 5*time.Second)
	defer cancel2()
	consumeFromSpecifiedPositionToEnd(ctx2, t, kc, topic, subName, lastMsgID, &total2)

	ctx3, cancel3 := context.WithTimeout(ctx, 5*time.Second)
	defer cancel3()
	ConsumeFromEarliestToEndPosition(ctx3, t, kc, topic, subName, &total3)

	cancel()

	//TODO enable, it seems that ack is unavailable
	//assert.Equal(t, len(arr)*2, total1+total2)

	assert.Equal(t, len(arr), total3)
}

func TestKafkaClient_SeekPosition(t *testing.T) {
	kc := createKafkaClient(t)
	defer kc.Close()

	rand.Seed(time.Now().UnixNano())
	ctx := context.Background()
	topic := fmt.Sprintf("test-topic-%d", rand.Int())
	subName := fmt.Sprintf("test-subname-%d", rand.Int())

	producer := createProducer(t, kc, topic)
	defer producer.Close()

	data := []int{1, 2, 3}
	ids := produceData(ctx, t, producer, data)

	consumer := createConsumer(t, kc, topic, subName, mqwrapper.SubscriptionPositionLatest)
	defer consumer.Close()

	err := consumer.Seek(ids[2], true)
	assert.Nil(t, err)

	select {
	case msg := <-consumer.Chan():
		consumer.Ack(msg)
		assert.Equal(t, 3, BytesToInt(msg.Payload()))
	case <-time.After(10 * time.Second):
		assert.FailNow(t, "should not wait")
	}
}

func TestKafkaClient_EarliestMessageID(t *testing.T) {
	kafkaAddress, _ := Params.Load("_KafkaBrokerList")
	kc := NewKafkaClientInstance(kafkaAddress)
	defer kc.Close()

	mid := kc.EarliestMessageID()
	assert.NotNil(t, mid)
}

func createKafkaClient(t *testing.T) *kafkaClient {
	kafkaAddress, _ := Params.Load("_KafkaBrokerList")
	kc := NewKafkaClientInstance(kafkaAddress)
	assert.NotNil(t, kc)
	return kc
}

func createConsumer(t *testing.T,
	kc *kafkaClient,
	topic string,
	groupID string,
	initPosition mqwrapper.SubscriptionInitialPosition) mqwrapper.Consumer {
	consumer, err := kc.Subscribe(mqwrapper.ConsumerOptions{
		Topic:                       topic,
		SubscriptionName:            groupID,
		BufSize:                     1024,
		SubscriptionInitialPosition: initPosition,
	})
	assert.Nil(t, err)
	return consumer
}

func createProducer(t *testing.T, kc *kafkaClient, topic string) mqwrapper.Producer {
	producer, err := kc.CreateProducer(mqwrapper.ProducerOptions{Topic: topic})
	assert.Nil(t, err)
	assert.NotNil(t, producer)
	return producer
}

func produceData(ctx context.Context, t *testing.T, producer mqwrapper.Producer, arr []int) []mqwrapper.MessageID {
	var msgIDs []mqwrapper.MessageID
	for _, v := range arr {
		msg := &mqwrapper.ProducerMessage{
			Payload:    IntToBytes(v),
			Properties: map[string]string{},
		}
		msgID, err := producer.Send(ctx, msg)
		msgIDs = append(msgIDs, msgID)
		assert.Nil(t, err)
	}
	return msgIDs
}
