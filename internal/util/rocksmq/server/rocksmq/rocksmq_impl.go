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
	"errors"
	"fmt"
	"math"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/tecbot/gorocksdb"
	"go.uber.org/zap"

	rocksdbkv "github.com/milvus-io/milvus/internal/kv/rocksdb"
)

// UniqueID is the type of message ID
type UniqueID = typeutil.UniqueID

// RocksmqPageSize is the size of a message page, default 2GB
var RocksmqPageSize int64 = 2 << 30

// Const variable that will be used in rocksmqs
const (
	DefaultMessageID        = "-1"
	FixedChannelNameLen     = 320
	RocksDBLRUCacheCapacity = 0

	kvSuffix = "_meta_kv"

	MessageSizeTitle  = "message_size/"
	PageMsgSizeTitle  = "page_message_size/"
	TopicBeginIDTitle = "topic_begin_id/"
	BeginIDTitle      = "begin_id/"
	AckedTsTitle      = "acked_ts/"
	AckedSizeTitle    = "acked_size/"
	LastRetTsTitle    = "last_retention_ts/"

	CurrentIDSuffix = "current_id"
)

/**
 * @brief fill with '_' to ensure channel name fixed length
 */
func fixChannelName(name string) (string, error) {
	if len(name) > FixedChannelNameLen {
		return "", errors.New("Channel name exceeds limit")
	}

	nameBytes := make([]byte, FixedChannelNameLen-len(name))

	for i := 0; i < len(nameBytes); i++ {
		nameBytes[i] = byte('*')
	}

	return name + string(nameBytes), nil
}

/**
 * Combine key with fixed channel name and unique id
 */
func combKey(channelName string, id UniqueID) (string, error) {
	fixName, err := fixChannelName(channelName)
	if err != nil {
		return "", err
	}

	return fixName + "/" + strconv.FormatInt(id, 10), nil
}

/**
 * Construct current id
 */
func constructCurrentID(topicName, groupName string) string {
	return groupName + "/" + topicName + "/" + CurrentIDSuffix
}

/**
 * Construct table name and fixed channel name to be a key with length of FixedChannelNameLen,
 * used for meta infos
 */
func constructKey(metaName, topic string) (string, error) {
	// Check metaName/topic
	oldLen := len(metaName + topic)
	if oldLen > FixedChannelNameLen {
		return "", errors.New("Topic name exceeds limit")
	}

	nameBytes := make([]byte, FixedChannelNameLen-oldLen)

	for i := 0; i < len(nameBytes); i++ {
		nameBytes[i] = byte('*')
	}
	return metaName + topic + string(nameBytes), nil
}

func checkRetention() bool {
	return RocksmqRetentionTimeInMinutes != -1 && RocksmqRetentionSizeInMB != -1
}

var topicMu sync.Map = sync.Map{}

type rocksmq struct {
	store       *gorocksdb.DB
	kv          kv.BaseKV
	idAllocator allocator.GIDAllocator
	storeMu     *sync.Mutex
	consumers   sync.Map
	ackedMu     sync.Map

	retentionInfo *retentionInfo
}

// NewRocksMQ step:
// 1. New rocksmq instance based on rocksdb with name and rocksdbkv with kvname
// 2. Init retention info, load retention info to memory
// 3. Start retention goroutine
func NewRocksMQ(name string, idAllocator allocator.GIDAllocator) (*rocksmq, error) {
	bbto := gorocksdb.NewDefaultBlockBasedTableOptions()
	bbto.SetCacheIndexAndFilterBlocks(true)
	bbto.SetBlockCache(gorocksdb.NewLRUCache(RocksDBLRUCacheCapacity))
	opts := gorocksdb.NewDefaultOptions()
	opts.SetBlockBasedTableFactory(bbto)
	opts.SetCreateIfMissing(true)
	opts.SetPrefixExtractor(gorocksdb.NewFixedPrefixTransform(FixedChannelNameLen + 1))
	// opts.SetMaxOpenFiles(-1)

	db, err := gorocksdb.OpenDb(opts, name)
	if err != nil {
		return nil, err
	}

	kvName := name + kvSuffix
	kv, err := rocksdbkv.NewRocksdbKV(kvName)
	if err != nil {
		return nil, err
	}

	rmq := &rocksmq{
		store:       db,
		kv:          kv,
		idAllocator: idAllocator,
		storeMu:     &sync.Mutex{},
		consumers:   sync.Map{},
		ackedMu:     sync.Map{},
	}

	ri, err := initRetentionInfo(kv, db)
	if err != nil {
		return nil, err
	}
	rmq.retentionInfo = ri

	if checkRetention() {
		rmq.retentionInfo.startRetentionInfo()
	}

	return rmq, nil
}

// Close step:
// 1. Stop retention
// 2. Destroy all consumer groups and topics
// 3. Close rocksdb instance
func (rmq *rocksmq) Close() {
	rmq.stopRetention()
	rmq.storeMu.Lock()
	defer rmq.storeMu.Unlock()
	rmq.consumers.Range(func(k, v interface{}) bool {
		var topic string
		for _, consumer := range v.([]*Consumer) {
			err := rmq.DestroyConsumerGroup(consumer.Topic, consumer.GroupName)
			if err != nil {
				log.Warn("Failed to destroy consumer group in rocksmq!", zap.Any("topic", consumer.Topic), zap.Any("groupName", consumer.GroupName), zap.Any("error", err))
			}
			topic = consumer.Topic
		}
		if topic != "" {
			err := rmq.DestroyTopic(topic)
			if err != nil {
				log.Warn("Rocksmq DestroyTopic failed!", zap.Any("topic", topic), zap.Any("error", err))
			}
		}
		return true
	})
	rmq.store.Close()
}

func (rmq *rocksmq) stopRetention() {
	if rmq.retentionInfo != nil {
		rmq.retentionInfo.Stop()
	}
}

func (rmq *rocksmq) checkKeyExist(key string) bool {
	val, _ := rmq.kv.Load(key)
	return val != ""
}

// CreateTopic writes initialized messages for topic in rocksdb
func (rmq *rocksmq) CreateTopic(topicName string) error {
	start := time.Now()
	beginKey := topicName + "/begin_id"
	endKey := topicName + "/end_id"

	// Check if topic exist
	if rmq.checkKeyExist(beginKey) || rmq.checkKeyExist(endKey) {
		log.Warn("RocksMQ: " + beginKey + " or " + endKey + " existed.")
		return nil
	}
	// TODO change rmq kv save logic into a batch
	err := rmq.kv.Save(beginKey, "0")
	if err != nil {
		return err
	}

	err = rmq.kv.Save(endKey, "0")
	if err != nil {
		return err
	}
	if _, ok := topicMu.Load(topicName); !ok {
		topicMu.Store(topicName, new(sync.Mutex))
	}
	if _, ok := rmq.ackedMu.Load(topicName); !ok {
		rmq.ackedMu.Store(topicName, new(sync.Mutex))
	}

	// Initialize retention infos
	// Initialize acked size to 0 for topic
	ackedSizeKey := AckedSizeTitle + topicName
	err = rmq.kv.Save(ackedSizeKey, "0")
	if err != nil {
		return err
	}

	// Initialize topic begin id to defaultMessageID
	topicBeginIDKey := TopicBeginIDTitle + topicName
	err = rmq.kv.Save(topicBeginIDKey, DefaultMessageID)
	if err != nil {
		return err
	}

	// Initialize topic message size to 0
	msgSizeKey := MessageSizeTitle + topicName
	err = rmq.kv.Save(msgSizeKey, "0")
	if err != nil {
		return err
	}

	rmq.retentionInfo.mutex.Lock()
	defer rmq.retentionInfo.mutex.Unlock()
	rmq.retentionInfo.topics.Store(topicName, time.Now().Unix())
	log.Debug("Rocksmq create topic successfully ", zap.String("topic", topicName), zap.Int64("elapsed", time.Since(start).Milliseconds()))
	return nil
}

// DestroyTopic removes messages for topic in rocksdb
func (rmq *rocksmq) DestroyTopic(topicName string) error {
	start := time.Now()
	beginKey := topicName + "/begin_id"
	endKey := topicName + "/end_id"

	err := rmq.kv.Remove(beginKey)
	if err != nil {
		log.Debug("RocksMQ: failed to remove key <" + beginKey + ">.")
		return err
	}

	err = rmq.kv.Remove(endKey)
	if err != nil {
		log.Debug("RocksMQ: failed to remove key <" + endKey + ">.")
		return err
	}

	rmq.consumers.Delete(topicName)

	ackedSizeKey := AckedSizeTitle + topicName
	err = rmq.kv.Remove(ackedSizeKey)
	if err != nil {
		return err
	}
	topicBeginIDKey := TopicBeginIDTitle + topicName
	err = rmq.kv.Remove(topicBeginIDKey)
	if err != nil {
		return err
	}

	// just for clean up old topics, for new topics this is not required
	lastRetTsKey := LastRetTsTitle + topicName
	err = rmq.kv.Remove(lastRetTsKey)
	if err != nil {
		return err
	}
	msgSizeKey := MessageSizeTitle + topicName
	err = rmq.kv.Remove(msgSizeKey)
	if err != nil {
		return err
	}

	topicMu.Delete(topicName)
	// clean up retention info
	rmq.retentionInfo.topics.Delete(topicName)
	log.Debug("Rocksmq destroy topic successfully ", zap.String("topic", topicName), zap.Int64("elapsed", time.Since(start).Milliseconds()))
	return nil
}

// ExistConsumerGroup check if a consumer exists and return the existed consumer
func (rmq *rocksmq) ExistConsumerGroup(topicName, groupName string) (bool, *Consumer) {
	key := constructCurrentID(topicName, groupName)
	if rmq.checkKeyExist(key) {
		if vals, ok := rmq.consumers.Load(topicName); ok {
			for _, v := range vals.([]*Consumer) {
				if v.GroupName == groupName {
					return true, v
				}
			}
		}
	}
	return false, nil
}

// CreateConsumerGroup creates an nonexistent consumer group for topic
func (rmq *rocksmq) CreateConsumerGroup(topicName, groupName string) error {
	start := time.Now()
	key := constructCurrentID(topicName, groupName)
	if rmq.checkKeyExist(key) {
		log.Debug("RocksMQ: " + key + " existed.")
		return nil
	}
	err := rmq.kv.Save(key, DefaultMessageID)
	if err != nil {
		return err
	}
	log.Debug("Rocksmq create consumer group successfully ", zap.String("topic", topicName),
		zap.String("group", groupName),
		zap.Int64("elapsed", time.Since(start).Milliseconds()))
	return nil
}

// RegisterConsumer registers a consumer in rocksmq consumers
func (rmq *rocksmq) RegisterConsumer(consumer *Consumer) {
	start := time.Now()
	if vals, ok := rmq.consumers.Load(consumer.Topic); ok {
		for _, v := range vals.([]*Consumer) {
			if v.GroupName == consumer.GroupName {
				return
			}
		}
		consumers := vals.([]*Consumer)
		consumers = append(consumers, consumer)
		rmq.consumers.Store(consumer.Topic, consumers)
	} else {
		consumers := make([]*Consumer, 1)
		consumers[0] = consumer
		rmq.consumers.Store(consumer.Topic, consumers)
	}
	log.Debug("Rocksmq register consumer successfully ", zap.String("topic", consumer.Topic), zap.Int64("elapsed", time.Since(start).Milliseconds()))
}

// DestroyConsumerGroup removes a consumer group from rocksdb_kv
func (rmq *rocksmq) DestroyConsumerGroup(topicName, groupName string) error {
	start := time.Now()
	ll, ok := topicMu.Load(topicName)
	if !ok {
		return fmt.Errorf("topic name = %s not exist", topicName)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return fmt.Errorf("get mutex failed, topic name = %s", topicName)
	}
	lock.Lock()
	defer lock.Unlock()
	key := constructCurrentID(topicName, groupName)

	err := rmq.kv.Remove(key)
	if err != nil {
		return err
	}
	if vals, ok := rmq.consumers.Load(topicName); ok {
		consumers := vals.([]*Consumer)
		for index, v := range consumers {
			if v.GroupName == groupName {
				close(v.MsgMutex)
				consumers = append(consumers[:index], consumers[index+1:]...)
				rmq.consumers.Store(topicName, consumers)
				break
			}
		}
	}
	log.Debug("Rocksmq destroy consumer group successfully ", zap.String("topic", topicName),
		zap.String("group", groupName),
		zap.Int64("elapsed", time.Since(start).Milliseconds()))
	return nil
}

// Produce produces messages for topic and updates page infos for retention
func (rmq *rocksmq) Produce(topicName string, messages []ProducerMessage) ([]UniqueID, error) {
	ll, ok := topicMu.Load(topicName)
	if !ok {
		return []UniqueID{}, fmt.Errorf("topic name = %s not exist", topicName)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return []UniqueID{}, fmt.Errorf("get mutex failed, topic name = %s", topicName)
	}
	lock.Lock()
	defer lock.Unlock()

	msgLen := len(messages)
	idStart, idEnd, err := rmq.idAllocator.Alloc(uint32(msgLen))

	if err != nil {
		log.Debug("RocksMQ: alloc id failed.")
		return []UniqueID{}, err
	}

	if UniqueID(msgLen) != idEnd-idStart {
		return []UniqueID{}, errors.New("Obtained id length is not equal that of message")
	}

	/* Step I: Insert data to store system */
	batch := gorocksdb.NewWriteBatch()
	defer batch.Destroy()
	msgSizes := make(map[UniqueID]int64)
	msgIDs := make([]UniqueID, msgLen)
	for i := 0; i < msgLen && idStart+UniqueID(i) < idEnd; i++ {
		msgID := idStart + UniqueID(i)
		key, err := combKey(topicName, msgID)
		if err != nil {
			return []UniqueID{}, err
		}

		batch.Put([]byte(key), messages[i].Payload)
		msgIDs[i] = msgID
		msgSizes[msgID] = int64(len(messages[i].Payload))
	}

	opts := gorocksdb.NewDefaultWriteOptions()
	defer opts.Destroy()
	err = rmq.store.Write(opts, batch)
	if err != nil {
		log.Debug("RocksMQ: write batch failed")
		return []UniqueID{}, err
	}

	/* Step II: Update meta data to kv system */
	kvChannelBeginID := topicName + "/begin_id"
	beginIDValue, err := rmq.kv.Load(kvChannelBeginID)
	if err != nil {
		log.Debug("RocksMQ: load " + kvChannelBeginID + " failed")
		return []UniqueID{}, err
	}

	kvValues := make(map[string]string)

	if beginIDValue == "0" {
		log.Debug("RocksMQ: overwrite " + kvChannelBeginID + " with " + strconv.FormatInt(idStart, 10))
		kvValues[kvChannelBeginID] = strconv.FormatInt(idStart, 10)
	}

	kvChannelEndID := topicName + "/end_id"
	kvValues[kvChannelEndID] = strconv.FormatInt(idEnd, 10)

	err = rmq.kv.MultiSave(kvValues)
	if err != nil {
		log.Debug("RocksMQ: multisave failed")
		return []UniqueID{}, err
	}

	if vals, ok := rmq.consumers.Load(topicName); ok {
		for _, v := range vals.([]*Consumer) {
			select {
			case v.MsgMutex <- struct{}{}:
				continue
			default:
				continue
			}
		}
	}

	// Update message page info
	// TODO(yukun): Should this be in a go routine
	err = rmq.updatePageInfo(topicName, msgIDs, msgSizes)
	if err != nil {
		return []UniqueID{}, err
	}
	return msgIDs, nil
}

func (rmq *rocksmq) updatePageInfo(topicName string, msgIDs []UniqueID, msgSizes map[UniqueID]int64) error {
	msgSizeKey := MessageSizeTitle + topicName
	msgSizeVal, err := rmq.kv.Load(msgSizeKey)
	if err != nil {
		return err
	}
	curMsgSize, err := strconv.ParseInt(msgSizeVal, 10, 64)
	if err != nil {
		return err
	}
	fixedPageSizeKey, err := constructKey(PageMsgSizeTitle, topicName)
	if err != nil {
		return err
	}
	for _, id := range msgIDs {
		msgSize := msgSizes[id]
		if curMsgSize+msgSize > RocksmqPageSize {
			// Current page is full
			newPageSize := curMsgSize + msgSize
			pageEndID := id
			// Update page message size for current page. key is page end ID
			pageMsgSizeKey := fixedPageSizeKey + "/" + strconv.FormatInt(pageEndID, 10)
			err := rmq.kv.Save(pageMsgSizeKey, strconv.FormatInt(newPageSize, 10))
			if err != nil {
				return err
			}

			// if pageInfo, ok := rmq.retentionInfo.pageInfo.Load(topicName); ok {
			// 	pageInfo.(*topicPageInfo).pageEndID = append(pageInfo.(*topicPageInfo).pageEndID, pageEndID)
			// 	pageInfo.(*topicPageInfo).pageMsgSize[pageEndID] = newPageSize
			// 	rmq.retentionInfo.pageInfo.Store(topicName, pageInfo)
			// }

			// Update message size to 0
			err = rmq.kv.Save(msgSizeKey, strconv.FormatInt(0, 10))
			if err != nil {
				return err
			}
			curMsgSize = 0
		} else {
			curMsgSize += msgSize
			// Update message size to current message size
			err := rmq.kv.Save(msgSizeKey, strconv.FormatInt(curMsgSize, 10))
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// Consume steps:
// 1. Consume n messages from rocksdb
// 2. Update current_id to the last consumed message
// 3. Update ack informations in rocksdb
func (rmq *rocksmq) Consume(topicName string, groupName string, n int) ([]ConsumerMessage, error) {
	ll, ok := topicMu.Load(topicName)
	if !ok {
		return nil, fmt.Errorf("topic name = %s not exist", topicName)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return nil, fmt.Errorf("get mutex failed, topic name = %s", topicName)
	}
	lock.Lock()
	defer lock.Unlock()

	metaKey := constructCurrentID(topicName, groupName)
	currentID, err := rmq.kv.Load(metaKey)
	if err != nil {
		log.Debug("RocksMQ: load " + metaKey + " failed")
		return nil, err
	}

	readOpts := gorocksdb.NewDefaultReadOptions()
	defer readOpts.Destroy()
	readOpts.SetPrefixSameAsStart(true)
	iter := rmq.store.NewIterator(readOpts)
	defer iter.Close()

	consumerMessage := make([]ConsumerMessage, 0, n)

	fixChanName, err := fixChannelName(topicName)
	if err != nil {
		log.Debug("RocksMQ: fixChannelName " + topicName + " failed")
		return nil, err
	}
	dataKey := fixChanName + "/" + currentID

	// msgID is DefaultMessageID means this is the first consume operation
	// currentID may be not valid if the deprecated values has been removed, when
	// we move currentID to first location.
	// Note that we assume currentId is always correct and not larger than the latest endID.
	if iter.Seek([]byte(dataKey)); currentID != DefaultMessageID && iter.Valid() {
		iter.Next()
	} else {
		newKey := fixChanName + "/"
		iter.Seek([]byte(newKey))
	}

	offset := 0
	for ; iter.Valid() && offset < n; iter.Next() {
		key := iter.Key()
		val := iter.Value()
		strKey := string(key.Data())
		key.Free()
		offset++
		msgID, err := strconv.ParseInt(strKey[FixedChannelNameLen+1:], 10, 64)
		if err != nil {
			log.Debug("RocksMQ: parse int " + strKey[FixedChannelNameLen+1:] + " failed")
			val.Free()
			return nil, err
		}
		msg := ConsumerMessage{
			MsgID: msgID,
		}
		origData := val.Data()
		dataLen := len(origData)
		if dataLen == 0 {
			msg.Payload = nil
		} else {
			msg.Payload = make([]byte, dataLen)
			copy(msg.Payload, origData)
		}
		consumerMessage = append(consumerMessage, msg)
		val.Free()
	}

	// When already consume to last mes, an empty slice will be returned
	if len(consumerMessage) == 0 {
		// log.Debug("RocksMQ: consumerMessage is empty")
		return consumerMessage, nil
	}

	consumedIDs := make([]UniqueID, 0, len(consumerMessage))
	msgSize := make([]int64, 0, len(consumerMessage))
	for _, msg := range consumerMessage {
		consumedIDs = append(consumedIDs, msg.MsgID)
		msgSize = append(msgSize, int64(len(msg.Payload)))
	}
	newID := consumedIDs[len(consumedIDs)-1]
	err = rmq.seek(topicName, groupName, newID)
	if err != nil {
		log.Debug("RocksMQ: Seek(" + groupName + "," + topicName + "," + strconv.FormatInt(newID, 10) + ") failed")
		return nil, err
	}

	go rmq.updateAckedInfo(topicName, groupName, consumedIDs, msgSize)

	return consumerMessage, nil
}

// seek is used for internal call without the topicMu
func (rmq *rocksmq) seek(topicName string, groupName string, msgID UniqueID) error {
	rmq.storeMu.Lock()
	defer rmq.storeMu.Unlock()
	key := constructCurrentID(topicName, groupName)
	if !rmq.checkKeyExist(key) {
		log.Warn("RocksMQ: channel " + key + " not exists")
		return fmt.Errorf("ConsumerGroup %s, channel %s not exists", groupName, topicName)
	}

	storeKey, err := combKey(topicName, msgID)
	if err != nil {
		log.Warn("RocksMQ: combKey(" + topicName + "," + strconv.FormatInt(msgID, 10) + ") failed")
		return err
	}

	opts := gorocksdb.NewDefaultReadOptions()
	defer opts.Destroy()
	val, err := rmq.store.Get(opts, []byte(storeKey))
	defer val.Free()
	if err != nil {
		log.Warn("RocksMQ: get " + storeKey + " failed")
		return err
	}

	/* Step II: Save current_id in kv */
	err = rmq.kv.Save(key, strconv.FormatInt(msgID, 10))
	if err != nil {
		log.Warn("RocksMQ: save " + key + " failed")
		return err
	}

	return nil
}

// Seek updates the current id to the given msgID
func (rmq *rocksmq) Seek(topicName string, groupName string, msgID UniqueID) error {
	/* Step I: Check if key exists */
	ll, ok := topicMu.Load(topicName)
	if !ok {
		return fmt.Errorf("topic name = %s not exist", topicName)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return fmt.Errorf("get mutex failed, topic name = %s", topicName)
	}
	lock.Lock()
	defer lock.Unlock()

	return rmq.seek(topicName, groupName, msgID)
}

// SeekToLatest updates current id to the msg id of latest message
func (rmq *rocksmq) SeekToLatest(topicName, groupName string) error {
	rmq.storeMu.Lock()
	defer rmq.storeMu.Unlock()
	key := constructCurrentID(topicName, groupName)
	if !rmq.checkKeyExist(key) {
		log.Debug("RocksMQ: channel " + key + " not exists")
		return fmt.Errorf("ConsumerGroup %s, channel %s not exists", groupName, topicName)
	}

	readOpts := gorocksdb.NewDefaultReadOptions()
	defer readOpts.Destroy()
	readOpts.SetPrefixSameAsStart(true)
	iter := rmq.store.NewIterator(readOpts)
	defer iter.Close()

	fixChanName, _ := fixChannelName(topicName)
	iter.Seek([]byte(fixChanName + "/"))
	iKey := iter.Key()
	// iter.SeekToLast bypass prefix limitation
	// use for range until iterator invalid for now
	if iter.Valid() {
		iter.Next()
		for iter.Valid() {
			iKey.Free()
			iKey = iter.Key()
			iter.Next()
		}
	} else {
		// In this case there are no messages, so shouldn't return error
		return nil
	}
	if iKey == nil {
		return nil
	}

	seekMsgID := string(iKey.Data()) // bytes to string, copy
	iKey.Free()

	msgID, err := strconv.ParseInt(seekMsgID[FixedChannelNameLen+1:], 10, 64)
	if err != nil {
		return err
	}
	err = rmq.kv.Save(key, strconv.FormatInt(msgID, 10))
	return err
}

// Notify sends a mutex in MsgMutex channel to tell consumers to consume
func (rmq *rocksmq) Notify(topicName, groupName string) {
	if vals, ok := rmq.consumers.Load(topicName); ok {
		for _, v := range vals.([]*Consumer) {
			if v.GroupName == groupName {
				select {
				case v.MsgMutex <- struct{}{}:
					continue
				default:
					continue
				}
			}
		}
	}
}

// updateAckedInfo update acked informations for retention after consume
func (rmq *rocksmq) updateAckedInfo(topicName, groupName string, ids []UniqueID, msgSize []int64) error {
	if len(ids) == 0 {
		return nil
	}
	ll, ok := topicMu.Load(topicName)
	if !ok {
		return fmt.Errorf("topic name = %s not exist", topicName)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return fmt.Errorf("get mutex failed, topic name = %s", topicName)
	}
	lock.Lock()
	defer lock.Unlock()

	lastID := ids[len(ids)-1]

	fixedBeginIDKey, err := constructKey(BeginIDTitle, topicName)
	if err != nil {
		return err
	}
	// Update begin_id for the consumer_group
	beginIDKey := fixedBeginIDKey + "/" + groupName
	err = rmq.kv.Save(beginIDKey, strconv.FormatInt(lastID, 10))
	if err != nil {
		return err
	}

	// Update begin_id for topic
	if vals, ok := rmq.consumers.Load(topicName); ok {
		var minBeginID int64 = math.MaxInt64
		for _, v := range vals.([]*Consumer) {
			curBeginIDKey := fixedBeginIDKey + "/" + v.GroupName
			curBeginIDVal, err := rmq.kv.Load(curBeginIDKey)
			if err != nil {
				return err
			}
			curBeginID, err := strconv.ParseInt(curBeginIDVal, 10, 64)
			if err != nil {
				return err
			}
			if curBeginID < minBeginID {
				minBeginID = curBeginID
			}
		}
		topicBeginIDKey := TopicBeginIDTitle + topicName
		err = rmq.kv.Save(topicBeginIDKey, strconv.FormatInt(minBeginID, 10))
		if err != nil {
			return err
		}

		// Update acked info for msg of begin id
		fixedAckedTsKey, err := constructKey(AckedTsTitle, topicName)
		if err != nil {
			return err
		}

		ts := strconv.FormatInt(time.Now().Unix(), 10)
		// current behavior is to ack all safe msgID(before minBeginID)
		// TODO @silverxia @yukun ack only page separator msg id
		ackMsgKvs := make(map[string]string)
		totalAckMsgSize := int64(0)
		for i, id := range ids {
			// depends on the ids are monotonically increasing
			if id <= minBeginID {
				totalAckMsgSize += msgSize[i]
				key := path.Join(fixedAckedTsKey, strconv.FormatInt(id, 10))
				ackMsgKvs[key] = ts
			}
		}
		err = rmq.kv.MultiSave(ackMsgKvs)
		if err != nil {
			return err
		}
		if minBeginID == lastID {
			// Means the begin_id of topic update to newID, so needs to update acked size
			ackedSizeKey := AckedSizeTitle + topicName
			ackedSizeVal, err := rmq.kv.Load(ackedSizeKey)
			if err != nil {
				return err
			}
			ackedSize, err := strconv.ParseInt(ackedSizeVal, 10, 64)
			if err != nil {
				return err
			}
			ackedSize += totalAckMsgSize
			err = rmq.kv.Save(ackedSizeKey, strconv.FormatInt(ackedSize, 10))
			if err != nil {
				return err
			}
		}
	}
	return nil
}
