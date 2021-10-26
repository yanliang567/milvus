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
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	rocksdbkv "github.com/milvus-io/milvus/internal/kv/rocksdb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/tecbot/gorocksdb"
	"go.uber.org/zap"
)

// RocksmqRetentionTimeInMinutes is the time of retention
var RocksmqRetentionTimeInMinutes int64

// RocksmqRetentionSizeInMB is the size of retention
var RocksmqRetentionSizeInMB int64

// Const value that used to convert unit
const (
	MB     = 2 << 20
	MINUTE = 60
)

// TickerTimeInSeconds is the time of expired check, default 10 minutes
var TickerTimeInSeconds int64 = 10 * MINUTE

type topicPageInfo struct {
	pageEndID   []UniqueID
	pageMsgSize map[UniqueID]int64
}

type topicAckedInfo struct {
	topicBeginID UniqueID
	// TODO(yukun): may need to delete ackedTs
	ackedTs   map[UniqueID]UniqueID
	ackedSize int64
}

type retentionInfo struct {
	topics []string
	// pageInfo  map[string]*topicPageInfo
	pageInfo sync.Map
	// ackedInfo map[string]*topicAckedInfo
	ackedInfo sync.Map
	// Key is last_retention_time/${topic}
	// lastRetentionTime map[string]int64
	lastRetentionTime sync.Map

	mutex sync.RWMutex

	kv *rocksdbkv.RocksdbKV
	db *gorocksdb.DB

	closeCh   chan struct{}
	closeWg   sync.WaitGroup
	closeOnce sync.Once
}

// Interface LoadWithPrefix() in rocksdbkv needs to close db instance first and then reopen,
// which will cause crash when other goroutines operate the db instance. So here implement a
// prefixLoad without reopen db instance.
func prefixLoad(db *gorocksdb.DB, prefix string) ([]string, []string, error) {
	if db == nil {
		return nil, nil, errors.New("Rocksdb instance is nil when do prefixLoad")
	}
	readOpts := gorocksdb.NewDefaultReadOptions()
	defer readOpts.Destroy()
	readOpts.SetPrefixSameAsStart(true)
	iter := db.NewIterator(readOpts)
	defer iter.Close()
	keys := make([]string, 0)
	values := make([]string, 0)
	iter.Seek([]byte(prefix))
	for ; iter.Valid(); iter.Next() {
		key := iter.Key()
		value := iter.Value()
		keys = append(keys, string(key.Data()))
		key.Free()
		values = append(values, string(value.Data()))
		value.Free()
	}
	return keys, values, nil
}

func initRetentionInfo(kv *rocksdbkv.RocksdbKV, db *gorocksdb.DB) (*retentionInfo, error) {
	ri := &retentionInfo{
		topics:            make([]string, 0),
		pageInfo:          sync.Map{},
		ackedInfo:         sync.Map{},
		lastRetentionTime: sync.Map{},
		mutex:             sync.RWMutex{},
		kv:                kv,
		db:                db,
		closeCh:           make(chan struct{}),
		closeWg:           sync.WaitGroup{},
	}
	// Get topic from topic begin id
	beginIDKeys, _, err := ri.kv.LoadWithPrefix(TopicBeginIDTitle)
	if err != nil {
		return nil, err
	}
	for _, key := range beginIDKeys {
		topic := key[len(TopicBeginIDTitle):]
		ri.topics = append(ri.topics, topic)
		topicMu.Store(topic, new(sync.Mutex))
	}
	return ri, nil
}

// Before do retention, load retention info from rocksdb to retention info structure in goroutines.
// Because loadRetentionInfo may need some time, so do this asynchronously. Finally start retention goroutine.
func (ri *retentionInfo) startRetentionInfo() {
	// var wg sync.WaitGroup
	ri.kv.ResetPrefixLength(FixedChannelNameLen)
	// for _, topic := range ri.topics {
	// log.Debug("Start load retention info", zap.Any("topic", topic))
	// Load all page infos
	// wg.Add(1)
	// go ri.loadRetentionInfo(topic, &wg)
	// }
	// wg.Wait()
	// log.Debug("Finish load retention info, start retention")
	ri.closeWg.Add(1)
	go ri.retention()
}

// Read retention infos from rocksdb so that retention check can be done based on memory data
func (ri *retentionInfo) loadRetentionInfo(topic string, wg *sync.WaitGroup) {
	// TODO(yukun): If there needs to add lock
	// ll, ok := topicMu.Load(topic)
	// if !ok {
	// 	return fmt.Errorf("topic name = %s not exist", topic)
	// }
	// lock, ok := ll.(*sync.Mutex)
	// if !ok {
	// 	return fmt.Errorf("get mutex failed, topic name = %s", topic)
	// }
	// lock.Lock()
	// defer lock.Unlock()
	defer wg.Done()
	pageEndID := make([]UniqueID, 0)
	pageMsgSize := make(map[int64]UniqueID)

	fixedPageSizeKey, err := constructKey(PageMsgSizeTitle, topic)
	if err != nil {
		log.Debug("ConstructKey failed", zap.Any("error", err))
		return
	}
	pageMsgSizePrefix := fixedPageSizeKey + "/"
	pageMsgSizeKeys, pageMsgSizeVals, err := prefixLoad(ri.kv.DB, pageMsgSizePrefix)
	if err != nil {
		log.Debug("PrefixLoad failed", zap.Any("error", err))
		return
	}
	for i, key := range pageMsgSizeKeys {
		endID, err := strconv.ParseInt(key[FixedChannelNameLen+1:], 10, 64)
		if err != nil {
			log.Debug("ParseInt failed", zap.Any("error", err))
			return
		}
		pageEndID = append(pageEndID, endID)

		msgSize, err := strconv.ParseInt(pageMsgSizeVals[i], 10, 64)
		if err != nil {
			log.Debug("ParseInt failed", zap.Any("error", err))
			return
		}
		pageMsgSize[endID] = msgSize
	}
	topicPageInfo := &topicPageInfo{
		pageEndID:   pageEndID,
		pageMsgSize: pageMsgSize,
	}

	// Load all acked infos
	ackedTs := make(map[UniqueID]UniqueID)

	topicBeginIDKey := TopicBeginIDTitle + topic
	topicBeginIDVal, err := ri.kv.Load(topicBeginIDKey)
	if err != nil {
		return
	}
	topicBeginID, err := strconv.ParseInt(topicBeginIDVal, 10, 64)
	if err != nil {
		log.Debug("ParseInt failed", zap.Any("error", err))
		return
	}

	ackedTsPrefix, err := constructKey(AckedTsTitle, topic)
	if err != nil {
		log.Debug("ConstructKey failed", zap.Any("error", err))
		return
	}
	keys, vals, err := prefixLoad(ri.kv.DB, ackedTsPrefix)
	if err != nil {
		log.Debug("PrefixLoad failed", zap.Any("error", err))
		return
	}

	for i, key := range keys {
		offset := FixedChannelNameLen + 1
		ackedID, err := strconv.ParseInt((key)[offset:], 10, 64)
		if err != nil {
			log.Debug("RocksMQ: parse int " + key[offset:] + " failed")
			return
		}

		ts, err := strconv.ParseInt(vals[i], 10, 64)
		if err != nil {
			return
		}
		ackedTs[ackedID] = ts
	}

	ackedSizeKey := AckedSizeTitle + topic
	ackedSizeVal, err := ri.kv.Load(ackedSizeKey)
	if err != nil {
		log.Debug("Load failed", zap.Any("error", err))
		return
	}
	var ackedSize int64
	if ackedSizeVal == "" {
		ackedSize = 0
	} else {
		ackedSize, err = strconv.ParseInt(ackedSizeVal, 10, 64)
		if err != nil {
			log.Debug("PrefixLoad failed", zap.Any("error", err))
			return
		}
	}

	ackedInfo := &topicAckedInfo{
		topicBeginID: topicBeginID,
		ackedTs:      ackedTs,
		ackedSize:    ackedSize,
	}

	//Load last retention timestamp
	lastRetentionTsKey := LastRetTsTitle + topic
	lastRetentionTsVal, err := ri.kv.Load(lastRetentionTsKey)
	if err != nil {
		log.Debug("Load failed", zap.Any("error", err))
		return
	}
	var lastRetentionTs int64
	if lastRetentionTsVal == "" {
		lastRetentionTs = math.MaxInt64
	} else {
		lastRetentionTs, err = strconv.ParseInt(lastRetentionTsVal, 10, 64)
		if err != nil {
			log.Debug("ParseInt failed", zap.Any("error", err))
			return
		}
	}

	ri.ackedInfo.Store(topic, ackedInfo)
	ri.pageInfo.Store(topic, topicPageInfo)
	ri.lastRetentionTime.Store(topic, lastRetentionTs)
}

// retention do time ticker and trigger retention check and operation for each topic
func (ri *retentionInfo) retention() error {
	log.Debug("Rocksmq retention goroutine start!")
	// Do retention check every 6s
	ticker := time.NewTicker(time.Duration(atomic.LoadInt64(&TickerTimeInSeconds) * int64(time.Second)))
	defer ri.closeWg.Done()

	for {
		select {
		case <-ri.closeCh:
			log.Debug("Rocksmq retention finish!")
			return nil
		case t := <-ticker.C:
			timeNow := t.Unix()
			checkTime := atomic.LoadInt64(&RocksmqRetentionTimeInMinutes) * MINUTE / 10
			log.Debug("In ticker: ", zap.Any("ticker", timeNow))
			ri.mutex.RLock()
			for _, topic := range ri.topics {
				lastRetentionTsKey := LastRetTsTitle + topic
				lastRetentionTsVal, err := ri.kv.Load(lastRetentionTsKey)
				if err != nil || lastRetentionTsVal == "" {
					log.Warn("Can't get lastRetentionTs", zap.Any("lastRetentionTsKey", lastRetentionTsKey))
					continue
				}
				lastRetentionTs, err := strconv.ParseInt(lastRetentionTsVal, 10, 64)
				if err != nil {
					log.Warn("Can't parse lastRetentionTsVal to int", zap.Any("lastRetentionTsKey", lastRetentionTsKey))
					continue
				}
				if lastRetentionTs+checkTime < timeNow {
					err := ri.newExpiredCleanUp(topic)
					if err != nil {
						log.Warn("Retention expired clean failed", zap.Any("error", err))
					}
				}
			}
			ri.mutex.RUnlock()
		}
	}
}

func (ri *retentionInfo) Stop() {
	ri.closeOnce.Do(func() {
		close(ri.closeCh)
		ri.closeWg.Wait()
	})
}

func (ri *retentionInfo) newExpiredCleanUp(topic string) error {
	log.Debug("Timeticker triggers an expiredCleanUp task for topic: " + topic)
	ll, ok := topicMu.Load(topic)
	if !ok {
		return fmt.Errorf("topic name = %s not exist", topic)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return fmt.Errorf("get mutex failed, topic name = %s", topic)
	}
	lock.Lock()
	defer lock.Unlock()

	var deletedAckedSize int64 = 0
	var startID UniqueID
	var endID UniqueID
	var pageStartID UniqueID = 0
	var err error

	fixedAckedTsKey, _ := constructKey(AckedTsTitle, topic)

	pageReadOpts := gorocksdb.NewDefaultReadOptions()
	defer pageReadOpts.Destroy()
	pageReadOpts.SetPrefixSameAsStart(true)
	pageIter := ri.kv.DB.NewIterator(pageReadOpts)
	defer pageIter.Close()
	pageMsgPrefix, _ := constructKey(PageMsgSizeTitle, topic)
	pageIter.Seek([]byte(pageMsgPrefix))
	if pageIter.Valid() {
		pageStartID, err = strconv.ParseInt(string(pageIter.Key().Data())[FixedChannelNameLen+1:], 10, 64)
		if err != nil {
			return err
		}

		for ; pageIter.Valid(); pageIter.Next() {
			pKey := pageIter.Key()
			pageID, err := strconv.ParseInt(string(pKey.Data())[FixedChannelNameLen+1:], 10, 64)
			if pKey != nil {
				pKey.Free()
			}
			if err != nil {
				return err
			}

			ackedTsKey := fixedAckedTsKey + "/" + strconv.FormatInt(pageID, 10)
			ackedTsVal, err := ri.kv.Load(ackedTsKey)
			if err != nil {
				return err
			}
			ackedTs, err := strconv.ParseInt(ackedTsVal, 10, 64)
			if err != nil {
				return err
			}
			if msgTimeExpiredCheck(ackedTs) {
				endID = pageID
				pValue := pageIter.Value()
				size, err := strconv.ParseInt(string(pValue.Data()), 10, 64)
				if pValue != nil {
					pValue.Free()
				}
				if err != nil {
					return err
				}
				deletedAckedSize += size
			} else {
				break
			}
		}
	}

	pageEndID := endID

	ackedReadOpts := gorocksdb.NewDefaultReadOptions()
	defer ackedReadOpts.Destroy()
	ackedReadOpts.SetPrefixSameAsStart(true)
	ackedIter := ri.kv.DB.NewIterator(ackedReadOpts)
	defer ackedIter.Close()
	if err != nil {
		return err
	}
	ackedIter.Seek([]byte(fixedAckedTsKey))
	if !ackedIter.Valid() {
		return nil
	}

	startID, err = strconv.ParseInt(string(ackedIter.Key().Data())[FixedChannelNameLen+1:], 10, 64)
	if err != nil {
		return err
	}
	if endID > startID {
		newPos := fixedAckedTsKey + "/" + strconv.FormatInt(endID, 10)
		ackedIter.Seek([]byte(newPos))
	}

	for ; ackedIter.Valid(); ackedIter.Next() {
		aKey := ackedIter.Key()
		aValue := ackedIter.Value()
		ackedTs, err := strconv.ParseInt(string(aValue.Data()), 10, 64)
		if aValue != nil {
			aValue.Free()
		}
		if err != nil {
			if aKey != nil {
				aKey.Free()
			}
			return err
		}
		if msgTimeExpiredCheck(ackedTs) {
			endID, err = strconv.ParseInt(string(aKey.Data())[FixedChannelNameLen+1:], 10, 64)
			if aKey != nil {
				aKey.Free()
			}
			if err != nil {
				return err
			}
		} else {
			if aKey != nil {
				aKey.Free()
			}
			break
		}
	}

	if endID == 0 {
		log.Debug("All messages are not time expired")
	}
	log.Debug("Expired check by retention time", zap.Any("topic", topic), zap.Any("startID", startID), zap.Any("endID", endID), zap.Any("deletedAckedSize", deletedAckedSize))

	ackedSizeKey := AckedSizeTitle + topic
	totalAckedSizeVal, err := ri.kv.Load(ackedSizeKey)
	if err != nil {
		return err
	}
	totalAckedSize, err := strconv.ParseInt(totalAckedSizeVal, 10, 64)
	if err != nil {
		return err
	}

	for ; pageIter.Valid(); pageIter.Next() {
		pValue := pageIter.Value()
		size, err := strconv.ParseInt(string(pValue.Data()), 10, 64)
		if pValue != nil {
			pValue.Free()
		}
		pKey := pageIter.Key()
		pKeyStr := string(pKey.Data())
		if pKey != nil {
			pKey.Free()
		}
		if err != nil {
			return err
		}
		curDeleteSize := deletedAckedSize + size
		if msgSizeExpiredCheck(curDeleteSize, totalAckedSize) {
			endID, err = strconv.ParseInt(pKeyStr[FixedChannelNameLen+1:], 10, 64)
			if err != nil {
				return err
			}
			pageEndID = endID
			deletedAckedSize += size
		} else {
			break
		}
	}
	if endID == 0 {
		log.Debug("All messages are not expired")
		return nil
	}
	log.Debug("ExpiredCleanUp: ", zap.Any("topic", topic), zap.Any("startID", startID), zap.Any("endID", endID), zap.Any("deletedAckedSize", deletedAckedSize))

	writeBatch := gorocksdb.NewWriteBatch()
	defer writeBatch.Destroy()

	pageStartIDKey := pageMsgPrefix + "/" + strconv.FormatInt(pageStartID, 10)
	pageEndIDKey := pageMsgPrefix + "/" + strconv.FormatInt(pageEndID+1, 10)
	if pageStartID == pageEndID {
		if pageStartID != 0 {
			writeBatch.Delete([]byte(pageStartIDKey))
		}
	} else if pageStartID < pageEndID {
		writeBatch.DeleteRange([]byte(pageStartIDKey), []byte(pageEndIDKey))
	}

	ackedStartIDKey := fixedAckedTsKey + "/" + strconv.Itoa(int(startID))
	ackedEndIDKey := fixedAckedTsKey + "/" + strconv.Itoa(int(endID+1))
	if startID > endID {
		return nil
	} else if startID == endID {
		writeBatch.Delete([]byte(ackedStartIDKey))
	} else {
		writeBatch.DeleteRange([]byte(ackedStartIDKey), []byte(ackedEndIDKey))
	}

	newAckedSize := totalAckedSize - deletedAckedSize
	writeBatch.Put([]byte(ackedSizeKey), []byte(strconv.FormatInt(newAckedSize, 10)))

	err = DeleteMessages(ri.db, topic, startID, endID)
	if err != nil {
		return err
	}

	writeOpts := gorocksdb.NewDefaultWriteOptions()
	defer writeOpts.Destroy()
	ri.kv.DB.Write(writeOpts, writeBatch)

	return nil
}

/*
// 1. Obtain pageAckedInfo and do time expired check, get the expired page scope;
// 2. Do iteration in the page after the last page in step 1 and get the last time expired message id;
// 3. Do size expired check in next page, and get the last size expired message id;
// 4. Do delete by range of [start_msg_id, end_msg_id) in rocksdb
// 5. Delete corresponding data in retentionInfo
func (ri *retentionInfo) expiredCleanUp(topic string) error {
	log.Debug("Timeticker triggers an expiredCleanUp task for topic: " + topic)
	var ackedInfo *topicAckedInfo
	if info, ok := ri.ackedInfo.Load(topic); ok {
		ackedInfo = info.(*topicAckedInfo)
	} else {
		log.Debug("Topic " + topic + " doesn't have acked infos")
		return nil
	}

	ll, ok := topicMu.Load(topic)
	if !ok {
		return fmt.Errorf("topic name = %s not exist", topic)
	}
	lock, ok := ll.(*sync.Mutex)
	if !ok {
		return fmt.Errorf("get mutex failed, topic name = %s", topic)
	}
	lock.Lock()
	defer lock.Unlock()

	readOpts := gorocksdb.NewDefaultReadOptions()
	defer readOpts.Destroy()
	readOpts.SetPrefixSameAsStart(true)
	iter := ri.kv.DB.NewIterator(readOpts)
	defer iter.Close()
	ackedTsPrefix, err := constructKey(AckedTsTitle, topic)
	if err != nil {
		return err
	}
	iter.Seek([]byte(ackedTsPrefix))
	if !iter.Valid() {
		return nil
	}
	var startID UniqueID
	var endID UniqueID
	endID = 0
	startID, err = strconv.ParseInt(string(iter.Key().Data())[FixedChannelNameLen+1:], 10, 64)
	if err != nil {
		return err
	}

	var deletedAckedSize int64 = 0
	pageRetentionOffset := 0
	var pageInfo *topicPageInfo
	if info, ok := ri.pageInfo.Load(topic); ok {
		pageInfo = info.(*topicPageInfo)
	}
	if pageInfo != nil {
		for i, pageEndID := range pageInfo.pageEndID {
			// Clean by RocksmqRetentionTimeInMinutes
			if msgTimeExpiredCheck(ackedInfo.ackedTs[pageEndID]) {
				// All of the page expired, set the pageEndID to current endID
				endID = pageEndID
				fixedAckedTsKey, err := constructKey(AckedTsTitle, topic)
				if err != nil {
					return err
				}
				newKey := fixedAckedTsKey + "/" + strconv.Itoa(int(pageEndID))
				iter.Seek([]byte(newKey))
				pageRetentionOffset = i + 1

				deletedAckedSize += pageInfo.pageMsgSize[pageEndID]
				delete(pageInfo.pageMsgSize, pageEndID)
			}
		}
	}
	log.Debug("Expired check by page info", zap.Any("topic", topic), zap.Any("pageEndID", endID), zap.Any("deletedAckedSize", deletedAckedSize))

	pageEndID := endID
	// The end msg of the page is not expired, find the last expired msg in this page
	for ; iter.Valid(); iter.Next() {
		ackedTs, err := strconv.ParseInt(string(iter.Value().Data()), 10, 64)
		if err != nil {
			return err
		}
		if msgTimeExpiredCheck(ackedTs) {
			endID, err = strconv.ParseInt(string(iter.Key().Data())[FixedChannelNameLen+1:], 10, 64)
			if err != nil {
				return err
			}
		} else {
			break
		}
	}
	log.Debug("Expired check by retention time", zap.Any("topic", topic), zap.Any("startID", startID), zap.Any("endID", endID), zap.Any("deletedAckedSize", deletedAckedSize))
	// if endID == 0 {
	// 	log.Debug("All messages are not expired")
	// 	return nil
	// }

	// Delete page message size in rocksdb_kv
	if pageInfo != nil {
		// Judge expire by ackedSize
		if msgSizeExpiredCheck(deletedAckedSize, ackedInfo.ackedSize) {
			for _, pEndID := range pageInfo.pageEndID[pageRetentionOffset:] {
				curDeletedSize := deletedAckedSize + pageInfo.pageMsgSize[pEndID]
				if msgSizeExpiredCheck(curDeletedSize, ackedInfo.ackedSize) {
					endID = pEndID
					pageEndID = pEndID
					deletedAckedSize = curDeletedSize
					delete(pageInfo.pageMsgSize, pEndID)
				} else {
					break
				}
			}
			log.Debug("Expired check by retention size", zap.Any("topic", topic), zap.Any("new endID", endID), zap.Any("new deletedAckedSize", deletedAckedSize))
		}

		if pageEndID > 0 && len(pageInfo.pageEndID) > 0 {
			pageStartID := pageInfo.pageEndID[0]
			fixedPageSizeKey, err := constructKey(PageMsgSizeTitle, topic)
			if err != nil {
				return err
			}
			pageStartKey := fixedPageSizeKey + "/" + strconv.Itoa(int(pageStartID))
			pageEndKey := fixedPageSizeKey + "/" + strconv.Itoa(int(pageEndID))
			pageWriteBatch := gorocksdb.NewWriteBatch()
			defer pageWriteBatch.Clear()
			log.Debug("Delete page info", zap.Any("topic", topic), zap.Any("pageStartID", pageStartID), zap.Any("pageEndID", pageEndID))
			if pageStartID == pageEndID {
				pageWriteBatch.Delete([]byte(pageStartKey))
			} else if pageStartID < pageEndID {
				pageWriteBatch.DeleteRange([]byte(pageStartKey), []byte(pageEndKey))
			}
			err = ri.kv.DB.Write(gorocksdb.NewDefaultWriteOptions(), pageWriteBatch)
			if err != nil {
				log.Error("rocksdb write error", zap.Error(err))
				return err
			}

			pageInfo.pageEndID = pageInfo.pageEndID[pageRetentionOffset:]
		}
		ri.pageInfo.Store(topic, pageInfo)
	}
	if endID == 0 {
		log.Debug("All messages are not expired")
		return nil
	}
	log.Debug("ExpiredCleanUp: ", zap.Any("topic", topic), zap.Any("startID", startID), zap.Any("endID", endID), zap.Any("deletedAckedSize", deletedAckedSize))

	// Delete acked_ts in rocksdb_kv
	fixedAckedTsTitle, err := constructKey(AckedTsTitle, topic)
	if err != nil {
		return err
	}
	ackedStartIDKey := fixedAckedTsTitle + "/" + strconv.Itoa(int(startID))
	ackedEndIDKey := fixedAckedTsTitle + "/" + strconv.Itoa(int(endID))
	ackedTsWriteBatch := gorocksdb.NewWriteBatch()
	defer ackedTsWriteBatch.Clear()
	if startID > endID {
		return nil
	} else if startID == endID {
		ackedTsWriteBatch.Delete([]byte(ackedStartIDKey))
	} else {
		ackedTsWriteBatch.DeleteRange([]byte(ackedStartIDKey), []byte(ackedEndIDKey))
	}
	err = ri.kv.DB.Write(gorocksdb.NewDefaultWriteOptions(), ackedTsWriteBatch)
	if err != nil {
		log.Error("rocksdb write error", zap.Error(err))
		return err
	}

	// Update acked_size in rocksdb_kv

	// Update last retention ts
	lastRetentionTsKey := LastRetTsTitle + topic
	err = ri.kv.Save(lastRetentionTsKey, strconv.FormatInt(time.Now().Unix(), 10))
	if err != nil {
		return err
	}

	ackedInfo.ackedSize -= deletedAckedSize
	ackedSizeKey := AckedSizeTitle + topic
	err = ri.kv.Save(ackedSizeKey, strconv.FormatInt(ackedInfo.ackedSize, 10))
	if err != nil {
		return err
	}

	for k := range ackedInfo.ackedTs {
		if k < endID {
			delete(ackedInfo.ackedTs, k)
		}
	}
	ri.ackedInfo.Store(topic, ackedInfo)

	return DeleteMessages(ri.db, topic, startID, endID)
}
*/

// DeleteMessages in rocksdb by range of [startID, endID)
func DeleteMessages(db *gorocksdb.DB, topic string, startID, endID UniqueID) error {
	// Delete msg by range of startID and endID
	startKey, err := combKey(topic, startID)
	if err != nil {
		log.Debug("RocksMQ: combKey(" + topic + "," + strconv.FormatInt(startID, 10) + ")")
		return err
	}
	endKey, err := combKey(topic, endID+1)
	if err != nil {
		log.Debug("RocksMQ: combKey(" + topic + "," + strconv.FormatInt(endID, 10) + ")")
		return err
	}

	writeBatch := gorocksdb.NewWriteBatch()
	defer writeBatch.Destroy()
	if startID == endID {
		writeBatch.Delete([]byte(startKey))
	} else {
		writeBatch.DeleteRange([]byte(startKey), []byte(endKey))
	}
	opts := gorocksdb.NewDefaultWriteOptions()
	defer opts.Destroy()
	err = db.Write(opts, writeBatch)
	if err != nil {
		return err
	}

	log.Debug("Delete message for topic: "+topic, zap.Any("startID", startID), zap.Any("endID", endID))

	return nil
}

func msgTimeExpiredCheck(ackedTs int64) bool {
	return ackedTs+atomic.LoadInt64(&RocksmqRetentionTimeInMinutes)*MINUTE < time.Now().Unix()
}

func msgSizeExpiredCheck(deletedAckedSize, ackedSize int64) bool {
	return ackedSize-deletedAckedSize > atomic.LoadInt64(&RocksmqRetentionSizeInMB)*MB
}
