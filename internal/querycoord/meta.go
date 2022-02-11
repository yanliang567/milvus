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

package querycoord

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/rootcoord"
	"github.com/milvus-io/milvus/internal/util"
)

const (
	collectionMetaPrefix   = "queryCoord-collectionMeta"
	dmChannelMetaPrefix    = "queryCoord-dmChannelWatchInfo"
	deltaChannelMetaPrefix = "queryCoord-deltaChannel"
)

type col2SegmentInfos = map[UniqueID][]*querypb.SegmentInfo
type col2SealedSegmentChangeInfos = map[UniqueID]*querypb.SealedSegmentsChangeInfo

// Meta contains information about all loaded collections and partitions, including segment information and vchannel information
type Meta interface {
	reloadFromKV() error
	setKvClient(kv kv.MetaKv)

	showCollections() []*querypb.CollectionInfo
	hasCollection(collectionID UniqueID) bool
	getCollectionInfoByID(collectionID UniqueID) (*querypb.CollectionInfo, error)
	addCollection(collectionID UniqueID, loadType querypb.LoadType, schema *schemapb.CollectionSchema) error
	releaseCollection(collectionID UniqueID) error

	addPartitions(collectionID UniqueID, partitionIDs []UniqueID) error
	showPartitions(collectionID UniqueID) ([]*querypb.PartitionStates, error)
	hasPartition(collectionID UniqueID, partitionID UniqueID) bool
	hasReleasePartition(collectionID UniqueID, partitionID UniqueID) bool
	releasePartitions(collectionID UniqueID, partitionIDs []UniqueID) error

	showSegmentInfos(collectionID UniqueID, partitionIDs []UniqueID) []*querypb.SegmentInfo
	getSegmentInfoByID(segmentID UniqueID) (*querypb.SegmentInfo, error)
	getSegmentInfosByNode(nodeID int64) []*querypb.SegmentInfo

	getPartitionStatesByID(collectionID UniqueID, partitionID UniqueID) (*querypb.PartitionStates, error)

	getDmChannelInfosByNodeID(nodeID int64) []*querypb.DmChannelWatchInfo
	setDmChannelInfos(channelInfos []*querypb.DmChannelWatchInfo) error

	getDeltaChannelsByCollectionID(collectionID UniqueID) ([]*datapb.VchannelInfo, error)
	setDeltaChannel(collectionID UniqueID, info []*datapb.VchannelInfo) error

	getQueryChannelInfoByID(collectionID UniqueID) *querypb.QueryChannelInfo
	getQueryStreamByID(collectionID UniqueID, queryChannel string) (msgstream.MsgStream, error)

	setLoadType(collectionID UniqueID, loadType querypb.LoadType) error
	setLoadPercentage(collectionID UniqueID, partitionID UniqueID, percentage int64, loadType querypb.LoadType) error
	//printMeta()
	saveGlobalSealedSegInfos(saves col2SegmentInfos) (col2SealedSegmentChangeInfos, error)
	removeGlobalSealedSegInfos(collectionID UniqueID, partitionIDs []UniqueID) (col2SealedSegmentChangeInfos, error)
	sendSealedSegmentChangeInfos(collectionID UniqueID, queryChannel string, changeInfos *querypb.SealedSegmentsChangeInfo) (*internalpb.MsgPosition, error)

	getWatchedChannelsByNodeID(nodeID int64) *querypb.UnsubscribeChannelInfo
}

// MetaReplica records the current load information on all querynodes
type MetaReplica struct {
	ctx         context.Context
	cancel      context.CancelFunc
	client      kv.MetaKv // client of a reliable kv service, i.e. etcd client
	msFactory   msgstream.Factory
	idAllocator func() (UniqueID, error)

	//sync.RWMutex
	collectionInfos   map[UniqueID]*querypb.CollectionInfo
	collectionMu      sync.RWMutex
	segmentInfos      map[UniqueID]*querypb.SegmentInfo
	segmentMu         sync.RWMutex
	queryChannelInfos map[UniqueID]*querypb.QueryChannelInfo
	channelMu         sync.RWMutex
	deltaChannelInfos map[UniqueID][]*datapb.VchannelInfo
	deltaChannelMu    sync.RWMutex
	dmChannelInfos    map[string]*querypb.DmChannelWatchInfo
	dmChannelMu       sync.RWMutex
	queryStreams      map[UniqueID]msgstream.MsgStream
	streamMu          sync.RWMutex

	//partitionStates map[UniqueID]*querypb.PartitionStates
}

func newMeta(ctx context.Context, kv kv.MetaKv, factory msgstream.Factory, idAllocator func() (UniqueID, error)) (Meta, error) {
	childCtx, cancel := context.WithCancel(ctx)
	collectionInfos := make(map[UniqueID]*querypb.CollectionInfo)
	segmentInfos := make(map[UniqueID]*querypb.SegmentInfo)
	queryChannelInfos := make(map[UniqueID]*querypb.QueryChannelInfo)
	deltaChannelInfos := make(map[UniqueID][]*datapb.VchannelInfo)
	dmChannelInfos := make(map[string]*querypb.DmChannelWatchInfo)
	queryMsgStream := make(map[UniqueID]msgstream.MsgStream)

	m := &MetaReplica{
		ctx:         childCtx,
		cancel:      cancel,
		client:      kv,
		msFactory:   factory,
		idAllocator: idAllocator,

		collectionInfos:   collectionInfos,
		segmentInfos:      segmentInfos,
		queryChannelInfos: queryChannelInfos,
		deltaChannelInfos: deltaChannelInfos,
		dmChannelInfos:    dmChannelInfos,
		queryStreams:      queryMsgStream,
	}

	err := m.reloadFromKV()
	if err != nil {
		return nil, err
	}

	return m, nil
}

func (m *MetaReplica) reloadFromKV() error {
	log.Debug("start reload from kv")
	collectionKeys, collectionValues, err := m.client.LoadWithPrefix(collectionMetaPrefix)
	if err != nil {
		return err
	}
	for index := range collectionKeys {
		collectionID, err := strconv.ParseInt(filepath.Base(collectionKeys[index]), 10, 64)
		if err != nil {
			return err
		}
		collectionInfo := &querypb.CollectionInfo{}
		err = proto.Unmarshal([]byte(collectionValues[index]), collectionInfo)
		if err != nil {
			return err
		}
		m.collectionInfos[collectionID] = collectionInfo
	}

	segmentKeys, segmentValues, err := m.client.LoadWithPrefix(util.SegmentMetaPrefix)
	if err != nil {
		return err
	}
	for index := range segmentKeys {
		segmentID, err := strconv.ParseInt(filepath.Base(segmentKeys[index]), 10, 64)
		if err != nil {
			return err
		}
		segmentInfo := &querypb.SegmentInfo{}
		err = proto.Unmarshal([]byte(segmentValues[index]), segmentInfo)
		if err != nil {
			return err
		}
		m.segmentInfos[segmentID] = segmentInfo
	}

	deltaChannelKeys, deltaChannelValues, err := m.client.LoadWithPrefix(deltaChannelMetaPrefix)
	if err != nil {
		return nil
	}
	for index, value := range deltaChannelValues {
		pathStrings := strings.Split(deltaChannelKeys[index], "/")
		collectionID, err := strconv.ParseInt(pathStrings[len(pathStrings)-2], 10, 64)
		if err != nil {
			return err
		}
		deltaChannelInfo := &datapb.VchannelInfo{}
		err = proto.Unmarshal([]byte(value), deltaChannelInfo)
		if err != nil {
			return err
		}
		m.deltaChannelInfos[collectionID] = append(m.deltaChannelInfos[collectionID], deltaChannelInfo)
	}

	dmChannelKeys, dmChannelValues, err := m.client.LoadWithPrefix(dmChannelMetaPrefix)
	if err != nil {
		return err
	}
	for index := range dmChannelKeys {
		dmChannel := filepath.Base(dmChannelKeys[index])
		dmChannelWatchInfo := &querypb.DmChannelWatchInfo{}
		err = proto.Unmarshal([]byte(dmChannelValues[index]), dmChannelWatchInfo)
		if err != nil {
			return err
		}
		m.dmChannelInfos[dmChannel] = dmChannelWatchInfo
	}

	//TODO::update partition states
	log.Debug("reload from kv finished")

	return nil
}

func (m *MetaReplica) setKvClient(kv kv.MetaKv) {
	m.client = kv
}

func (m *MetaReplica) showCollections() []*querypb.CollectionInfo {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	collections := make([]*querypb.CollectionInfo, 0)
	for _, info := range m.collectionInfos {
		collections = append(collections, proto.Clone(info).(*querypb.CollectionInfo))
	}
	return collections
}

func (m *MetaReplica) showPartitions(collectionID UniqueID) ([]*querypb.PartitionStates, error) {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	//TODO::should update after load collection
	results := make([]*querypb.PartitionStates, 0)
	if info, ok := m.collectionInfos[collectionID]; ok {
		for _, state := range info.PartitionStates {
			results = append(results, proto.Clone(state).(*querypb.PartitionStates))
		}
		return results, nil
	}

	return nil, errors.New("showPartitions: can't find collection in collectionInfos")
}

func (m *MetaReplica) hasCollection(collectionID UniqueID) bool {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	if _, ok := m.collectionInfos[collectionID]; ok {
		return true
	}

	return false
}

func (m *MetaReplica) hasPartition(collectionID UniqueID, partitionID UniqueID) bool {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	if info, ok := m.collectionInfos[collectionID]; ok {
		for _, id := range info.PartitionIDs {
			if partitionID == id {
				return true
			}
		}
	}

	return false
}

func (m *MetaReplica) hasReleasePartition(collectionID UniqueID, partitionID UniqueID) bool {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	if info, ok := m.collectionInfos[collectionID]; ok {
		for _, id := range info.ReleasedPartitionIDs {
			if partitionID == id {
				return true
			}
		}
	}

	return false
}

func (m *MetaReplica) addCollection(collectionID UniqueID, loadType querypb.LoadType, schema *schemapb.CollectionSchema) error {
	hasCollection := m.hasCollection(collectionID)
	if !hasCollection {
		var partitionIDs []UniqueID
		var partitionStates []*querypb.PartitionStates
		newCollection := &querypb.CollectionInfo{
			CollectionID:    collectionID,
			PartitionIDs:    partitionIDs,
			PartitionStates: partitionStates,
			LoadType:        loadType,
			Schema:          schema,
		}
		err := saveGlobalCollectionInfo(collectionID, newCollection, m.client)
		if err != nil {
			log.Error("save collectionInfo error", zap.Any("error", err.Error()), zap.Int64("collectionID", collectionID))
			return err
		}
		m.collectionMu.Lock()
		m.collectionInfos[collectionID] = newCollection
		m.collectionMu.Unlock()
	}

	return nil
}

func (m *MetaReplica) addPartitions(collectionID UniqueID, partitionIDs []UniqueID) error {
	m.collectionMu.Lock()
	defer m.collectionMu.Unlock()

	if info, ok := m.collectionInfos[collectionID]; ok {
		collectionInfo := proto.Clone(info).(*querypb.CollectionInfo)
		loadedPartitionID2State := make(map[UniqueID]*querypb.PartitionStates)
		for _, partitionID := range partitionIDs {
			loadedPartitionID2State[partitionID] = &querypb.PartitionStates{
				PartitionID: partitionID,
				State:       querypb.PartitionState_NotPresent,
			}
		}

		for offset, partitionID := range collectionInfo.PartitionIDs {
			loadedPartitionID2State[partitionID] = collectionInfo.PartitionStates[offset]
		}

		newPartitionIDs := make([]UniqueID, 0)
		newPartitionStates := make([]*querypb.PartitionStates, 0)
		for partitionID, state := range loadedPartitionID2State {
			newPartitionIDs = append(newPartitionIDs, partitionID)
			newPartitionStates = append(newPartitionStates, state)
		}

		newReleasedPartitionIDs := make([]UniqueID, 0)
		for _, releasedPartitionID := range collectionInfo.ReleasedPartitionIDs {
			if _, ok = loadedPartitionID2State[releasedPartitionID]; !ok {
				newReleasedPartitionIDs = append(newReleasedPartitionIDs, releasedPartitionID)
			}
		}

		collectionInfo.PartitionIDs = newPartitionIDs
		collectionInfo.PartitionStates = newPartitionStates
		collectionInfo.ReleasedPartitionIDs = newReleasedPartitionIDs

		log.Debug("add a  partition to MetaReplica", zap.Int64("collectionID", collectionID), zap.Int64s("partitionIDs", collectionInfo.PartitionIDs))
		err := saveGlobalCollectionInfo(collectionID, collectionInfo, m.client)
		if err != nil {
			log.Error("save collectionInfo error", zap.Int64("collectionID", collectionID), zap.Int64s("partitionIDs", collectionInfo.PartitionIDs), zap.Any("error", err.Error()))
			return err
		}
		m.collectionInfos[collectionID] = collectionInfo
		return nil
	}
	return fmt.Errorf("addPartition: can't find collection %d when add partition", collectionID)
}

func (m *MetaReplica) releaseCollection(collectionID UniqueID) error {
	err := removeCollectionMeta(collectionID, m.client)
	if err != nil {
		log.Warn("remove collectionInfo from etcd failed", zap.Int64("collectionID", collectionID), zap.Any("error", err.Error()))
		return err
	}

	m.collectionMu.Lock()
	delete(m.collectionInfos, collectionID)
	m.collectionMu.Unlock()

	m.deltaChannelMu.Lock()
	delete(m.deltaChannelInfos, collectionID)
	m.deltaChannelMu.Unlock()

	m.dmChannelMu.Lock()
	for dmChannel, info := range m.dmChannelInfos {
		if info.CollectionID == collectionID {
			delete(m.dmChannelInfos, dmChannel)
		}
	}
	m.dmChannelMu.Unlock()

	return nil
}

func (m *MetaReplica) releasePartitions(collectionID UniqueID, releasedPartitionIDs []UniqueID) error {
	m.collectionMu.Lock()
	defer m.collectionMu.Unlock()
	info, ok := m.collectionInfos[collectionID]
	if !ok {
		return nil
	}
	collectionInfo := proto.Clone(info).(*querypb.CollectionInfo)

	releasedPartitionMap := make(map[UniqueID]struct{})
	for _, partitionID := range releasedPartitionIDs {
		releasedPartitionMap[partitionID] = struct{}{}
	}
	for _, partitionID := range collectionInfo.ReleasedPartitionIDs {
		releasedPartitionMap[partitionID] = struct{}{}
	}

	newPartitionIDs := make([]UniqueID, 0)
	newPartitionStates := make([]*querypb.PartitionStates, 0)
	for offset, partitionID := range collectionInfo.PartitionIDs {
		if _, ok = releasedPartitionMap[partitionID]; !ok {
			newPartitionIDs = append(newPartitionIDs, partitionID)
			newPartitionStates = append(newPartitionStates, collectionInfo.PartitionStates[offset])
		}
	}

	newReleasedPartitionIDs := make([]UniqueID, 0)
	for partitionID := range releasedPartitionMap {
		newReleasedPartitionIDs = append(newReleasedPartitionIDs, partitionID)
	}

	collectionInfo.PartitionIDs = newPartitionIDs
	collectionInfo.PartitionStates = newPartitionStates
	collectionInfo.ReleasedPartitionIDs = newReleasedPartitionIDs

	err := saveGlobalCollectionInfo(collectionID, collectionInfo, m.client)
	if err != nil {
		log.Error("releasePartition: remove partition infos error", zap.Int64("collectionID", collectionID), zap.Int64s("partitionIDs", releasedPartitionIDs), zap.Any("error", err.Error()))
		return err
	}

	m.collectionInfos[collectionID] = collectionInfo

	return nil
}

func (m *MetaReplica) saveGlobalSealedSegInfos(saves col2SegmentInfos) (col2SealedSegmentChangeInfos, error) {
	if len(saves) == 0 {
		return nil, nil
	}
	// generate segment change info according segment info to updated
	col2SegmentChangeInfos := make(col2SealedSegmentChangeInfos)

	segmentsCompactionFrom := make([]*querypb.SegmentInfo, 0)
	// get segmentInfos to colSegmentInfos
	for collectionID, onlineInfos := range saves {
		segmentsChangeInfo := &querypb.SealedSegmentsChangeInfo{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_SealedSegmentsChangeInfo,
			},
			Infos: []*querypb.SegmentChangeInfo{},
		}
		for _, info := range onlineInfos {
			segmentID := info.SegmentID
			onlineNodeID := info.NodeID
			changeInfo := &querypb.SegmentChangeInfo{
				OnlineNodeID:   onlineNodeID,
				OnlineSegments: []*querypb.SegmentInfo{info},
			}
			offlineInfo, err := m.getSegmentInfoByID(segmentID)
			if err == nil {
				offlineNodeID := offlineInfo.NodeID
				// if the offline segment state is growing, it will not impact the global sealed segments
				if offlineInfo.SegmentState == commonpb.SegmentState_Sealed {
					changeInfo.OfflineNodeID = offlineNodeID
					changeInfo.OfflineSegments = []*querypb.SegmentInfo{offlineInfo}
				}
			}
			segmentsChangeInfo.Infos = append(segmentsChangeInfo.Infos, changeInfo)

			// generate offline segment change info if the loaded segment is compacted from other sealed segments
			for _, compactionSegmentID := range info.CompactionFrom {
				compactionSegmentInfo, err := m.getSegmentInfoByID(compactionSegmentID)
				if err == nil && compactionSegmentInfo.SegmentState == commonpb.SegmentState_Sealed {
					segmentsChangeInfo.Infos = append(segmentsChangeInfo.Infos, &querypb.SegmentChangeInfo{
						OfflineNodeID:   compactionSegmentInfo.NodeID,
						OfflineSegments: []*querypb.SegmentInfo{compactionSegmentInfo},
					})
					segmentsCompactionFrom = append(segmentsCompactionFrom, compactionSegmentInfo)
				} else {
					return nil, fmt.Errorf("saveGlobalSealedSegInfos: the compacted segment %d has not been loaded into memory", compactionSegmentID)
				}
			}
		}
		col2SegmentChangeInfos[collectionID] = segmentsChangeInfo
	}

	queryChannelInfosMap := make(map[UniqueID]*querypb.QueryChannelInfo)
	for collectionID, segmentChangeInfos := range col2SegmentChangeInfos {
		// get msgStream to produce sealedSegmentChangeInfos to query channel
		queryChannelInfo := m.getQueryChannelInfoByID(collectionID)
		msgPosition, err := m.sendSealedSegmentChangeInfos(collectionID, queryChannelInfo.QueryChannel, segmentChangeInfos)
		if err != nil {
			return nil, err
		}
		queryChannelInfo.SeekPosition = msgPosition

		// update segmentInfo, queryChannelInfo meta to cache and etcd
		seg2Info := make(map[UniqueID]*querypb.SegmentInfo)
		for _, segmentInfo := range queryChannelInfo.GlobalSealedSegments {
			segmentID := segmentInfo.SegmentID
			seg2Info[segmentID] = segmentInfo
		}
		if infos, ok := saves[collectionID]; ok {
			for _, segmentInfo := range infos {
				segmentID := segmentInfo.SegmentID
				seg2Info[segmentID] = segmentInfo
			}
		}

		globalSealedSegmentInfos := make([]*querypb.SegmentInfo, 0)
		for _, info := range seg2Info {
			globalSealedSegmentInfos = append(globalSealedSegmentInfos, info)
		}
		queryChannelInfo.GlobalSealedSegments = globalSealedSegmentInfos
		queryChannelInfosMap[collectionID] = queryChannelInfo
	}

	// save segmentInfo to etcd
	segmentInfoKvs := make(map[string]string)
	for _, infos := range saves {
		for _, info := range infos {
			segmentInfoBytes, err := proto.Marshal(info)
			if err != nil {
				return col2SegmentChangeInfos, err
			}
			segmentKey := fmt.Sprintf("%s/%d/%d/%d", util.SegmentMetaPrefix, info.CollectionID, info.PartitionID, info.SegmentID)
			segmentInfoKvs[segmentKey] = string(segmentInfoBytes)
		}
	}
	for key, value := range segmentInfoKvs {
		err := m.client.Save(key, value)
		if err != nil {
			panic(err)
		}
	}

	// remove compacted segment info from etcd
	for _, segmentInfo := range segmentsCompactionFrom {
		segmentKey := fmt.Sprintf("%s/%d/%d/%d", util.SegmentMetaPrefix, segmentInfo.CollectionID, segmentInfo.PartitionID, segmentInfo.SegmentID)
		err := m.client.Remove(segmentKey)
		if err != nil {
			panic(err)
		}
	}

	// save sealedSegmentsChangeInfo to etcd
	saveKvs := make(map[string]string)
	// save segmentChangeInfo into etcd, query node will deal the changeInfo if the msgID key exist in etcd
	// avoid the produce process success but save meta to etcd failed
	// then the msgID key will not exist, and changeIndo will be ignored by query node
	for _, changeInfos := range col2SegmentChangeInfos {
		changeInfoBytes, err := proto.Marshal(changeInfos)
		if err != nil {
			return col2SegmentChangeInfos, err
		}
		// TODO:: segmentChangeInfo clear in etcd with coord gc and queryNode watch the changeInfo meta to deal changeInfoMsg
		changeInfoKey := fmt.Sprintf("%s/%d", util.ChangeInfoMetaPrefix, changeInfos.Base.MsgID)
		saveKvs[changeInfoKey] = string(changeInfoBytes)
	}

	err := m.client.MultiSave(saveKvs)
	if err != nil {
		panic(err)
	}

	m.segmentMu.Lock()
	for _, segmentInfos := range saves {
		for _, info := range segmentInfos {
			segmentID := info.SegmentID
			m.segmentInfos[segmentID] = info
		}
	}
	for _, segmentInfo := range segmentsCompactionFrom {
		delete(m.segmentInfos, segmentInfo.SegmentID)
	}
	m.segmentMu.Unlock()

	m.channelMu.Lock()
	for collectionID, channelInfo := range queryChannelInfosMap {
		m.queryChannelInfos[collectionID] = channelInfo
	}
	m.channelMu.Unlock()

	return col2SegmentChangeInfos, nil
}

func (m *MetaReplica) removeGlobalSealedSegInfos(collectionID UniqueID, partitionIDs []UniqueID) (col2SealedSegmentChangeInfos, error) {
	removes := m.showSegmentInfos(collectionID, partitionIDs)
	if len(removes) == 0 {
		return nil, nil
	}
	// get segmentInfos to remove
	segmentChangeInfos := &querypb.SealedSegmentsChangeInfo{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_SealedSegmentsChangeInfo,
		},
		Infos: []*querypb.SegmentChangeInfo{},
	}
	for _, info := range removes {
		offlineNodeID := info.NodeID
		changeInfo := &querypb.SegmentChangeInfo{
			OfflineNodeID:   offlineNodeID,
			OfflineSegments: []*querypb.SegmentInfo{info},
		}

		segmentChangeInfos.Infos = append(segmentChangeInfos.Infos, changeInfo)
	}

	// produce sealedSegmentChangeInfos to query channel
	queryChannelInfo := m.getQueryChannelInfoByID(collectionID)
	msgPosition, err := m.sendSealedSegmentChangeInfos(collectionID, queryChannelInfo.QueryChannel, segmentChangeInfos)
	if err != nil {
		return nil, err
	}
	queryChannelInfo.SeekPosition = msgPosition

	// update segmentInfo, queryChannelInfo meta to cache and etcd
	seg2Info := make(map[UniqueID]*querypb.SegmentInfo)
	for _, segmentInfo := range queryChannelInfo.GlobalSealedSegments {
		segmentID := segmentInfo.SegmentID
		seg2Info[segmentID] = segmentInfo
	}

	for _, segmentInfo := range removes {
		segmentID := segmentInfo.SegmentID
		delete(seg2Info, segmentID)
	}

	globalSealedSegmentInfos := make([]*querypb.SegmentInfo, 0)
	for _, info := range seg2Info {
		globalSealedSegmentInfos = append(globalSealedSegmentInfos, info)
	}
	queryChannelInfo.GlobalSealedSegments = globalSealedSegmentInfos

	// remove meta from etcd
	for _, info := range removes {
		segmentKey := fmt.Sprintf("%s/%d/%d/%d", util.SegmentMetaPrefix, info.CollectionID, info.PartitionID, info.SegmentID)
		err = m.client.Remove(segmentKey)
		if err != nil {
			panic(err)
		}
	}

	saveKvs := make(map[string]string)
	// save segmentChangeInfo into etcd, query node will deal the changeInfo if the msgID key exist in etcd
	// avoid the produce process success but save meta to etcd failed
	// then the msgID key will not exist, and changeIndo will be ignored by query node
	changeInfoBytes, err := proto.Marshal(segmentChangeInfos)
	if err != nil {
		return col2SealedSegmentChangeInfos{collectionID: segmentChangeInfos}, err
	}
	// TODO:: segmentChangeInfo clear in etcd with coord gc and queryNode watch the changeInfo meta to deal changeInfoMsg
	changeInfoKey := fmt.Sprintf("%s/%d", util.ChangeInfoMetaPrefix, segmentChangeInfos.Base.MsgID)
	saveKvs[changeInfoKey] = string(changeInfoBytes)

	err = m.client.MultiSave(saveKvs)
	if err != nil {
		panic(err)
	}

	m.segmentMu.Lock()
	for _, info := range removes {
		delete(m.segmentInfos, info.SegmentID)
	}
	m.segmentMu.Unlock()

	m.channelMu.Lock()
	m.queryChannelInfos[collectionID] = queryChannelInfo
	m.channelMu.Unlock()

	return col2SealedSegmentChangeInfos{collectionID: segmentChangeInfos}, nil
}

// send sealed segment change infos into query channels
func (m *MetaReplica) sendSealedSegmentChangeInfos(collectionID UniqueID, queryChannel string, changeInfos *querypb.SealedSegmentsChangeInfo) (*internalpb.MsgPosition, error) {
	// get msgStream to produce sealedSegmentChangeInfos to query channel
	queryStream, err := m.getQueryStreamByID(collectionID, queryChannel)
	if err != nil {
		log.Error("sendSealedSegmentChangeInfos: get query stream failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}

	var msgPack = &msgstream.MsgPack{
		Msgs: []msgstream.TsMsg{},
	}
	id, err := m.idAllocator()
	if err != nil {
		log.Error("sendSealedSegmentChangeInfos: allocator trigger taskID failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}
	changeInfos.Base.MsgID = id
	segmentChangeMsg := &msgstream.SealedSegmentsChangeInfoMsg{
		BaseMsg: msgstream.BaseMsg{
			HashValues: []uint32{0},
		},
		SealedSegmentsChangeInfo: *changeInfos,
	}
	msgPack.Msgs = append(msgPack.Msgs, segmentChangeMsg)

	messageIDInfos, err := queryStream.ProduceMark(msgPack)
	if err != nil {
		log.Error("sendSealedSegmentChangeInfos: send sealed segment change info failed", zap.Int64("collectionID", collectionID), zap.Error(err))
		return nil, err
	}

	messageIDs, ok := messageIDInfos[queryChannel]
	if !ok {
		return nil, fmt.Errorf("sendSealedSegmentChangeInfos: send sealed segment change info to wrong query channel, collectionID = %d, query channel = %s", collectionID, queryChannel)
	}

	// len(messageIDs) = 1
	if len(messageIDs) != 1 {
		return nil, fmt.Errorf("sendSealedSegmentChangeInfos: length of the positions in stream is not correct, collectionID = %d, query channel = %s, len = %d", collectionID, queryChannel, len(messageIDs))
	}

	log.Debug("updateGlobalSealedSegmentInfos: send sealed segment change info to queryChannel", zap.Any("msgPack", msgPack))
	return &internalpb.MsgPosition{
		ChannelName: queryChannel,
		MsgID:       messageIDs[0].Serialize(),
	}, nil
}

func (m *MetaReplica) showSegmentInfos(collectionID UniqueID, partitionIDs []UniqueID) []*querypb.SegmentInfo {
	m.segmentMu.RLock()
	defer m.segmentMu.RUnlock()

	results := make([]*querypb.SegmentInfo, 0)
	segmentInfos := make([]*querypb.SegmentInfo, 0)
	for _, info := range m.segmentInfos {
		if info.CollectionID == collectionID {
			segmentInfos = append(segmentInfos, proto.Clone(info).(*querypb.SegmentInfo))
		}
	}
	if len(partitionIDs) == 0 {
		return segmentInfos
	}

	partitionIDMap := getCompareMapFromSlice(partitionIDs)
	for _, info := range segmentInfos {
		partitionID := info.PartitionID
		if _, ok := partitionIDMap[partitionID]; ok {
			results = append(results, info)
		}
	}
	return results
}

func (m *MetaReplica) getSegmentInfoByID(segmentID UniqueID) (*querypb.SegmentInfo, error) {
	m.segmentMu.RLock()
	defer m.segmentMu.RUnlock()

	if info, ok := m.segmentInfos[segmentID]; ok {
		return proto.Clone(info).(*querypb.SegmentInfo), nil
	}

	return nil, errors.New("getSegmentInfoByID: can't find segmentID in segmentInfos")
}
func (m *MetaReplica) getSegmentInfosByNode(nodeID int64) []*querypb.SegmentInfo {
	m.segmentMu.RLock()
	defer m.segmentMu.RUnlock()

	segmentInfos := make([]*querypb.SegmentInfo, 0)
	for _, info := range m.segmentInfos {
		if info.NodeID == nodeID {
			segmentInfos = append(segmentInfos, proto.Clone(info).(*querypb.SegmentInfo))
		}
	}

	return segmentInfos
}

func (m *MetaReplica) getCollectionInfoByID(collectionID UniqueID) (*querypb.CollectionInfo, error) {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	if info, ok := m.collectionInfos[collectionID]; ok {
		return proto.Clone(info).(*querypb.CollectionInfo), nil
	}

	return nil, errors.New("getCollectionInfoByID: can't find collectionID in collectionInfo")
}

func (m *MetaReplica) getPartitionStatesByID(collectionID UniqueID, partitionID UniqueID) (*querypb.PartitionStates, error) {
	m.collectionMu.RLock()
	defer m.collectionMu.RUnlock()

	if info, ok := m.collectionInfos[collectionID]; ok {
		for offset, id := range info.PartitionIDs {
			if id == partitionID {
				return proto.Clone(info.PartitionStates[offset]).(*querypb.PartitionStates), nil
			}
		}
		return nil, errors.New("getPartitionStateByID: can't find partitionID in partitionStates")
	}

	return nil, errors.New("getPartitionStateByID: can't find collectionID in collectionInfo")
}

func (m *MetaReplica) getDmChannelInfosByNodeID(nodeID int64) []*querypb.DmChannelWatchInfo {
	m.dmChannelMu.RLock()
	defer m.dmChannelMu.RUnlock()

	var watchedDmChannelWatchInfo []*querypb.DmChannelWatchInfo
	for _, channelInfo := range m.dmChannelInfos {
		if channelInfo.NodeIDLoaded == nodeID {
			watchedDmChannelWatchInfo = append(watchedDmChannelWatchInfo, proto.Clone(channelInfo).(*querypb.DmChannelWatchInfo))
		}
	}

	return watchedDmChannelWatchInfo
}

func (m *MetaReplica) setDmChannelInfos(dmChannelWatchInfos []*querypb.DmChannelWatchInfo) error {
	m.dmChannelMu.Lock()
	defer m.dmChannelMu.Unlock()

	err := saveDmChannelWatchInfos(dmChannelWatchInfos, m.client)
	if err != nil {
		log.Error("save dmChannelWatchInfo error", zap.Any("error", err.Error()))
		return err
	}
	for _, channelInfo := range dmChannelWatchInfos {
		m.dmChannelInfos[channelInfo.DmChannel] = channelInfo
	}

	return nil
}

func (m *MetaReplica) createQueryChannel(collectionID UniqueID) *querypb.QueryChannelInfo {
	// TODO::to remove
	// all collection use the same query channel
	colIDForAssignChannel := UniqueID(0)

	searchPrefix := Params.MsgChannelCfg.QueryCoordSearch
	searchResultPrefix := Params.MsgChannelCfg.QueryCoordSearchResult
	allocatedQueryChannel := searchPrefix + "-" + strconv.FormatInt(colIDForAssignChannel, 10)
	allocatedQueryResultChannel := searchResultPrefix + "-" + strconv.FormatInt(colIDForAssignChannel, 10)
	log.Debug("query coordinator create query channel", zap.String("queryChannelName", allocatedQueryChannel), zap.String("queryResultChannelName", allocatedQueryResultChannel))

	seekPosition := &internalpb.MsgPosition{
		ChannelName: allocatedQueryChannel,
	}
	segmentInfos := m.showSegmentInfos(collectionID, nil)
	info := &querypb.QueryChannelInfo{
		CollectionID:         collectionID,
		QueryChannel:         allocatedQueryChannel,
		QueryResultChannel:   allocatedQueryResultChannel,
		GlobalSealedSegments: segmentInfos,
		SeekPosition:         seekPosition,
	}

	return info
}

// Get delta channel info for collection, so far all the collection share the same query channel 0
func (m *MetaReplica) getDeltaChannelsByCollectionID(collectionID UniqueID) ([]*datapb.VchannelInfo, error) {
	m.deltaChannelMu.RLock()
	defer m.deltaChannelMu.RUnlock()
	if infos, ok := m.deltaChannelInfos[collectionID]; ok {
		return infos, nil
	}

	return nil, fmt.Errorf("delta channel not exist in meta, collectionID = %d", collectionID)
}

func (m *MetaReplica) setDeltaChannel(collectionID UniqueID, infos []*datapb.VchannelInfo) error {
	m.deltaChannelMu.Lock()
	defer m.deltaChannelMu.Unlock()
	_, ok := m.deltaChannelInfos[collectionID]
	if ok {
		log.Debug("delta channel already exist", zap.Any("collectionID", collectionID))
		return nil
	}

	err := saveDeltaChannelInfo(collectionID, infos, m.client)
	if err != nil {
		log.Error("save delta channel info error", zap.Int64("collectionID", collectionID), zap.Error(err))
		return err
	}
	log.Debug("save delta channel infos to meta", zap.Any("collectionID", collectionID))
	m.deltaChannelInfos[collectionID] = infos
	return nil
}

// Get Query channel info for collection, so far all the collection share the same query channel 0
func (m *MetaReplica) getQueryChannelInfoByID(collectionID UniqueID) *querypb.QueryChannelInfo {
	m.channelMu.Lock()
	defer m.channelMu.Unlock()

	var channelInfo *querypb.QueryChannelInfo
	if info, ok := m.queryChannelInfos[collectionID]; ok {
		channelInfo = proto.Clone(info).(*querypb.QueryChannelInfo)
	} else {
		channelInfo = m.createQueryChannel(collectionID)
		m.queryChannelInfos[collectionID] = channelInfo
	}

	return proto.Clone(channelInfo).(*querypb.QueryChannelInfo)
}

func (m *MetaReplica) getQueryStreamByID(collectionID UniqueID, queryChannel string) (msgstream.MsgStream, error) {
	m.streamMu.Lock()
	defer m.streamMu.Unlock()

	var queryStream msgstream.MsgStream
	var err error
	if stream, ok := m.queryStreams[collectionID]; ok {
		queryStream = stream
	} else {
		queryStream, err = m.msFactory.NewMsgStream(m.ctx)
		if err != nil {
			log.Error("updateGlobalSealedSegmentInfos: create msgStream failed", zap.Error(err))
			return nil, err
		}

		queryStream.AsProducer([]string{queryChannel})
		m.queryStreams[collectionID] = queryStream
		log.Debug("getQueryStreamByID: create query msgStream for collection", zap.Int64("collectionID", collectionID))
	}

	return queryStream, nil
}

func (m *MetaReplica) setLoadType(collectionID UniqueID, loadType querypb.LoadType) error {
	m.collectionMu.Lock()
	defer m.collectionMu.Unlock()

	if _, ok := m.collectionInfos[collectionID]; ok {
		info := proto.Clone(m.collectionInfos[collectionID]).(*querypb.CollectionInfo)
		info.LoadType = loadType
		err := saveGlobalCollectionInfo(collectionID, info, m.client)
		if err != nil {
			log.Error("save collectionInfo error", zap.Any("error", err.Error()), zap.Int64("collectionID", collectionID))
			return err
		}

		m.collectionInfos[collectionID] = info
		return nil
	}

	return errors.New("setLoadType: can't find collection in collectionInfos")
}

func (m *MetaReplica) setLoadPercentage(collectionID UniqueID, partitionID UniqueID, percentage int64, loadType querypb.LoadType) error {
	m.collectionMu.Lock()
	defer m.collectionMu.Unlock()

	if _, ok := m.collectionInfos[collectionID]; !ok {
		return errors.New("setLoadPercentage: can't find collection in collectionInfos")
	}

	info := proto.Clone(m.collectionInfos[collectionID]).(*querypb.CollectionInfo)
	if loadType == querypb.LoadType_loadCollection {
		info.InMemoryPercentage = percentage
		for _, partitionState := range info.PartitionStates {
			if percentage >= 100 {
				partitionState.State = querypb.PartitionState_InMemory
			} else {
				partitionState.State = querypb.PartitionState_PartialInMemory
			}
			partitionState.InMemoryPercentage = percentage
		}
		err := saveGlobalCollectionInfo(collectionID, info, m.client)
		if err != nil {
			log.Error("save collectionInfo error", zap.Any("error", err.Error()), zap.Int64("collectionID", collectionID))
			return err
		}
	} else {
		findPartition := false
		for _, partitionState := range info.PartitionStates {
			if partitionState.PartitionID == partitionID {
				findPartition = true
				if percentage >= 100 {
					partitionState.State = querypb.PartitionState_InMemory
				} else {
					partitionState.State = querypb.PartitionState_PartialInMemory
				}
				partitionState.InMemoryPercentage = percentage
				err := saveGlobalCollectionInfo(collectionID, info, m.client)
				if err != nil {
					log.Error("save collectionInfo error", zap.Any("error", err.Error()), zap.Int64("collectionID", collectionID))
					return err
				}
			}
		}
		if !findPartition {
			return errors.New("setLoadPercentage: can't find partitionID in collectionInfos")
		}
	}

	m.collectionInfos[collectionID] = info
	return nil
}

func (m *MetaReplica) getWatchedChannelsByNodeID(nodeID int64) *querypb.UnsubscribeChannelInfo {
	// 1. find all the search/dmChannel/deltaChannel the node has watched
	colID2DmChannels := make(map[UniqueID][]string)
	colID2DeltaChannels := make(map[UniqueID][]string)
	colID2QueryChannel := make(map[UniqueID]string)
	dmChannelInfos := m.getDmChannelInfosByNodeID(nodeID)
	// get dmChannel/search channel the node has watched
	for _, channelInfo := range dmChannelInfos {
		collectionID := channelInfo.CollectionID
		dmChannel := rootcoord.ToPhysicalChannel(channelInfo.DmChannel)
		if _, ok := colID2DmChannels[collectionID]; !ok {
			colID2DmChannels[collectionID] = []string{}
		}
		colID2DmChannels[collectionID] = append(colID2DmChannels[collectionID], dmChannel)
		if _, ok := colID2QueryChannel[collectionID]; !ok {
			queryChannelInfo := m.getQueryChannelInfoByID(collectionID)
			colID2QueryChannel[collectionID] = queryChannelInfo.QueryChannel
		}
	}
	segmentInfos := m.getSegmentInfosByNode(nodeID)
	// get delta/search channel the node has watched
	for _, segmentInfo := range segmentInfos {
		collectionID := segmentInfo.CollectionID
		if _, ok := colID2DeltaChannels[collectionID]; !ok {
			deltaChanelInfos, err := m.getDeltaChannelsByCollectionID(collectionID)
			if err != nil {
				// all nodes succeeded in releasing the Data, but queryCoord hasn't cleaned up the meta in time, and a Node went down
				// and meta was cleaned after m.getSegmentInfosByNode(nodeID)
				continue
			}
			deltaChannels := make([]string, len(deltaChanelInfos))
			for offset, channelInfo := range deltaChanelInfos {
				deltaChannels[offset] = rootcoord.ToPhysicalChannel(channelInfo.ChannelName)
			}
			colID2DeltaChannels[collectionID] = deltaChannels
		}
		if _, ok := colID2QueryChannel[collectionID]; !ok {
			queryChannelInfo := m.getQueryChannelInfoByID(collectionID)
			colID2QueryChannel[collectionID] = queryChannelInfo.QueryChannel
		}
	}

	// creating unsubscribeChannelInfo, which will be written to etcd
	colID2Channels := make(map[UniqueID][]string)
	for collectionID, channels := range colID2DmChannels {
		colID2Channels[collectionID] = append(colID2Channels[collectionID], channels...)
	}
	for collectionID, channels := range colID2DeltaChannels {
		colID2Channels[collectionID] = append(colID2Channels[collectionID], channels...)
	}
	for collectionID, channel := range colID2QueryChannel {
		colID2Channels[collectionID] = append(colID2Channels[collectionID], channel)
	}

	unsubscribeChannelInfo := &querypb.UnsubscribeChannelInfo{
		NodeID: nodeID,
	}

	for collectionID, channels := range colID2Channels {
		unsubscribeChannelInfo.CollectionChannels = append(unsubscribeChannelInfo.CollectionChannels,
			&querypb.UnsubscribeChannels{
				CollectionID: collectionID,
				Channels:     channels,
			})
	}

	return unsubscribeChannelInfo
}

//func (m *MetaReplica) printMeta() {
//	m.RLock()
//	defer m.RUnlock()
//	for id, info := range m.collectionInfos {
//		log.Debug("query coordinator MetaReplica: collectionInfo", zap.Int64("collectionID", id), zap.Any("info", info))
//	}
//
//	for id, info := range m.segmentInfos {
//		log.Debug("query coordinator MetaReplica: segmentInfo", zap.Int64("segmentID", id), zap.Any("info", info))
//	}
//
//	for id, info := range m.queryChannelInfos {
//		log.Debug("query coordinator MetaReplica: queryChannelInfo", zap.Int64("collectionID", id), zap.Any("info", info))
//	}
//}

func saveGlobalCollectionInfo(collectionID UniqueID, info *querypb.CollectionInfo, kv kv.MetaKv) error {
	infoBytes, err := proto.Marshal(info)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("%s/%d", collectionMetaPrefix, collectionID)
	return kv.Save(key, string(infoBytes))
}

func saveDeltaChannelInfo(collectionID UniqueID, infos []*datapb.VchannelInfo, kv kv.MetaKv) error {
	kvs := make(map[string]string)
	for _, info := range infos {
		infoBytes, err := proto.Marshal(info)
		if err != nil {
			return err
		}

		key := fmt.Sprintf("%s/%d/%s", deltaChannelMetaPrefix, collectionID, info.ChannelName)
		kvs[key] = string(infoBytes)
	}
	return kv.MultiSave(kvs)
}

func saveDmChannelWatchInfos(infos []*querypb.DmChannelWatchInfo, kv kv.MetaKv) error {
	kvs := make(map[string]string)
	for _, info := range infos {
		infoBytes, err := proto.Marshal(info)
		if err != nil {
			return err
		}

		key := fmt.Sprintf("%s/%d/%s", dmChannelMetaPrefix, info.CollectionID, info.DmChannel)
		kvs[key] = string(infoBytes)
	}
	return kv.MultiSave(kvs)
}

func removeCollectionMeta(collectionID UniqueID, kv kv.MetaKv) error {
	var prefixes []string
	collectionInfosPrefix := fmt.Sprintf("%s/%d", collectionMetaPrefix, collectionID)
	prefixes = append(prefixes, collectionInfosPrefix)
	dmChannelInfosPrefix := fmt.Sprintf("%s/%d", dmChannelMetaPrefix, collectionID)
	prefixes = append(prefixes, dmChannelInfosPrefix)
	deltaChannelInfosPrefix := fmt.Sprintf("%s/%d", deltaChannelMetaPrefix, collectionID)
	prefixes = append(prefixes, deltaChannelInfosPrefix)

	return kv.MultiRemoveWithPrefix(prefixes)
}
