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

package indexnode

import (
	"context"
	"errors"
	"fmt"
	"path"
	"runtime"
	"strconv"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/trace"
)

const (
	// paramsKeyToParse is the key of the param to build index.
	paramsKeyToParse = "params"

	// IndexBuildTaskName is the name of the operation to add an index task.
	IndexBuildTaskName = "IndexBuildTask"
)

type task interface {
	Ctx() context.Context
	ID() UniqueID // return ReqID
	Name() string
	SetID(uid UniqueID) // set ReqID
	PreExecute(ctx context.Context) error
	Execute(ctx context.Context) error
	PostExecute(ctx context.Context) error
	WaitToFinish() error
	Notify(err error)
	OnEnqueue() error
	SetError(err error)
}

// BaseTask is an basic instance of task.
type BaseTask struct {
	done        chan error
	ctx         context.Context
	id          UniqueID
	err         error
	internalErr error
}

// SetError sets an error to task.
func (bt *BaseTask) SetError(err error) {
	bt.err = err
}

// ID returns the id of index task.
func (bt *BaseTask) ID() UniqueID {
	return bt.id
}

func (bt *BaseTask) setID(id UniqueID) {
	bt.id = id
}

// WaitToFinish will wait for the task to complete, if the context is done, it means that the execution of the task has timed out.
func (bt *BaseTask) WaitToFinish() error {
	select {
	case <-bt.ctx.Done():
		return errors.New("timeout")
	case err := <-bt.done:
		return err
	}
}

// Notify will notify WaitToFinish that the task is completed or failed.
func (bt *BaseTask) Notify(err error) {
	bt.done <- err
}

// IndexBuildTask is used to record the information of the index tasks.
type IndexBuildTask struct {
	BaseTask
	index     Index
	kv        kv.BaseKV
	etcdKV    *etcdkv.EtcdKV
	savePaths []string
	req       *indexpb.CreateIndexRequest
	nodeID    UniqueID
}

// Ctx is the context of index tasks.
func (it *IndexBuildTask) Ctx() context.Context {
	return it.ctx
}

// ID returns the id of index task.
func (it *IndexBuildTask) ID() UniqueID {
	return it.id
}

// SetID sets the id for index task.
func (it *IndexBuildTask) SetID(ID UniqueID) {
	it.BaseTask.setID(ID)
}

// Name is the name of task to build index.
func (bt *BaseTask) Name() string {
	return IndexBuildTaskName
}

// OnEnqueue enqueues indexing tasks.
func (it *IndexBuildTask) OnEnqueue() error {
	it.SetID(it.req.IndexBuildID)
	log.Debug("IndexNode IndexBuilderTask Enqueue", zap.Int64("TaskID", it.ID()))
	return nil
}

func (it *IndexBuildTask) checkIndexMeta(ctx context.Context, pre bool) error {
	fn := func() error {
		//TODO error handling need to be optimized, return Unrecoverable to avoid retry
		indexMeta := indexpb.IndexMeta{}
		_, values, versions, err := it.etcdKV.LoadWithPrefix2(it.req.MetaPath)
		if err != nil {
			log.Error("IndexNode checkIndexMeta", zap.Any("load meta error with path", it.req.MetaPath),
				zap.Error(err), zap.Any("pre", pre))
			return err
		}
		if len(values) == 0 {
			return fmt.Errorf("IndexNode checkIndexMeta the indexMeta is empty")
		}
		log.Debug("IndexNode checkIndexMeta load meta success", zap.Any("path", it.req.MetaPath), zap.Any("pre", pre))
		err = proto.Unmarshal([]byte(values[0]), &indexMeta)
		if err != nil {
			log.Error("IndexNode failed to unmarshal index meta", zap.Error(err))
			return err
		}
		log.Debug("IndexNode checkIndexMeta Unmarshal success", zap.Any("IndexMeta", indexMeta))
		if indexMeta.Version > it.req.Version || indexMeta.State == commonpb.IndexState_Finished {
			log.Info("IndexNode checkIndexMeta version mismatch",
				zap.Any("req version", it.req.Version),
				zap.Any("index meta version", indexMeta.Version))
			return nil
		}
		if indexMeta.MarkDeleted {
			indexMeta.State = commonpb.IndexState_Finished
			v, err := proto.Marshal(&indexMeta)
			if err != nil {
				return err
			}
			err = it.etcdKV.CompareVersionAndSwap(it.req.MetaPath, versions[0], string(v))
			if err != nil {
				return err
			}
			errMsg := fmt.Sprintf("the index has been deleted with indexBuildID %d", indexMeta.IndexBuildID)
			log.Warn(errMsg)
			return fmt.Errorf(errMsg)
		}
		if pre {
			return nil
		}
		indexMeta.IndexFilePaths = it.savePaths
		indexMeta.State = commonpb.IndexState_Finished
		// Under normal circumstances, it.err and it.internalErr will not be non-nil at the same time, but for the sake of insurance, the else judgment is added.
		if it.err != nil {
			log.Error("IndexNode CreateIndex failed and can not be retried", zap.Int64("IndexBuildID", indexMeta.IndexBuildID), zap.Any("err", it.err))
			indexMeta.State = commonpb.IndexState_Failed
			indexMeta.FailReason = it.err.Error()
		} else if it.internalErr != nil {
			log.Error("IndexNode CreateIndex failed, but it can retried", zap.Int64("IndexBuildID", indexMeta.IndexBuildID), zap.Any("err", it.internalErr))
			indexMeta.State = commonpb.IndexState_Unissued
		}

		log.Debug("IndexNode", zap.Int64("indexBuildID", indexMeta.IndexBuildID), zap.Any("IndexState", indexMeta.State))
		var metaValue []byte
		metaValue, err = proto.Marshal(&indexMeta)
		if err != nil {
			log.Warn("IndexNode", zap.Int64("indexBuildID", indexMeta.IndexBuildID), zap.Any("IndexState", indexMeta.State),
				zap.Any("proto.Marshal failed:", err))
			return err
		}
		err = it.etcdKV.CompareVersionAndSwap(it.req.MetaPath, versions[0], string(metaValue))
		if err != nil {
			log.Warn("IndexNode checkIndexMeta CompareVersionAndSwap", zap.Error(err))
		}
		return err
	}

	err := retry.Do(ctx, fn, retry.Attempts(3))
	if err != nil {
		log.Error("IndexNode failed to checkIndexMeta", zap.Error(err))
	}
	return err
}

// PreExecute does some checks before building the index, for example, whether the index has been deleted.
func (it *IndexBuildTask) PreExecute(ctx context.Context) error {
	log.Debug("IndexNode IndexBuildTask preExecute...", zap.Int64("buildId", it.req.IndexBuildID))
	sp, ctx := trace.StartSpanFromContextWithOperationName(ctx, "CreateIndex-PreExecute")
	defer sp.Finish()
	return it.checkIndexMeta(ctx, true)
}

// PostExecute does some checks after building the index, for example, whether the index has been deleted or
// whether the index task is up to date.
func (it *IndexBuildTask) PostExecute(ctx context.Context) error {
	log.Debug("IndexNode IndexBuildTask PostExecute...", zap.Int64("buildId", it.req.IndexBuildID))
	sp, _ := trace.StartSpanFromContextWithOperationName(ctx, "CreateIndex-PostExecute")
	defer sp.Finish()
	return it.checkIndexMeta(ctx, false)
}

// Execute actually performs the task of building an index.
func (it *IndexBuildTask) Execute(ctx context.Context) error {
	log.Debug("IndexNode IndexBuildTask Execute ...", zap.Int64("buildId", it.req.IndexBuildID))
	sp, _ := trace.StartSpanFromContextWithOperationName(ctx, "CreateIndex-Execute")
	defer sp.Finish()
	tr := timerecord.NewTimeRecorder(fmt.Sprintf("IndexBuildTask %d", it.req.IndexBuildID))
	var err error

	typeParams := make(map[string]string)
	for _, kvPair := range it.req.GetTypeParams() {
		key, value := kvPair.GetKey(), kvPair.GetValue()
		_, ok := typeParams[key]
		if ok {
			return errors.New("duplicated key in type params")
		}
		if key == paramsKeyToParse {
			params, err := funcutil.ParseIndexParamsMap(value)
			if err != nil {
				return err
			}
			for pk, pv := range params {
				typeParams[pk] = pv
			}
		} else {
			typeParams[key] = value
		}
	}

	indexParams := make(map[string]string)
	for _, kvPair := range it.req.GetIndexParams() {
		key, value := kvPair.GetKey(), kvPair.GetValue()
		_, ok := indexParams[key]
		if ok {
			return errors.New("duplicated key in index params")
		}
		if key == paramsKeyToParse {
			params, err := funcutil.ParseIndexParamsMap(value)
			if err != nil {
				return err
			}
			for pk, pv := range params {
				indexParams[pk] = pv
			}
		} else {
			indexParams[key] = value
		}
	}

	it.index, err = NewCIndex(typeParams, indexParams)
	if err != nil {
		log.Error("IndexNode IndexBuildTask Execute NewCIndex failed",
			zap.Int64("buildId", it.req.IndexBuildID),
			zap.Error(err))
		return err
	}
	defer func() {
		err = it.index.Delete()
		if err != nil {
			log.Warn("IndexNode IndexBuildTask Execute CIndexDelete failed",
				zap.Int64("buildId", it.req.IndexBuildID),
				zap.Error(err))
		}
	}()

	getKeyByPathNaive := func(path string) string {
		// splitElements := strings.Split(path, "/")
		// return splitElements[len(splitElements)-1]
		return path
	}
	getValueByPath := func(path string) ([]byte, error) {
		data, err := it.kv.Load(path)
		if err != nil {
			return nil, err
		}
		return []byte(data), nil
	}
	getBlobByPath := func(path string) (*Blob, error) {
		value, err := getValueByPath(path)
		if err != nil {
			return nil, err
		}
		return &Blob{
			Key:   getKeyByPathNaive(path),
			Value: value,
		}, nil
	}
	getStorageBlobs := func(blobs []*Blob) []*storage.Blob {
		return blobs
	}

	toLoadDataPaths := it.req.GetDataPaths()
	keys := make([]string, len(toLoadDataPaths))
	blobs := make([]*Blob, len(toLoadDataPaths))

	loadKey := func(idx int) error {
		keys[idx] = getKeyByPathNaive(toLoadDataPaths[idx])
		blob, err := getBlobByPath(toLoadDataPaths[idx])
		if err != nil {
			return err
		}

		blobs[idx] = blob

		return nil
	}
	err = funcutil.ProcessFuncParallel(len(toLoadDataPaths), runtime.NumCPU(), loadKey, "loadKey")
	if err != nil {
		log.Warn("loadKey from minio failed", zap.Error(err))
		it.internalErr = err
		// In this case, it.internalErr is no longer nil and err does not need to be returned, otherwise it.err will also be assigned.
		return nil
	}
	log.Debug("IndexNode load data success", zap.Int64("buildId", it.req.IndexBuildID))
	tr.Record("loadKey done")

	storageBlobs := getStorageBlobs(blobs)
	var insertCodec storage.InsertCodec
	defer insertCodec.Close()
	collectionID, partitionID, segmentID, insertData, err2 := insertCodec.DeserializeAll(storageBlobs)
	if err2 != nil {
		return err2
	}
	if len(insertData.Data) != 1 {
		return errors.New("we expect only one field in deserialized insert data")
	}
	tr.Record("deserialize storage blobs done")

	for fieldID, value := range insertData.Data {
		// TODO: BinaryVectorFieldData
		floatVectorFieldData, fOk := value.(*storage.FloatVectorFieldData)
		if fOk {
			err = it.index.BuildFloatVecIndexWithoutIds(floatVectorFieldData.Data)
			if err != nil {
				log.Error("IndexNode BuildFloatVecIndexWithoutIds failed", zap.Error(err))
				return err
			}
			tr.Record("build float vector index done")
		}

		binaryVectorFieldData, bOk := value.(*storage.BinaryVectorFieldData)
		if bOk {
			err = it.index.BuildBinaryVecIndexWithoutIds(binaryVectorFieldData.Data)
			if err != nil {
				log.Error("IndexNode BuildBinaryVecIndexWithoutIds failed", zap.Error(err))
				return err
			}
			tr.Record("build binary vector index done")
		}

		if !fOk && !bOk {
			return errors.New("we expect FloatVectorFieldData or BinaryVectorFieldData")
		}

		indexBlobs, err := it.index.Serialize()
		if err != nil {
			log.Error("IndexNode index Serialize failed", zap.Error(err))
			return err
		}
		tr.Record("serialize index done")

		codec := storage.NewIndexFileBinlogCodec()
		serializedIndexBlobs, err := codec.Serialize(
			it.req.IndexBuildID,
			it.req.Version,
			collectionID,
			partitionID,
			segmentID,
			fieldID,
			indexParams,
			it.req.IndexName,
			it.req.IndexID,
			getStorageBlobs(indexBlobs),
		)
		if err != nil {
			return err
		}
		_ = codec.Close()
		tr.Record("serialize index codec done")

		getSavePathByKey := func(key string) string {

			return path.Join(Params.IndexStorageRootPath, strconv.Itoa(int(it.req.IndexBuildID)), strconv.Itoa(int(it.req.Version)),
				strconv.Itoa(int(partitionID)), strconv.Itoa(int(segmentID)), key)
		}
		saveBlob := func(path string, value []byte) error {
			return it.kv.Save(path, string(value))
		}

		it.savePaths = make([]string, len(serializedIndexBlobs))
		saveIndexFile := func(idx int) error {
			blob := serializedIndexBlobs[idx]
			key, value := blob.Key, blob.Value

			savePath := getSavePathByKey(key)

			saveIndexFileFn := func() error {
				v, err := it.etcdKV.Load(it.req.MetaPath)
				if err != nil {
					log.Warn("IndexNode load meta failed", zap.Any("path", it.req.MetaPath), zap.Error(err))
					return err
				}
				indexMeta := indexpb.IndexMeta{}
				err = proto.Unmarshal([]byte(v), &indexMeta)
				if err != nil {
					log.Warn("IndexNode Unmarshal indexMeta error ", zap.Error(err))
					return err
				}
				//log.Debug("IndexNode Unmarshal indexMeta success ", zap.Any("meta", indexMeta))
				if indexMeta.Version > it.req.Version {
					log.Warn("IndexNode try saveIndexFile failed req.Version is low", zap.Any("req.Version", it.req.Version),
						zap.Any("indexMeta.Version", indexMeta.Version))
					return errors.New("This task has been reassigned, check indexMeta.version and request ")
				}
				return saveBlob(savePath, value)
			}
			err := retry.Do(ctx, saveIndexFileFn, retry.Attempts(5))
			if err != nil {
				log.Warn("IndexNode try saveIndexFile final", zap.Error(err), zap.Any("savePath", savePath))
				return err
			}

			it.savePaths[idx] = savePath

			return nil
		}
		err = funcutil.ProcessFuncParallel(len(serializedIndexBlobs), runtime.NumCPU(), saveIndexFile, "saveIndexFile")
		if err != nil {
			log.Warn("saveIndexFile to minio failed", zap.Error(err))
			it.internalErr = err
			// In this case, it.internalErr is no longer nil and err does not need to be returned, otherwise it.err will also be assigned.
			return nil
		}
		tr.Record("save index file done")
	}
	log.Info("IndexNode CreateIndex successfully ", zap.Int64("collect", collectionID),
		zap.Int64("partition", partitionID), zap.Int64("segment", partitionID))
	tr.Elapse("all done")
	return nil
}
