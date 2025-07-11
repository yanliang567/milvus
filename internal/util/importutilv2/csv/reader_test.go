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

package csv

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/testutil"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/objectstorage"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type mockReader struct {
	io.Reader
	io.Closer
	io.ReaderAt
	io.Seeker
	size int64
}

func (mr *mockReader) Size() (int64, error) {
	return mr.size, nil
}

type ReaderSuite struct {
	suite.Suite

	numRows     int
	pkDataType  schemapb.DataType
	vecDataType schemapb.DataType
}

func (suite *ReaderSuite) SetupSuite() {
	paramtable.Get().Init(paramtable.NewBaseTable())
}

func (suite *ReaderSuite) SetupTest() {
	suite.numRows = 100
	suite.pkDataType = schemapb.DataType_Int64
	suite.vecDataType = schemapb.DataType_FloatVector
}

func (suite *ReaderSuite) run(dataType schemapb.DataType, elemType schemapb.DataType, nullable bool) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      100,
				Name:         "pk",
				IsPrimaryKey: true,
				DataType:     suite.pkDataType,
				TypeParams: []*commonpb.KeyValuePair{
					{
						Key:   common.MaxLengthKey,
						Value: "128",
					},
				},
			},
			{
				FieldID:  101,
				Name:     "vec",
				DataType: suite.vecDataType,
				TypeParams: []*commonpb.KeyValuePair{
					{
						Key:   common.DimKey,
						Value: "8",
					},
				},
			},
			{
				FieldID:     102,
				Name:        dataType.String(),
				DataType:    dataType,
				ElementType: elemType,
				TypeParams: []*commonpb.KeyValuePair{
					{
						Key:   common.MaxLengthKey,
						Value: "128",
					},
					{
						Key:   common.MaxCapacityKey,
						Value: "256",
					},
				},
				Nullable: nullable,
			},
		},
	}
	if dataType == schemapb.DataType_VarChar {
		// Add a BM25 function if data type is VarChar
		schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
			FieldID:          103,
			Name:             "sparse",
			DataType:         schemapb.DataType_SparseFloatVector,
			IsFunctionOutput: true,
		})
		schema.Functions = append(schema.Functions, &schemapb.FunctionSchema{
			Id:               1000,
			Name:             "bm25",
			Type:             schemapb.FunctionType_BM25,
			InputFieldIds:    []int64{102},
			InputFieldNames:  []string{dataType.String()},
			OutputFieldIds:   []int64{103},
			OutputFieldNames: []string{"sparse"},
		})
	}

	// config
	// csv separator
	sep := ','
	// csv writer write null value as empty string
	nullkey := ""

	// generate csv data
	insertData, err := testutil.CreateInsertData(schema, suite.numRows)
	suite.NoError(err)
	csvData, err := testutil.CreateInsertDataForCSV(schema, insertData, nullkey)
	suite.NoError(err)

	// write to csv file
	filePath := fmt.Sprintf("/tmp/test_%d_reader.csv", rand.Int())
	defer os.Remove(filePath)
	wf, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0o666)
	suite.NoError(err)
	writer := csv.NewWriter(wf)
	writer.Comma = sep
	err = writer.WriteAll(csvData)
	suite.NoError(err)

	// read from csv file
	ctx := context.Background()
	f := storage.NewChunkManagerFactory("local", objectstorage.RootPath("/tmp/milvus_test/test_csv_reader/"))
	cm, err := f.NewPersistentStorageChunkManager(ctx)
	suite.NoError(err)

	// check reader separate fields by '\t'
	wrongSep := '\t'
	_, err = NewReader(ctx, cm, schema, filePath, 64*1024*1024, wrongSep, nullkey)
	suite.Error(err)
	suite.Contains(err.Error(), "value of field is missed:")

	// check data
	reader, err := NewReader(ctx, cm, schema, filePath, 64*1024*1024, sep, nullkey)
	suite.NoError(err)

	checkFn := func(actualInsertData *storage.InsertData, offsetBegin, expectRows int) {
		expectInsertData := insertData
		for fieldID, data := range actualInsertData.Data {
			suite.Equal(expectRows, data.RowNum())
			for i := 0; i < expectRows; i++ {
				expect := expectInsertData.Data[fieldID].GetRow(i + offsetBegin)
				actual := data.GetRow(i)
				suite.Equal(expect, actual)
			}
		}
	}

	res, err := reader.Read()
	suite.NoError(err)
	checkFn(res, 0, suite.numRows)
}

func (suite *ReaderSuite) TestReadScalarFields() {
	suite.run(schemapb.DataType_Bool, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Int8, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Int16, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Int64, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Float, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_Double, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_String, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_VarChar, schemapb.DataType_None, false)
	suite.run(schemapb.DataType_JSON, schemapb.DataType_None, false)

	suite.run(schemapb.DataType_Array, schemapb.DataType_Bool, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int8, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int16, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int32, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int64, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Float, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Double, false)
	suite.run(schemapb.DataType_Array, schemapb.DataType_String, false)

	suite.run(schemapb.DataType_Bool, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Int8, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Int16, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Int64, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Float, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_Double, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_String, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_VarChar, schemapb.DataType_None, true)
	suite.run(schemapb.DataType_JSON, schemapb.DataType_None, true)

	suite.run(schemapb.DataType_Array, schemapb.DataType_Bool, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int8, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int16, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int32, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Int64, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Float, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_Double, true)
	suite.run(schemapb.DataType_Array, schemapb.DataType_String, true)
}

func (suite *ReaderSuite) TestStringPK() {
	suite.pkDataType = schemapb.DataType_VarChar
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
}

func (suite *ReaderSuite) TestVector() {
	suite.vecDataType = schemapb.DataType_BinaryVector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.vecDataType = schemapb.DataType_FloatVector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.vecDataType = schemapb.DataType_Float16Vector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.vecDataType = schemapb.DataType_BFloat16Vector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.vecDataType = schemapb.DataType_SparseFloatVector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
	suite.vecDataType = schemapb.DataType_Int8Vector
	suite.run(schemapb.DataType_Int32, schemapb.DataType_None, false)
}

func (suite *ReaderSuite) TestError() {
	testNewReaderErr := func(ioErr error, schema *schemapb.CollectionSchema, content string, bufferSize int) {
		cm := mocks.NewChunkManager(suite.T())
		cm.EXPECT().Reader(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, s string) (storage.FileReader, error) {
			if ioErr != nil {
				return nil, ioErr
			} else {
				r := &mockReader{Reader: strings.NewReader(content)}
				return r, nil
			}
		})

		reader, err := NewReader(context.Background(), cm, schema, "dummy path", bufferSize, ',', "")
		suite.Error(err)
		suite.Nil(reader)
	}

	testNewReaderErr(merr.WrapErrImportFailed("error"), nil, "", 1)

	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      100,
				Name:         "pk",
				IsPrimaryKey: true,
				DataType:     schemapb.DataType_Int64,
			},
			{
				FieldID:  101,
				Name:     "int32",
				DataType: schemapb.DataType_Int32,
			},
		},
	}
	testNewReaderErr(nil, schema, "", -1)
	testNewReaderErr(nil, schema, "", 1)

	testReadErr := func(schema *schemapb.CollectionSchema, content string) {
		cm := mocks.NewChunkManager(suite.T())
		cm.EXPECT().Reader(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, s string) (storage.FileReader, error) {
			r := &mockReader{Reader: strings.NewReader(content)}
			return r, nil
		})
		cm.EXPECT().Size(mock.Anything, mock.Anything).Return(128, nil)

		reader, err := NewReader(context.Background(), cm, schema, "dummy path", 1024, ',', "")
		suite.NoError(err)
		suite.NotNil(reader)

		size, err := reader.Size()
		suite.NoError(err)
		suite.Equal(int64(128), size)

		_, err = reader.Read()
		suite.Error(err)
	}
	content := "pk,int32\n1,A"
	testReadErr(schema, content)
}

func (suite *ReaderSuite) TestReadLoop() {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      100,
				Name:         "pk",
				IsPrimaryKey: true,
				DataType:     schemapb.DataType_Int64,
			},
			{
				FieldID:  101,
				Name:     "float",
				DataType: schemapb.DataType_Float,
			},
		},
	}
	content := "pk,float\n1,0.1\n2,0.2"

	cm := mocks.NewChunkManager(suite.T())
	cm.EXPECT().Reader(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, s string) (storage.FileReader, error) {
		reader := strings.NewReader(content)
		r := &mockReader{
			Reader: reader,
			Closer: io.NopCloser(reader),
		}
		return r, nil
	})
	cm.EXPECT().Size(mock.Anything, mock.Anything).Return(128, nil)

	reader, err := NewReader(context.Background(), cm, schema, "dummy path", 1, ',', "")
	suite.NoError(err)
	suite.NotNil(reader)
	defer reader.Close()

	size, err := reader.Size()
	suite.NoError(err)
	suite.Equal(int64(128), size)
	size2, err := reader.Size() // size is cached
	suite.NoError(err)
	suite.Equal(size, size2)

	reader.count = 1
	data, err := reader.Read()
	suite.NoError(err)
	suite.Equal(1, data.GetRowNum())

	data, err = reader.Read()
	suite.NoError(err)
	suite.Equal(1, data.GetRowNum())

	data, err = reader.Read()
	suite.EqualError(io.EOF, err.Error())
	suite.Nil(data)
}

func TestCsvReader(t *testing.T) {
	suite.Run(t, new(ReaderSuite))
}
