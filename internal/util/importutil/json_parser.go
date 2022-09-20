package importutil

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/milvus-io/milvus/api/schemapb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
)

const (
	// root field of row-based json format
	RowRootNode = "rows"
	// minimal size of a buffer
	MinBufferSize = 1024
	// split file into batches no more than this count
	MaxBatchCount = 16
)

type JSONParser struct {
	ctx          context.Context  // for canceling parse process
	bufSize      int64            // max rows in a buffer
	fields       map[string]int64 // fields need to be parsed
	name2FieldID map[string]storage.FieldID
}

// NewJSONParser helper function to create a JSONParser
func NewJSONParser(ctx context.Context, collectionSchema *schemapb.CollectionSchema) *JSONParser {
	fields := make(map[string]int64)
	name2FieldID := make(map[string]storage.FieldID)
	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]
		fields[schema.GetName()] = 0
		name2FieldID[schema.GetName()] = schema.GetFieldID()
	}

	parser := &JSONParser{
		ctx:          ctx,
		bufSize:      MinBufferSize,
		fields:       fields,
		name2FieldID: name2FieldID,
	}
	adjustBufSize(parser, collectionSchema)

	return parser
}

func adjustBufSize(parser *JSONParser, collectionSchema *schemapb.CollectionSchema) {
	sizePerRecord, _ := typeutil.EstimateSizePerRecord(collectionSchema)
	if sizePerRecord <= 0 {
		return
	}

	// split the file into no more than MaxBatchCount batches to parse
	// for high dimensional vector, the bufSize is a small value, read few rows each time
	// for low dimensional vector, the bufSize is a large value, read more rows each time
	maxRows := MaxFileSize / sizePerRecord
	bufSize := maxRows / MaxBatchCount

	// bufSize should not be less than MinBufferSize
	if bufSize < MinBufferSize {
		bufSize = MinBufferSize
	}

	log.Info("JSON parse: reset bufSize", zap.Int("sizePerRecord", sizePerRecord), zap.Int("bufSize", bufSize))
	parser.bufSize = int64(bufSize)
}

func (p *JSONParser) logError(msg string) error {
	log.Error(msg)
	return errors.New(msg)
}

func (p *JSONParser) ParseRows(r io.Reader, handler JSONRowHandler) error {
	if handler == nil {
		return p.logError("JSON parse handler is nil")
	}

	dec := json.NewDecoder(r)

	t, err := dec.Token()
	if err != nil {
		return p.logError("JSON parse: row count is 0")
	}
	if t != json.Delim('{') {
		return p.logError("JSON parse: invalid JSON format, the content should be started with'{'")
	}

	// read the first level
	isEmpty := true
	for dec.More() {
		// read the key
		t, err := dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}
		key := t.(string)
		keyLower := strings.ToLower(key)

		// the root key should be RowRootNode
		if keyLower != RowRootNode {
			return p.logError("JSON parse: invalid row-based JSON format, the key " + key + " is not found")
		}

		// started by '['
		t, err = dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}

		if t != json.Delim('[') {
			return p.logError("JSON parse: invalid row-based JSON format, rows list should begin with '['")
		}

		// read buffer
		buf := make([]map[storage.FieldID]interface{}, 0, MinBufferSize)
		for dec.More() {
			var value interface{}
			if err := dec.Decode(&value); err != nil {
				return p.logError("JSON parse: " + err.Error())
			}

			switch value.(type) {
			case map[string]interface{}:
				break
			default:
				return p.logError("JSON parse: invalid JSON format, each row should be a key-value map")
			}

			row := make(map[storage.FieldID]interface{})
			stringMap := value.(map[string]interface{})
			for k, v := range stringMap {
				row[p.name2FieldID[k]] = v
			}

			buf = append(buf, row)
			if len(buf) >= int(p.bufSize) {
				isEmpty = false
				if err = handler.Handle(buf); err != nil {
					return p.logError(err.Error())
				}

				// clear the buffer
				buf = make([]map[storage.FieldID]interface{}, 0, MinBufferSize)
			}
		}

		// some rows in buffer not parsed, parse them
		if len(buf) > 0 {
			isEmpty = false
			if err = handler.Handle(buf); err != nil {
				return p.logError(err.Error())
			}
		}

		// end by ']'
		t, err = dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}

		if t != json.Delim(']') {
			return p.logError("JSON parse: invalid column-based JSON format, rows list should end with a ']'")
		}

		// canceled?
		select {
		case <-p.ctx.Done():
			return p.logError("import task was canceled")
		default:
			break
		}

		// this break means we require the first node must be RowRootNode
		// once the RowRootNode is parsed, just finish
		break
	}

	if isEmpty {
		return p.logError("JSON parse: row count is 0")
	}

	// send nil to notify the handler all have done
	return handler.Handle(nil)
}

func (p *JSONParser) ParseColumns(r io.Reader, handler JSONColumnHandler) error {
	if handler == nil {
		return p.logError("JSON parse handler is nil")
	}

	dec := json.NewDecoder(r)

	t, err := dec.Token()
	if err != nil {
		return p.logError("JSON parse: row count is 0")
	}
	if t != json.Delim('{') {
		return p.logError("JSON parse: invalid JSON format, the content should be started with'{'")
	}

	// read the first level
	isEmpty := true
	for dec.More() {
		// read the key
		t, err := dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}
		key := t.(string)

		// not a valid column name, skip
		_, isValidField := p.fields[key]

		// started by '['
		t, err = dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}

		if t != json.Delim('[') {
			return p.logError("JSON parse: invalid column-based JSON format, each field should begin with '['")
		}

		id := p.name2FieldID[key]
		// read buffer
		buf := make(map[storage.FieldID][]interface{})
		buf[id] = make([]interface{}, 0, MinBufferSize)
		for dec.More() {
			var value interface{}
			if err := dec.Decode(&value); err != nil {
				return p.logError("JSON parse: " + err.Error())
			}

			if !isValidField {
				continue
			}

			buf[id] = append(buf[id], value)
			if len(buf[id]) >= int(p.bufSize) {
				isEmpty = false
				if err = handler.Handle(buf); err != nil {
					return p.logError(err.Error())
				}

				// clear the buffer
				buf[id] = make([]interface{}, 0, MinBufferSize)
			}
		}

		// some values in buffer not parsed, parse them
		if len(buf[id]) > 0 {
			isEmpty = false
			if err = handler.Handle(buf); err != nil {
				return p.logError(err.Error())
			}
		}

		// end by ']'
		t, err = dec.Token()
		if err != nil {
			return p.logError("JSON parse: " + err.Error())
		}

		if t != json.Delim(']') {
			return p.logError("JSON parse: invalid column-based JSON format, each field should end with a ']'")
		}

		// canceled?
		select {
		case <-p.ctx.Done():
			return p.logError("import task was canceled")
		default:
			break
		}
	}

	if isEmpty {
		return p.logError("JSON parse: row count is 0")
	}

	// send nil to notify the handler all have done
	return handler.Handle(nil)
}
