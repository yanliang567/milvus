package importutil

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

const (
	// root field of row-based json format
	RowRootNode = "rows"
	// initial size of a buffer
	BufferSize = 1024
)

type JSONParser struct {
	ctx     context.Context  // for canceling parse process
	bufSize int64            // max rows in a buffer
	fields  map[string]int64 // fields need to be parsed
}

// newImportManager helper function to create a importManager
func NewJSONParser(ctx context.Context, collectionSchema *schemapb.CollectionSchema) *JSONParser {
	fields := make(map[string]int64)
	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]
		fields[schema.GetName()] = 0
	}

	parser := &JSONParser{
		ctx:     ctx,
		bufSize: 4096,
		fields:  fields,
	}

	return parser
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
		return p.logError("JSON parse: " + err.Error())
	}
	if t != json.Delim('{') {
		return p.logError("JSON parse: invalid JSON format, the content should be started with'{'")
	}

	// read the first level
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
		buf := make([]map[string]interface{}, 0, BufferSize)
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

			row := value.(map[string]interface{})

			buf = append(buf, row)
			if len(buf) >= int(p.bufSize) {
				if err = handler.Handle(buf); err != nil {
					return p.logError(err.Error())
				}

				// clear the buffer
				buf = make([]map[string]interface{}, 0, BufferSize)
			}
		}

		// some rows in buffer not parsed, parse them
		if len(buf) > 0 {
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
		return p.logError("JSON parse: " + err.Error())
	}
	if t != json.Delim('{') {
		return p.logError("JSON parse: invalid JSON format, the content should be started with'{'")
	}

	// read the first level
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

		// read buffer
		buf := make(map[string][]interface{})
		buf[key] = make([]interface{}, 0, BufferSize)
		for dec.More() {
			var value interface{}
			if err := dec.Decode(&value); err != nil {
				return p.logError("JSON parse: " + err.Error())
			}

			if !isValidField {
				continue
			}

			buf[key] = append(buf[key], value)
			if len(buf[key]) >= int(p.bufSize) {
				if err = handler.Handle(buf); err != nil {
					return p.logError(err.Error())
				}

				// clear the buffer
				buf[key] = make([]interface{}, 0, BufferSize)
			}
		}

		// some values in buffer not parsed, parse them
		if len(buf[key]) > 0 {
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

	// send nil to notify the handler all have done
	return handler.Handle(nil)
}
