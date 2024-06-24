//go:build test
// +build test

package walimplstest

import (
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/helper"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/walimpls"
	"github.com/milvus-io/milvus/internal/util/streamingutil/message"
)

var _ walimpls.ScannerImpls = &scannerImpls{}

func newScannerImpls(opts walimpls.ReadOption, data *messageLog, offset int) *scannerImpls {
	s := &scannerImpls{
		ScannerHelper: helper.NewScannerHelper(opts.Name),
		datas:         data,
		ch:            make(chan message.ImmutableMessage),
		offset:        offset,
	}
	go s.executeConsume()
	return s
}

type scannerImpls struct {
	*helper.ScannerHelper
	datas  *messageLog
	ch     chan message.ImmutableMessage
	offset int
}

func (s *scannerImpls) executeConsume() {
	defer close(s.ch)
	for {
		msg, err := s.datas.ReadAt(s.Context(), s.offset)
		if err != nil {
			s.Finish(nil)
			return
		}
		s.ch <- msg
		s.offset++
	}
}

func (s *scannerImpls) Chan() <-chan message.ImmutableMessage {
	return s.ch
}

func (s *scannerImpls) Close() error {
	return s.ScannerHelper.Close()
}
