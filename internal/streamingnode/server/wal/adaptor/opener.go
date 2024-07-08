package adaptor

import (
	"context"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/streamingnode/server/wal"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/interceptors"
	"github.com/milvus-io/milvus/internal/util/streamingutil/status"
	"github.com/milvus-io/milvus/internal/util/streamingutil/util"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/streaming/walimpls"
	"github.com/milvus-io/milvus/pkg/util/lifetime"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

var _ wal.Opener = (*openerAdaptorImpl)(nil)

// adaptImplsToOpener creates a new wal opener with opener impls.
func adaptImplsToOpener(opener walimpls.OpenerImpls, builders []interceptors.InterceptorBuilder) wal.Opener {
	return &openerAdaptorImpl{
		lifetime:            lifetime.NewLifetime(lifetime.Working),
		opener:              opener,
		idAllocator:         util.NewIDAllocator(),
		walInstances:        typeutil.NewConcurrentMap[int64, wal.WAL](),
		interceptorBuilders: builders,
	}
}

// openerAdaptorImpl is the wrapper of OpenerImpls to Opener.
type openerAdaptorImpl struct {
	lifetime            lifetime.Lifetime[lifetime.State]
	opener              walimpls.OpenerImpls
	idAllocator         *util.IDAllocator
	walInstances        *typeutil.ConcurrentMap[int64, wal.WAL] // store all wal instances allocated by these allocator.
	interceptorBuilders []interceptors.InterceptorBuilder
}

// Open opens a wal instance for the channel.
func (o *openerAdaptorImpl) Open(ctx context.Context, opt *wal.OpenOption) (wal.WAL, error) {
	if o.lifetime.Add(lifetime.IsWorking) != nil {
		return nil, status.NewOnShutdownError("wal opener is on shutdown")
	}
	defer o.lifetime.Done()

	id := o.idAllocator.Allocate()
	log := log.With(zap.Any("channel", opt.Channel), zap.Int64("id", id))

	l, err := o.opener.Open(ctx, &walimpls.OpenOption{
		Channel: opt.Channel,
	})
	if err != nil {
		log.Warn("open wal failed", zap.Error(err))
		return nil, err
	}

	// wrap the wal into walExtend with cleanup function and interceptors.
	wal := adaptImplsToWAL(l, o.interceptorBuilders, func() {
		o.walInstances.Remove(id)
		log.Info("wal deleted from allocator")
	})

	o.walInstances.Insert(id, wal)
	log.Info("new wal created")
	metrics.StreamingNodeWALTotal.WithLabelValues(paramtable.GetStringNodeID()).Inc()
	return wal, nil
}

// Close the wal opener, release the underlying resources.
func (o *openerAdaptorImpl) Close() {
	o.lifetime.SetState(lifetime.Stopped)
	o.lifetime.Wait()
	o.lifetime.Close()

	// close all wal instances.
	o.walInstances.Range(func(id int64, l wal.WAL) bool {
		l.Close()
		log.Info("close wal by opener", zap.Int64("id", id), zap.Any("channel", l.Channel()))
		return true
	})
	// close the opener
	o.opener.Close()
}
