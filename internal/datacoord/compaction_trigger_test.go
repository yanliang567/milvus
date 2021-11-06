package datacoord

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/stretchr/testify/assert"
)

type spyCompactionHandler struct {
	spyChan chan *datapb.CompactionPlan
}

// execCompactionPlan start to execute plan and return immediately
func (h *spyCompactionHandler) execCompactionPlan(plan *datapb.CompactionPlan) error {
	h.spyChan <- plan
	return nil
}

// completeCompaction record the result of a compaction
func (h *spyCompactionHandler) completeCompaction(result *datapb.CompactionResult) error {
	return nil
}

// getCompaction return compaction task. If planId does not exist, return nil.
func (h *spyCompactionHandler) getCompaction(planID int64) *compactionTask {
	panic("not implemented") // TODO: Implement
}

// expireCompaction set the compaction state to expired
func (h *spyCompactionHandler) expireCompaction(ts Timestamp) error {
	panic("not implemented") // TODO: Implement
}

// isFull return true if the task pool is full
func (h *spyCompactionHandler) isFull() bool {
	return false
}

// get compaction by signal id and return the number of executing/completed/timeout plans
func (h *spyCompactionHandler) getCompactionBySignalID(signalID int64) (executing int, completed int, timeout int) {
	panic("not implemented") // TODO: Implement
}

func (h *spyCompactionHandler) start() {}

func (h *spyCompactionHandler) stop() {}

var _ compactionPlanContext = (*spyCompactionHandler)(nil)

func Test_compactionTrigger_forceTriggerCompaction(t *testing.T) {
	type fields struct {
		meta                   *meta
		allocator              allocator
		signals                chan *compactionSignal
		singleCompactionPolicy singleCompactionPolicy
		mergeCompactionPolicy  mergeCompactionPolicy
		compactionHandler      compactionPlanContext
		globalTrigger          *time.Ticker
	}
	type args struct {
		collectionID int64
		timetravel   *timetravel
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantErr   bool
		wantPlans []*datapb.CompactionPlan
	}{
		{
			"test only merge compaction",
			fields{
				&meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							1: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             1,
									CollectionID:   2,
									PartitionID:    1,
									LastExpireTime: 100,
									NumOfRows:      100,
									MaxRowNum:      300,
									InsertChannel:  "ch1",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"log1"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5, DeltaLogPath: "deltalog1"},
									},
								},
							},
							2: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             2,
									CollectionID:   2,
									PartitionID:    1,
									LastExpireTime: 100,
									NumOfRows:      100,
									MaxRowNum:      300,
									InsertChannel:  "ch1",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"log2"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5, DeltaLogPath: "deltalog2"},
									},
								},
							},
						},
					},
				},
				newMockAllocator(),
				nil,
				(singleCompactionFunc)(chooseAllBinlogs),
				(mergeCompactionFunc)(greedyGeneratePlans),
				&spyCompactionHandler{spyChan: make(chan *datapb.CompactionPlan, 1)},
				nil,
			},
			args{
				2,
				&timetravel{time: 200},
			},
			false,
			[]*datapb.CompactionPlan{
				{
					PlanID: 2,
					SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
						{
							SegmentID: 1,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"log1"}},
							},
							Field2StatslogPaths: nil,
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5, DeltaLogPath: "deltalog1"},
							},
						},
						{
							SegmentID: 2,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"log2"}},
							},
							Field2StatslogPaths: nil,
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5, DeltaLogPath: "deltalog2"},
							},
						},
					},
					TimeoutInSeconds: maxCompactionTimeoutInSeconds,
					Type:             datapb.CompactionType_MergeCompaction,
					Timetravel:       200,
					Channel:          "ch1",
				},
			},
		},
		{
			"test only single compaction",
			fields{
				&meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							1: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             1,
									CollectionID:   2,
									LastExpireTime: 100,
									NumOfRows:      10,
									InsertChannel:  "ch1",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"log1"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5},
									},
								},
							},
						},
					},
				},
				newMockAllocator(),
				nil,
				(singleCompactionFunc)(chooseAllBinlogs),
				(mergeCompactionFunc)(greedyGeneratePlans),
				&spyCompactionHandler{spyChan: make(chan *datapb.CompactionPlan, 1)},
				nil,
			},
			args{
				2,
				&timetravel{time: 200},
			},
			false,
			[]*datapb.CompactionPlan{
				{
					PlanID: 2,
					SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
						{
							SegmentID: 1,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"log1"}},
							},
							Field2StatslogPaths: nil,
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5},
							},
						},
					},
					TimeoutInSeconds: maxCompactionTimeoutInSeconds,
					Type:             datapb.CompactionType_InnerCompaction,
					Timetravel:       200,
					Channel:          "ch1",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := &compactionTrigger{
				meta:                   tt.fields.meta,
				allocator:              tt.fields.allocator,
				signals:                tt.fields.signals,
				singleCompactionPolicy: tt.fields.singleCompactionPolicy,
				mergeCompactionPolicy:  tt.fields.mergeCompactionPolicy,
				compactionHandler:      tt.fields.compactionHandler,
				globalTrigger:          tt.fields.globalTrigger,
			}
			_, err := tr.forceTriggerCompaction(tt.args.collectionID, tt.args.timetravel)
			assert.Equal(t, tt.wantErr, err != nil)
			spy := (tt.fields.compactionHandler).(*spyCompactionHandler)
			plan := <-spy.spyChan
			sortPlanCompactionBinlogs(plan)
			assert.EqualValues(t, tt.wantPlans[0], plan)
		})
	}
}

func sortPlanCompactionBinlogs(plan *datapb.CompactionPlan) {
	sort.Slice(plan.SegmentBinlogs, func(i, j int) bool {
		return plan.SegmentBinlogs[i].SegmentID < plan.SegmentBinlogs[j].SegmentID
	})
}

func Test_compactionTrigger_triggerCompaction(t *testing.T) {
	type fields struct {
		meta                            *meta
		allocator                       allocator
		signals                         chan *compactionSignal
		singleCompactionPolicy          singleCompactionPolicy
		mergeCompactionPolicy           mergeCompactionPolicy
		compactionHandler               compactionPlanContext
		mergeCompactionSegmentThreshold int
	}
	type args struct {
		timetravel *timetravel
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantErr   bool
		wantPlans []*datapb.CompactionPlan
	}{
		{
			"test normal case",
			fields{
				&meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							1: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             1,
									CollectionID:   1,
									PartitionID:    1,
									LastExpireTime: 100,
									NumOfRows:      10,
									MaxRowNum:      100,
									InsertChannel:  "ch1",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"binlog1"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5, DeltaLogPath: "deltalog1"},
									},
								},
							},
							2: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             2,
									CollectionID:   2,
									PartitionID:    1,
									LastExpireTime: 100,
									NumOfRows:      300,
									MaxRowNum:      1000,
									InsertChannel:  "ch2",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"binlog2"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5, DeltaLogPath: "deltalog2"},
									},
								},
							},
							3: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             3,
									CollectionID:   2,
									PartitionID:    1,
									LastExpireTime: 100,
									NumOfRows:      300,
									MaxRowNum:      1000,
									InsertChannel:  "ch2",
									State:          commonpb.SegmentState_Flushed,
									Binlogs: []*datapb.FieldBinlog{
										{FieldID: 1, Binlogs: []string{"binlog3"}},
									},
									Deltalogs: []*datapb.DeltaLogInfo{
										{RecordEntries: 5, DeltaLogPath: "deltalog3"},
									},
								},
							},
						},
					},
				},
				newMockAllocator(),
				make(chan *compactionSignal, 1),
				(singleCompactionFunc)(chooseAllBinlogs),
				(mergeCompactionFunc)(greedyMergeCompaction),
				&spyCompactionHandler{spyChan: make(chan *datapb.CompactionPlan, 2)},
				2,
			},
			args{
				&timetravel{200},
			},
			false,
			[]*datapb.CompactionPlan{
				{
					PlanID: 2,
					SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
						{
							SegmentID: 1,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"binlog1"}},
							},
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5, DeltaLogPath: "deltalog1"},
							},
						},
					},
					TimeoutInSeconds: maxCompactionTimeoutInSeconds,
					Type:             datapb.CompactionType_InnerCompaction,
					Timetravel:       200,
					Channel:          "ch1",
				},
				{
					PlanID: 3,
					SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
						{
							SegmentID: 2,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"binlog2"}},
							},
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5, DeltaLogPath: "deltalog2"},
							},
						},
						{
							SegmentID: 3,
							FieldBinlogs: []*datapb.FieldBinlog{
								{FieldID: 1, Binlogs: []string{"binlog3"}},
							},
							Deltalogs: []*datapb.DeltaLogInfo{
								{RecordEntries: 5, DeltaLogPath: "deltalog3"},
							},
						},
					},
					TimeoutInSeconds: maxCompactionTimeoutInSeconds,
					Type:             datapb.CompactionType_MergeCompaction,
					Timetravel:       200,
					Channel:          "ch2",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := &compactionTrigger{
				meta:                            tt.fields.meta,
				allocator:                       tt.fields.allocator,
				signals:                         tt.fields.signals,
				singleCompactionPolicy:          tt.fields.singleCompactionPolicy,
				mergeCompactionPolicy:           tt.fields.mergeCompactionPolicy,
				compactionHandler:               tt.fields.compactionHandler,
				mergeCompactionSegmentThreshold: tt.fields.mergeCompactionSegmentThreshold,
			}
			tr.start()
			defer tr.stop()
			err := tr.triggerCompaction(tt.args.timetravel)
			assert.Equal(t, tt.wantErr, err != nil)
			spy := tt.fields.compactionHandler.(*spyCompactionHandler)
			gotPlans := make([]*datapb.CompactionPlan, 0, len(tt.wantPlans))
			for i := 0; i < len(tt.wantPlans); i++ {
				plan := <-spy.spyChan
				gotPlans = append(gotPlans, plan)
			}
			for _, plan := range gotPlans {
				sortPlanCompactionBinlogs(plan)
			}
			assert.EqualValues(t, tt.wantPlans, gotPlans)
		})
	}
}

func Test_compactionTrigger_singleTriggerCompaction(t *testing.T) {
	type fields struct {
		meta                   *meta
		allocator              allocator
		signals                chan *compactionSignal
		singleCompactionPolicy singleCompactionPolicy
		mergeCompactionPolicy  mergeCompactionPolicy
		compactionHandler      compactionPlanContext
		globalTrigger          *time.Ticker
	}
	type args struct {
		collectionID int64
		partitionID  int64
		segmentID    int64
		channelName  string
		timetravel   *timetravel
	}
	tests := []struct {
		name        string
		fields      fields
		args        args
		wantErr     bool
		wantPlan    bool
		wantBinlogs []*datapb.CompactionSegmentBinlogs
	}{
		{
			name: "normal single flush",
			fields: fields{
				meta: &meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							101: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             101,
									CollectionID:   1,
									PartitionID:    10,
									InsertChannel:  "test_chan_01",
									NumOfRows:      10000,
									State:          commonpb.SegmentState_Flushed,
									MaxRowNum:      12000,
									LastExpireTime: 50,
									StartPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   10,
									},
									DmlPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   45,
									},
									Binlogs:   []*datapb.FieldBinlog{},
									Statslogs: []*datapb.FieldBinlog{},
									Deltalogs: []*datapb.DeltaLogInfo{
										{
											RecordEntries: 2001,
										},
									},
									CreatedByCompaction: false,
									CompactionFrom:      []int64{},
								},
								isCompacting: false,
							},
						},
					},
				},
				allocator:              newMockAllocator(),
				signals:                make(chan *compactionSignal, 1),
				singleCompactionPolicy: (singleCompactionFunc)(chooseAllBinlogs),
				mergeCompactionPolicy:  (mergeCompactionFunc)(greedyGeneratePlans),
				compactionHandler: &spyCompactionHandler{
					spyChan: make(chan *datapb.CompactionPlan, 1),
				},
				globalTrigger: time.NewTicker(time.Hour),
			},
			args: args{
				collectionID: 1,
				partitionID:  10,
				segmentID:    101,
				channelName:  "test_ch_01",
				timetravel: &timetravel{
					time: 100,
				},
			},
			wantErr:  false,
			wantPlan: true,
			wantBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           101,
					FieldBinlogs:        []*datapb.FieldBinlog{},
					Field2StatslogPaths: []*datapb.FieldBinlog{},
					Deltalogs: []*datapb.DeltaLogInfo{
						{
							RecordEntries: 2001,
						},
					},
				},
			},
		},
		{
			name: "part delta out of range",
			fields: fields{
				meta: &meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							101: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             101,
									CollectionID:   1,
									PartitionID:    10,
									InsertChannel:  "test_chan_01",
									NumOfRows:      10000,
									State:          commonpb.SegmentState_Flushed,
									MaxRowNum:      12000,
									LastExpireTime: 100,
									StartPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   10,
									},
									DmlPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   45,
									},
									Binlogs:   []*datapb.FieldBinlog{},
									Statslogs: []*datapb.FieldBinlog{},
									Deltalogs: []*datapb.DeltaLogInfo{
										{
											RecordEntries: 1000,
											TimestampFrom: 10,
											TimestampTo:   20,
										},
										{
											RecordEntries: 1001,
											TimestampFrom: 30,
											TimestampTo:   45,
										},
									},
									CreatedByCompaction: false,
									CompactionFrom:      []int64{},
								},
								isCompacting: false,
							},
						},
					},
				},
				allocator:              newMockAllocator(),
				signals:                make(chan *compactionSignal, 1),
				singleCompactionPolicy: (singleCompactionFunc)(chooseAllBinlogs),
				mergeCompactionPolicy:  (mergeCompactionFunc)(greedyGeneratePlans),
				compactionHandler: &spyCompactionHandler{
					spyChan: make(chan *datapb.CompactionPlan, 1),
				},
				globalTrigger: time.NewTicker(time.Hour),
			},
			args: args{
				collectionID: 1,
				partitionID:  10,
				segmentID:    101,
				channelName:  "test_ch_01",
				timetravel: &timetravel{
					time: 30,
				},
			},
			wantErr:  false,
			wantPlan: false,
			wantBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           101,
					FieldBinlogs:        []*datapb.FieldBinlog{},
					Field2StatslogPaths: []*datapb.FieldBinlog{},
					Deltalogs: []*datapb.DeltaLogInfo{
						{
							RecordEntries: 2001,
						},
					},
				},
			},
		},
		{
			name: "delte log size",
			fields: fields{
				meta: &meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							101: {
								SegmentInfo: &datapb.SegmentInfo{
									ID:             101,
									CollectionID:   1,
									PartitionID:    10,
									InsertChannel:  "test_chan_01",
									NumOfRows:      10000,
									State:          commonpb.SegmentState_Flushed,
									MaxRowNum:      12000,
									LastExpireTime: 100,
									StartPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   10,
									},
									DmlPosition: &internalpb.MsgPosition{
										ChannelName: "",
										MsgID:       []byte{},
										MsgGroup:    "",
										Timestamp:   45,
									},
									Binlogs:   []*datapb.FieldBinlog{},
									Statslogs: []*datapb.FieldBinlog{},
									Deltalogs: []*datapb.DeltaLogInfo{
										{
											RecordEntries: 1000,
											TimestampFrom: 10,
											TimestampTo:   20,
											DeltaLogSize:  10*1024*1024 + 1,
										},
									},
									CreatedByCompaction: false,
									CompactionFrom:      []int64{},
								},
								isCompacting: false,
							},
						},
					},
				},
				allocator:              newMockAllocator(),
				signals:                make(chan *compactionSignal, 1),
				singleCompactionPolicy: (singleCompactionFunc)(chooseAllBinlogs),
				mergeCompactionPolicy:  (mergeCompactionFunc)(greedyGeneratePlans),
				compactionHandler: &spyCompactionHandler{
					spyChan: make(chan *datapb.CompactionPlan, 1),
				},
				globalTrigger: time.NewTicker(time.Hour),
			},
			args: args{
				collectionID: 1,
				partitionID:  10,
				segmentID:    101,
				channelName:  "test_ch_01",
				timetravel: &timetravel{
					time: 120,
				},
			},
			wantErr:  false,
			wantPlan: true,
			wantBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           101,
					FieldBinlogs:        []*datapb.FieldBinlog{},
					Field2StatslogPaths: []*datapb.FieldBinlog{},
					Deltalogs: []*datapb.DeltaLogInfo{
						{
							RecordEntries: 1000,
							TimestampFrom: 10,
							TimestampTo:   20,
							DeltaLogSize:  10*1024*1024 + 1,
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := &compactionTrigger{
				meta:                   tt.fields.meta,
				allocator:              tt.fields.allocator,
				signals:                tt.fields.signals,
				singleCompactionPolicy: tt.fields.singleCompactionPolicy,
				mergeCompactionPolicy:  tt.fields.mergeCompactionPolicy,
				compactionHandler:      tt.fields.compactionHandler,
				globalTrigger:          tt.fields.globalTrigger,
			}
			tr.start()
			defer tr.stop()

			err := tr.triggerSingleCompaction(tt.args.collectionID, tt.args.partitionID,
				tt.args.segmentID, tt.args.channelName, tt.args.timetravel)
			assert.Equal(t, tt.wantErr, err != nil)
			spy := (tt.fields.compactionHandler).(*spyCompactionHandler)
			ctx, cancel := context.WithTimeout(context.TODO(), time.Millisecond*50)
			select {
			case plan := <-spy.spyChan:
				if tt.wantPlan {
					assert.EqualValues(t, tt.wantBinlogs, plan.GetSegmentBinlogs())
				} else {
					t.Fail()
				}
			case <-ctx.Done():
				if tt.wantPlan {
					t.Fail()
				}
			}
			cancel()
		})
	}
}

func Test_newCompactionTrigger(t *testing.T) {
	type args struct {
		meta              *meta
		compactionHandler compactionPlanContext
		allocator         allocator
	}
	tests := []struct {
		name string
		args args
	}{
		{
			"test new trigger",
			args{
				&meta{},
				&compactionPlanHandler{},
				newMockAllocator(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newCompactionTrigger(tt.args.meta, tt.args.compactionHandler, tt.args.allocator)
			assert.Equal(t, tt.args.meta, got.meta)
			assert.Equal(t, tt.args.compactionHandler, got.compactionHandler)
			assert.Equal(t, tt.args.allocator, got.allocator)
		})
	}
}
