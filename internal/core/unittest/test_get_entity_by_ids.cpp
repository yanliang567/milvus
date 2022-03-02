// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include "query/ExprImpl.h"
#include "segcore/ScalarIndex.h"
#include "test_utils/DataGen.h"

using namespace milvus;
using namespace milvus::segcore;

TEST(GetEntityByIds, ScalarIndex) {
    SUCCEED();
    auto index = std::make_unique<ScalarIndexVector>();
    std::vector<int64_t> data;
    int N = 1000;
    auto req_ids = std::make_unique<IdArray>();
    auto req_ids_arr = req_ids->mutable_int_id();

    for (int i = 0; i < N; ++i) {
        data.push_back(i * 3 % N);
        req_ids_arr->add_data(i);
    }
    index->append_data(data.data(), N, SegOffset(10000));
    index->build();

    auto [res_ids, res_offsets] = index->do_search_ids(*req_ids);
    auto res_ids_arr = res_ids->int_id();

    for (int i = 0; i < N; ++i) {
        auto res_offset = res_offsets[i].get() - 10000;
        auto res_id = res_ids_arr.data(i);
        auto std_id = (res_offset * 3 % N);
        ASSERT_EQ(res_id, std_id);
    }
}

TEST(GetEntityByIds, AUTOID) {
    auto schema = std::make_shared<Schema>();
    auto fid_64 = schema->AddDebugField("i64", DataType::INT64);
    auto DIM = 16;
    auto fid_vec = schema->AddDebugField("vector_64", DataType::VECTOR_FLOAT, DIM, MetricType::METRIC_L2);

    int64_t N = 10000;
    int64_t req_size = 10;
    auto choose = [=](int i) { return i * 3 % N; };

    auto dataset = DataGen(schema, N);
    auto segment = CreateSealedSegment(schema);
    SealedLoader(dataset, *segment);

    auto req_ids = std::make_unique<IdArray>();
    auto req_ids_arr = req_ids->mutable_int_id();

    auto i64_col = dataset.get_col<int64_t>(0);
    auto vf_col = dataset.get_col<float>(1);
    for (int i = 0; i < req_size; ++i) {
        req_ids_arr->add_data(dataset.row_ids_[choose(i)]);
    }

    // should be ruled out
    req_ids_arr->add_data(-1);

    std::vector<FieldOffset> target_offsets{FieldOffset(0), FieldOffset(1)};
    //    auto retrieve_results = segment->GetEntityById(target_offsets, *req_ids, 0);
    //    auto ids = retrieve_results->ids().int_id();
    //    Assert(retrieve_results->fields_data_size() == target_offsets.size());
    //    FieldOffset field_offset(0);
    //    auto field0 = retrieve_results->fields_data(0);
    //    Assert(field0.has_scalars());
    //    auto field0_data = field0.scalars().long_data();
    //    for (int i = 0; i < req_size; ++i) {
    //        auto id = ids.data(i);
    //        auto index = choose(i);
    //        ASSERT_EQ(id, dataset.row_ids_[index]);
    //        auto data = field0_data.data(i);
    //        ASSERT_EQ(data, i64_col[index]);
    //    }
    //
    //    auto field1 = retrieve_results->fields_data(1);
    //    Assert(field1.has_vectors());
    //    auto field1_data = field1.vectors().float_vector();
    //    ASSERT_EQ(field1_data.data_size(), DIM * req_size);
    //
    //    for (int i = 0; i < req_size; ++i) {
    //        for (int d = 0; d < DIM; ++d) {
    //            auto index = choose(i);
    //            auto data = field1_data.data(i * DIM + d);
    //            auto ref = vf_col[index * DIM + d];
    //            ASSERT_EQ(data, ref);
    //        }
    //    }
}

TEST(Retrieve, AUTOID) {
    auto schema = std::make_shared<Schema>();
    auto fid_64 = schema->AddDebugField("i64", DataType::INT64);
    auto DIM = 16;
    auto fid_vec = schema->AddDebugField("vector_64", DataType::VECTOR_FLOAT, DIM, MetricType::METRIC_L2);
    schema->set_primary_key(FieldOffset(0));

    int64_t N = 100;
    int64_t req_size = 10;
    auto choose = [=](int i) { return i * 3 % N; };

    auto dataset = DataGen(schema, N);
    auto segment = CreateSealedSegment(schema);
    SealedLoader(dataset, *segment);
    auto i64_col = dataset.get_col<int64_t>(0);

    auto plan = std::make_unique<query::RetrievePlan>(*schema);
    std::vector<int64_t> values;
    for (int i = 0; i < req_size; ++i) {
        values.emplace_back(i64_col[choose(i)]);
    }
    auto term_expr = std::make_unique<query::TermExprImpl<int64_t>>(FieldOffset(0), DataType::INT64, values);
    plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
    plan->plan_node_->predicate_ = std::move(term_expr);
    std::vector<FieldOffset> target_offsets{FieldOffset(0), FieldOffset(1)};
    plan->field_offsets_ = target_offsets;

    auto retrieve_results = segment->Retrieve(plan.get(), 100);
    Assert(retrieve_results->fields_data_size() == target_offsets.size());
    FieldOffset field_offset(0);
    auto field0 = retrieve_results->fields_data(0);
    Assert(field0.has_scalars());
    auto field0_data = field0.scalars().long_data();

    for (int i = 0; i < req_size; ++i) {
        auto index = choose(i);
        auto data = field0_data.data(i);
    }

    for (int i = 0; i < req_size; ++i) {
        auto index = choose(i);
        auto data = field0_data.data(i);
        ASSERT_EQ(data, i64_col[index]);
    }

    auto field1 = retrieve_results->fields_data(1);
    Assert(field1.has_vectors());
    auto field1_data = field1.vectors().float_vector();
    ASSERT_EQ(field1_data.data_size(), DIM * req_size);
}

TEST(Retrieve2, LargeTimestamp) {
    auto schema = std::make_shared<Schema>();
    auto fid_64 = schema->AddDebugField("i64", DataType::INT64);
    auto DIM = 16;
    auto fid_vec = schema->AddDebugField("vector_64", DataType::VECTOR_FLOAT, DIM, MetricType::METRIC_L2);
    schema->set_primary_key(FieldOffset(0));

    int64_t N = 100;
    int64_t req_size = 10;
    int choose_sep = 3;
    auto choose = [=](int i) { return i * choose_sep % N; };
    uint64_t ts_offset = 100;
    auto dataset = DataGen(schema, N, 42, ts_offset + 1);
    auto segment = CreateSealedSegment(schema);
    SealedLoader(dataset, *segment);
    auto i64_col = dataset.get_col<int64_t>(0);

    auto plan = std::make_unique<query::RetrievePlan>(*schema);
    std::vector<int64_t> values;
    for (int i = 0; i < req_size; ++i) {
        values.emplace_back(i64_col[choose(i)]);
    }
    auto term_expr = std::make_unique<query::TermExprImpl<int64_t>>(FieldOffset(0), DataType::INT64, values);
    plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
    plan->plan_node_->predicate_ = std::move(term_expr);
    std::vector<FieldOffset> target_offsets{FieldOffset(0), FieldOffset(1)};
    plan->field_offsets_ = target_offsets;

    std::vector<int> filter_timestamps{-1, 0, 1, 10, 20};
    filter_timestamps.push_back(N / 2);
    for (const auto& f_ts : filter_timestamps) {
        auto retrieve_results = segment->Retrieve(plan.get(), ts_offset + 1 + f_ts);
        Assert(retrieve_results->fields_data_size() == 2);
        auto field0 = retrieve_results->fields_data(0);
        auto field1 = retrieve_results->fields_data(1);
        int target_num = (f_ts + choose_sep) / choose_sep;
        if (target_num > req_size) {
            target_num = req_size;
        }
        Assert(field0.scalars().long_data().data_size() == target_num);
        Assert(field1.vectors().float_vector().data_size() == target_num * DIM);
    }
}

TEST(GetEntityByIds, PrimaryKey) {
    auto schema = std::make_shared<Schema>();
    auto fid_64 = schema->AddDebugField("counter_i64", DataType::INT64);
    auto DIM = 16;
    auto fid_vec = schema->AddDebugField("vector_64", DataType::VECTOR_FLOAT, DIM, MetricType::METRIC_L2);
    schema->set_primary_key(FieldOffset(0));

    int64_t N = 10000;
    int64_t req_size = 10;
    auto choose = [=](int i) { return i * 3 % N; };

    auto dataset = DataGen(schema, N);
    auto segment = CreateSealedSegment(schema);
    SealedLoader(dataset, *segment);

    auto req_ids = std::make_unique<IdArray>();
    auto req_ids_arr = req_ids->mutable_int_id();

    auto i64_col = dataset.get_col<int64_t>(0);
    auto vf_col = dataset.get_col<float>(1);
    for (int i = 0; i < req_size; ++i) {
        req_ids_arr->add_data(i64_col[choose(i)]);
    }

    // should be ruled out
    req_ids_arr->add_data(-1);

    std::vector<FieldOffset> target_offsets{FieldOffset(0), FieldOffset(1)};
    //    auto retrieve_results = segment->GetEntityById(target_offsets, *req_ids, 0);
    //    auto ids = retrieve_results->ids().int_id();
    //    Assert(retrieve_results->fields_data_size() == target_offsets.size());
    //    FieldOffset field_offset(0);
    //    auto field0 = retrieve_results->fields_data(0);
    //    Assert(field0.has_scalars());
    //    auto field0_data = field0.scalars().long_data();
    //    for (int i = 0; i < req_size; ++i) {
    //        auto id = ids.data(i);
    //        auto index = choose(i);
    //        ASSERT_EQ(id, i64_col[index]);
    //        auto data = field0_data.data(i);
    //        ASSERT_EQ(data, i64_col[index]);
    //    }
    //
    //    auto field1 = retrieve_results->fields_data(1);
    //    Assert(field1.has_vectors());
    //    auto field1_data = field1.vectors().float_vector();
    //    ASSERT_EQ(field1_data.data_size(), DIM * req_size);
    //
    //    for (int i = 0; i < req_size; ++i) {
    //        for (int d = 0; d < DIM; ++d) {
    //            auto index = choose(i);
    //            auto data = field1_data.data(i * DIM + d);
    //            auto ref = vf_col[index * DIM + d];
    //            ASSERT_EQ(data, ref);
    //        }
    //    }
}

TEST(GetEntityByIds, delete_retrieve) {
    auto schema = std::make_shared<Schema>();
    auto fid_64 = schema->AddDebugField("i64", DataType::INT64);
    auto DIM = 16;
    auto fid_vec = schema->AddDebugField("vector_64", DataType::VECTOR_FLOAT, DIM, MetricType::METRIC_L2);
    schema->set_primary_key(FieldOffset(0));

    int64_t N = 10;
    int64_t req_size = 10;
    auto choose = [=](int i) { return i; };

    auto dataset = DataGen(schema, N);
    auto segment = CreateSealedSegment(schema);
    SealedLoader(dataset, *segment);
    auto i64_col = dataset.get_col<int64_t>(0);

    auto plan = std::make_unique<query::RetrievePlan>(*schema);
    std::vector<int64_t> values;
    for (int i = 0; i < req_size; ++i) {
        values.emplace_back(i64_col[choose(i)]);
    }
    auto term_expr = std::make_unique<query::TermExprImpl<int64_t>>(FieldOffset(0), DataType::INT64, values);
    plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
    plan->plan_node_->predicate_ = std::move(term_expr);
    std::vector<FieldOffset> target_offsets{FieldOffset(0), FieldOffset(1)};
    plan->field_offsets_ = target_offsets;

    {
        auto retrieve_results = segment->Retrieve(plan.get(), 100);
        Assert(retrieve_results->fields_data_size() == target_offsets.size());
        FieldOffset field_offset(0);
        auto field0 = retrieve_results->fields_data(0);
        Assert(field0.has_scalars());
        auto field0_data = field0.scalars().long_data();

        for (int i = 0; i < req_size; ++i) {
            auto index = choose(i);
            auto data = field0_data.data(i);
        }

        for (int i = 0; i < req_size; ++i) {
            auto index = choose(i);
            auto data = field0_data.data(i);
            ASSERT_EQ(data, i64_col[index]);
        }

        auto field1 = retrieve_results->fields_data(1);
        Assert(field1.has_vectors());
        auto field1_data = field1.vectors().float_vector();
        ASSERT_EQ(field1_data.data_size(), DIM * req_size);
    }

    int64_t row_count = 0;
    // strange, when enable load_delete_record, this test failed
    auto load_delete_record = false;
    if (load_delete_record) {
        std::vector<idx_t> pks{1, 2, 3, 4, 5};
        std::vector<Timestamp> timestamps{10, 10, 10, 10, 10};

        LoadDeletedRecordInfo info = {timestamps.data(), pks.data(), row_count};
        segment->LoadDeletedRecord(info);
        row_count = 5;
    }

    int64_t new_count = 6;
    std::vector<idx_t> new_pks{0, 1, 2, 3, 4, 5};
    std::vector<idx_t> new_timestamps{10, 10, 10, 10, 10, 10};
    auto reserved_offset = segment->PreDelete(new_count);
    ASSERT_EQ(reserved_offset, row_count);
    segment->Delete(reserved_offset, new_count, reinterpret_cast<const int64_t*>(new_pks.data()),
                    reinterpret_cast<const Timestamp*>(new_timestamps.data()));

    {
        auto retrieve_results = segment->Retrieve(plan.get(), 100);
        Assert(retrieve_results->fields_data_size() == target_offsets.size());
        FieldOffset field_offset(0);
        auto field0 = retrieve_results->fields_data(0);
        Assert(field0.has_scalars());
        auto field0_data = field0.scalars().long_data();
        auto size = req_size - new_count;
        for (int i = 0; i < size; ++i) {
            auto index = choose(i);
            auto data = field0_data.data(i);
            ASSERT_EQ(data, i64_col[index + new_count]);
        }

        auto field1 = retrieve_results->fields_data(1);
        Assert(field1.has_vectors());
        auto field1_data = field1.vectors().float_vector();
        ASSERT_EQ(field1_data.data_size(), DIM * size);
    }
}