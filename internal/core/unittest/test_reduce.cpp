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
#include <queue>
#include <random>
#include <vector>

#include "query/SubSearchResult.h"

using namespace milvus;
using namespace milvus::query;

TEST(Reduce, SubQueryResult) {
    int64_t num_queries = 512;
    int64_t topk = 32;
    int64_t iteration = 50;
    int64_t round_decimal = 3;
    constexpr int64_t limit = 100000000L;
    auto metric_type = knowhere::metric::L2;
    using queue_type = std::priority_queue<int64_t>;

    std::vector<queue_type> ref_results(num_queries);
    for (auto& ref_result : ref_results) {
        for (int i = 0; i < topk; ++i) {
            ref_result.push(limit);
        }
    }
    std::default_random_engine e(42);
    SubSearchResult final_result(num_queries, topk, metric_type, round_decimal);
    for (int i = 0; i < iteration; ++i) {
        std::vector<int64_t> ids;
        std::vector<float> distances;
        for (int n = 0; n < num_queries; ++n) {
            for (int k = 0; k < topk; ++k) {
                auto gen_x = e() % limit;
                ref_results[n].push(gen_x);
                ref_results[n].pop();
                ids.push_back(gen_x);
                distances.push_back(gen_x);
            }
            std::sort(ids.begin() + n * topk, ids.begin() + n * topk + topk);
            std::sort(distances.begin() + n * topk, distances.begin() + n * topk + topk);
        }
        SubSearchResult sub_result(num_queries, topk, metric_type, round_decimal);
        sub_result.mutable_distances() = distances;
        sub_result.mutable_seg_offsets() = ids;
        final_result.merge(sub_result);
    }

    for (int n = 0; n < num_queries; ++n) {
        ASSERT_EQ(ref_results[n].size(), topk);
        for (int k = 0; k < topk; ++k) {
            auto ref_x = ref_results[n].top();
            ref_results[n].pop();
            auto index = n * topk + topk - 1 - k;
            auto id = final_result.get_seg_offsets()[index];
            auto distance = final_result.get_distances()[index];
            ASSERT_EQ(id, ref_x);
            ASSERT_EQ(distance, ref_x);
        }
    }
}

TEST(Reduce, SubSearchResultDesc) {
    int64_t num_queries = 512;
    int64_t topk = 32;
    int64_t iteration = 50;
    int64_t round_decimal = 3;
    constexpr int64_t limit = 100000000L;
    constexpr int64_t init_value = 0;
    auto metric_type = knowhere::metric::IP;
    using queue_type = std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>>;

    std::vector<queue_type> ref_results(num_queries);
    for (auto& ref_result : ref_results) {
        for (int i = 0; i < topk; ++i) {
            ref_result.push(init_value);
        }
    }
    std::default_random_engine e(42);
    SubSearchResult final_result(num_queries, topk, metric_type, round_decimal);
    for (int i = 0; i < iteration; ++i) {
        std::vector<int64_t> ids;
        std::vector<float> distances;
        for (int n = 0; n < num_queries; ++n) {
            for (int k = 0; k < topk; ++k) {
                auto gen_x = e() % limit;
                ref_results[n].push(gen_x);
                ref_results[n].pop();
                ids.push_back(gen_x);
                distances.push_back(gen_x);
            }
            std::sort(ids.begin() + n * topk, ids.begin() + n * topk + topk, std::greater<int64_t>());
            std::sort(distances.begin() + n * topk, distances.begin() + n * topk + topk, std::greater<float>());
        }
        SubSearchResult sub_result(num_queries, topk, metric_type, round_decimal);
        sub_result.mutable_distances() = distances;
        sub_result.mutable_seg_offsets() = ids;
        final_result.merge(sub_result);
    }

    for (int n = 0; n < num_queries; ++n) {
        ASSERT_EQ(ref_results[n].size(), topk);
        for (int k = 0; k < topk; ++k) {
            auto ref_x = ref_results[n].top();
            ref_results[n].pop();
            auto index = n * topk + topk - 1 - k;
            auto id = final_result.get_seg_offsets()[index];
            auto distance = final_result.get_distances()[index];
            ASSERT_EQ(id, ref_x);
            ASSERT_EQ(distance, ref_x);
        }
    }
}