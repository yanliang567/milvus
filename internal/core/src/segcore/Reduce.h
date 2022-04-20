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

#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

#include "utils/Status.h"
#include "common/type_c.h"

namespace milvus::segcore {

// SearchResultDataBlobs contains the marshal blobs of many `milvus::proto::schema::SearchResultData`
struct SearchResultDataBlobs {
    std::vector<std::vector<char>> blobs;
};

Status
merge_into(int64_t num_queries,
           int64_t topk,
           float* distances,
           int64_t* uids,
           const float* new_distances,
           const int64_t* new_uids);
}  // namespace milvus::segcore
