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

#include <memory>
#include <knowhere/index/Index.h>
#include <knowhere/common/Dataset.h>
#include <boost/dynamic_bitset.hpp>

namespace milvus::scalar {
using Index = knowhere::Index;
using IndexPtr = std::unique_ptr<Index>;
using BinarySet = knowhere::BinarySet;
using Config = knowhere::Config;
using DatasetPtr = knowhere::DatasetPtr;
using TargetBitmap = boost::dynamic_bitset<>;
using TargetBitmapPtr = std::unique_ptr<TargetBitmap>;

class IndexBase : public Index {
 public:
    virtual void
    BuildWithDataset(const DatasetPtr& dataset) = 0;

    virtual const TargetBitmapPtr
    Query(const DatasetPtr& dataset) = 0;

    virtual size_t
    Count() = 0;
};
using IndexBasePtr = std::unique_ptr<IndexBase>;

}  // namespace milvus::scalar
