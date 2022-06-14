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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "knowhere/common/Exception.h"
#include "index/IndexStructure.h"
#include "index/ScalarIndex.h"

namespace milvus::scalar {

template <typename T>
class ScalarIndexSort : public ScalarIndex<T> {
 public:
    ScalarIndexSort();
    ScalarIndexSort(size_t n, const T* values);

    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& index_binary) override;

    void
    BuildWithDataset(const DatasetPtr& dataset) override;

    size_t
    Count() override {
        return data_.size();
    }

    void
    Build(size_t n, const T* values) override;

    void
    build();

    const TargetBitmapPtr
    In(size_t n, const T* values) override;

    const TargetBitmapPtr
    NotIn(size_t n, const T* values) override;

    const TargetBitmapPtr
    Range(T value, OpType op) override;

    const TargetBitmapPtr
    Range(T lower_bound_value, bool lb_inclusive, T upper_bound_value, bool ub_inclusive) override;

    T
    Reverse_Lookup(size_t offset) const override;

 public:
    const std::vector<IndexStructure<T>>&
    GetData() {
        return data_;
    }

    int64_t
    Size() override {
        return (int64_t)data_.size();
    }

    bool
    IsBuilt() const {
        return is_built_;
    }

 private:
    bool is_built_;
    std::vector<size_t> idx_to_offsets_;  // used to retrieve.
    std::vector<IndexStructure<T>> data_;
};

template <typename T>
using ScalarIndexSortPtr = std::unique_ptr<ScalarIndexSort<T>>;

}  // namespace milvus::scalar

#include "index/ScalarIndexSort-inl.h"

namespace milvus::scalar {
template <typename T>
inline ScalarIndexSortPtr<T>
CreateScalarIndexSort() {
    return std::make_unique<ScalarIndexSort<T>>();
}
}  // namespace milvus::scalar
