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
#include "common/Types.h"
#include "common/Schema.h"
#include "query/Plan.h"
#include "common/Span.h"
#include "FieldIndexing.h"
#include <knowhere/index/vector_index/VecIndex.h>
#include "common/SystemProperty.h"
#include "query/PlanNode.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include <memory>
#include <deque>
#include <vector>
#include <utility>
#include <string>

namespace milvus::segcore {

// common interface of SegmentSealed and SegmentGrowing used by C API
class SegmentInterface {
 public:
    virtual void
    FillPrimaryKeys(const query::Plan* plan, SearchResult& results) const = 0;

    virtual void
    FillTargetEntry(const query::Plan* plan, SearchResult& results) const = 0;

    virtual SearchResult
    Search(const query::Plan* Plan, const query::PlaceholderGroup& placeholder_group, Timestamp timestamp) const = 0;

    virtual std::unique_ptr<proto::segcore::RetrieveResults>
    Retrieve(const query::RetrievePlan* Plan, Timestamp timestamp) const = 0;

    virtual int64_t
    GetMemoryUsageInBytes() const = 0;

    virtual int64_t
    get_row_count() const = 0;

    virtual const Schema&
    get_schema() const = 0;

    virtual int64_t
    PreDelete(int64_t size) = 0;

    virtual Status
    Delete(int64_t reserved_offset, int64_t size, const int64_t* row_ids, const Timestamp* timestamps) = 0;

    virtual ~SegmentInterface() = default;
};

// internal API for DSL calculation
// only for implementation
class SegmentInternalInterface : public SegmentInterface {
 public:
    template <typename T>
    Span<T>
    chunk_data(FieldOffset field_offset, int64_t chunk_id) const {
        return static_cast<Span<T>>(chunk_data_impl(field_offset, chunk_id));
    }

    template <typename T>
    const knowhere::scalar::StructuredIndex<T>&
    chunk_scalar_index(FieldOffset field_offset, int64_t chunk_id) const {
        static_assert(IsScalar<T>);
        using IndexType = knowhere::scalar::StructuredIndex<T>;
        auto base_ptr = chunk_index_impl(field_offset, chunk_id);
        auto ptr = dynamic_cast<const IndexType*>(base_ptr);
        AssertInfo(ptr, "entry mismatch");
        return *ptr;
    }

    SearchResult
    Search(const query::Plan* Plan,
           const query::PlaceholderGroup& placeholder_group,
           Timestamp timestamp) const override;

    void
    FillPrimaryKeys(const query::Plan* plan, SearchResult& results) const override;

    void
    FillTargetEntry(const query::Plan* plan, SearchResult& results) const override;

    std::unique_ptr<proto::segcore::RetrieveResults>
    Retrieve(const query::RetrievePlan* plan, Timestamp timestamp) const override;

    virtual std::string
    debug() const = 0;

 public:
    virtual void
    vector_search(int64_t vec_count,
                  query::SearchInfo search_info,
                  const void* query_data,
                  int64_t query_count,
                  Timestamp timestamp,
                  const BitsetView& bitset,
                  SearchResult& output) const = 0;

    virtual BitsetView
    get_filtered_bitmap(const BitsetView& bitset, int64_t ins_barrier, Timestamp timestamp) const = 0;

    // count of chunk that has index available
    virtual int64_t
    num_chunk_index(FieldOffset field_offset) const = 0;

    virtual void
    mask_with_timestamps(boost::dynamic_bitset<>& bitset_chunk, Timestamp timestamp) const = 0;

    // count of chunks
    virtual int64_t
    num_chunk() const = 0;

    // element size in each chunk
    virtual int64_t
    size_per_chunk() const = 0;

    virtual int64_t
    get_active_count(Timestamp ts) const = 0;

    virtual std::vector<SegOffset>
    search_ids(const boost::dynamic_bitset<>& view, Timestamp timestamp) const = 0;

    virtual std::vector<SegOffset>
    search_ids(const BitsetView& view, Timestamp timestamp) const = 0;

 protected:
    // internal API: return chunk_data in span
    virtual SpanBase
    chunk_data_impl(FieldOffset field_offset, int64_t chunk_id) const = 0;

    // internal API: return chunk_index in span, support scalar index only
    virtual const knowhere::Index*
    chunk_index_impl(FieldOffset field_offset, int64_t chunk_id) const = 0;

    // TODO remove system fields
    // calculate output[i] = Vec[seg_offsets[i]}, where Vec binds to system_type
    virtual void
    bulk_subscript(SystemFieldType system_type, const int64_t* seg_offsets, int64_t count, void* output) const = 0;

    // calculate output[i] = Vec[seg_offsets[i]}, where Vec binds to field_offset
    virtual void
    bulk_subscript(FieldOffset field_offset, const int64_t* seg_offsets, int64_t count, void* output) const = 0;

    // TODO: special hack: FieldOffset == -1 -> RowId.
    // TODO: remove this hack when transfer is done
    virtual std::unique_ptr<DataArray>
    BulkSubScript(FieldOffset field_offset, const SegOffset* seg_offsets, int64_t count) const;

    virtual std::pair<std::unique_ptr<IdArray>, std::vector<SegOffset>>
    search_ids(const IdArray& id_array, Timestamp timestamp) const = 0;

    virtual void
    check_search(const query::Plan* plan) const = 0;

 protected:
    mutable std::shared_mutex mutex_;
};

}  // namespace milvus::segcore
