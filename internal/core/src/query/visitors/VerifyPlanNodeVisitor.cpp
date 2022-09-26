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

#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/ConfAdapter.h"
#include "query/generated/VerifyPlanNodeVisitor.h"

namespace milvus::query {

namespace impl {
// THIS CONTAINS EXTRA BODY FOR VISITOR
// WILL BE USED BY GENERATOR UNDER suvlim/core_gen/
class VerifyPlanNodeVisitor : PlanNodeVisitor {
 public:
    using RetType = SearchResult;
    VerifyPlanNodeVisitor() = default;

 private:
    std::optional<RetType> ret_;
};
}  // namespace impl

static IndexType
InferIndexType(const Json& search_params) {
    // ivf -> nprobe
    // hnsw -> ef
    // annoy -> search_k
    // ngtpanng / ngtonng -> max_search_edges / epsilon
    static const std::map<std::string, IndexType> key_list = [] {
        std::map<std::string, IndexType> list;
        namespace ip = knowhere::indexparam;
        namespace ie = knowhere::IndexEnum;
        list.emplace(ip::NPROBE, ie::INDEX_FAISS_IVFFLAT);
        list.emplace(ip::EF, ie::INDEX_HNSW);
        list.emplace(ip::SEARCH_K, ie::INDEX_ANNOY);
        return list;
    }();
    auto dbg_str = search_params.dump();
    for (auto& kv : search_params.items()) {
        std::string key = kv.key();
        if (key_list.count(key)) {
            return key_list.at(key);
        }
    }
    PanicCodeInfo(ErrorCodeEnum::IllegalArgument, "failed to infer index type");
}

static IndexType
InferBinaryIndexType(const Json& search_params) {
    namespace ip = knowhere::indexparam;
    namespace ie = knowhere::IndexEnum;
    if (search_params.contains(ip::NPROBE)) {
        return ie::INDEX_FAISS_BIN_IVFFLAT;
    } else {
        return ie::INDEX_FAISS_BIN_IDMAP;
    }
}

void
VerifyPlanNodeVisitor::visit(FloatVectorANNS& node) {
    auto& search_params = node.search_info_.search_params_;
    auto inferred_type = InferIndexType(search_params);
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(inferred_type);
    auto index_mode = knowhere::IndexMode::MODE_CPU;

    // mock the api, topk will be passed from placeholder
    auto params_copy = search_params;
    knowhere::SetMetaTopk(params_copy, 10);

    // NOTE: the second parameter is not checked in knowhere, may be redundant
    auto passed = adapter->CheckSearch(params_copy, inferred_type, index_mode);
    if (!passed) {
        PanicCodeInfo(ErrorCodeEnum::IllegalArgument, "invalid search params");
    }
}

void
VerifyPlanNodeVisitor::visit(BinaryVectorANNS& node) {
    auto& search_params = node.search_info_.search_params_;
    auto inferred_type = InferBinaryIndexType(search_params);
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(inferred_type);
    auto index_mode = knowhere::IndexMode::MODE_CPU;

    // mock the api, topk will be passed from placeholder
    auto params_copy = search_params;
    knowhere::SetMetaTopk(params_copy, 10);

    // NOTE: the second parameter is not checked in knowhere, may be redundant
    auto passed = adapter->CheckSearch(params_copy, inferred_type, index_mode);
    if (!passed) {
        PanicCodeInfo(ErrorCodeEnum::IllegalArgument, "invalid search params");
    }
}

void
VerifyPlanNodeVisitor::visit(RetrievePlanNode& node) {
}

}  // namespace milvus::query
