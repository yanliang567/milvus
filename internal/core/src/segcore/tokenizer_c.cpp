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

#include "segcore/tokenizer_c.h"
#include <memory>
#include "common/FieldMeta.h"
#include "common/protobuf_utils.h"
#include "monitor/scope_metric.h"
#include "pb/schema.pb.h"
#include "common/EasyAssert.h"
#include "tokenizer.h"

using Map = std::map<std::string, std::string>;

CStatus
create_tokenizer(const char* params, CTokenizer* tokenizer) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto impl = std::make_unique<milvus::tantivy::Tokenizer>(params);
        *tokenizer = impl.release();
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
clone_tokenizer(CTokenizer* tokenizer, CTokenizer* rst) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto impl = reinterpret_cast<milvus::tantivy::Tokenizer*>(*tokenizer);
        *rst = impl->Clone().release();
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

void
free_tokenizer(CTokenizer tokenizer) {
    SCOPE_CGO_CALL_METRIC();

    auto impl = reinterpret_cast<milvus::tantivy::Tokenizer*>(tokenizer);
    delete impl;
}

CTokenStream
create_token_stream(CTokenizer tokenizer, const char* text, uint32_t text_len) {
    SCOPE_CGO_CALL_METRIC();

    auto impl = reinterpret_cast<milvus::tantivy::Tokenizer*>(tokenizer);
    return impl->CreateTokenStream(std::string(text, text_len)).release();
}

CStatus
validate_tokenizer(const char* params) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto impl = std::make_unique<milvus::tantivy::Tokenizer>(params);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
validate_text_schema(const uint8_t* field_schema, uint64_t length) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto schema = std::make_unique<milvus::proto::schema::FieldSchema>();
        AssertInfo(schema->ParseFromArray(field_schema, length),
                   "failed to create field schema");

        auto type_params = milvus::RepeatedKeyValToMap(schema->type_params());
        milvus::tantivy::Tokenizer _(milvus::ParseTokenizerParams(type_params));

        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}
