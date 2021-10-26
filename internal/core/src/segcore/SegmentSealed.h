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

#include "segcore/SegmentInterface.h"
#include "pb/segcore.pb.h"
#include "common/LoadInfo.h"
#include <utility>

namespace milvus::segcore {

class SegmentSealed : public SegmentInternalInterface {
 public:
    virtual void
    LoadIndex(const LoadIndexInfo& info) = 0;
    virtual void
    LoadSegmentMeta(const milvus::proto::segcore::LoadSegmentMeta& meta) = 0;
    virtual void
    LoadFieldData(const LoadFieldDataInfo& info) = 0;
    virtual void
    LoadDeletedRecord(const LoadDeletedRecordInfo& info) = 0;
    virtual void
    DropIndex(const FieldId field_id) = 0;
    virtual void
    DropFieldData(const FieldId field_id) = 0;
    virtual bool
    HasIndex(FieldId field_id) const = 0;
    virtual bool
    HasFieldData(FieldId field_id) const = 0;
};

using SegmentSealedPtr = std::unique_ptr<SegmentSealed>;

SegmentSealedPtr
CreateSealedSegment(SchemaPtr schema);

}  // namespace milvus::segcore
