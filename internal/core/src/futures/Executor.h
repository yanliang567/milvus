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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/task_queue/PriorityLifoSemMPMCQueue.h>
#include <folly/system/HardwareConcurrency.h>

namespace milvus::futures {

namespace ExecutePriority {
const int LOW = 2;
const int NORMAL = 1;
const int HIGH = 0;
}  // namespace ExecutePriority

folly::CPUThreadPoolExecutor*
getGlobalCPUExecutor();

};  // namespace milvus::futures
