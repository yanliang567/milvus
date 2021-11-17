#!/bin/bash

# Copyright 2018 Istio Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -u
set -x
function milvus_ci_release_name(){
    # rules for helm release name 
    local name="m"
    if [[ "${MILVUS_SERVER_TYPE:-}" == "distributed" ]]; then
        # distributed mode
       name+="d"
    else 
       #standalone mode      
        name+="s"

    fi 
    #[debug] add pr number into release name 
    if [[ -n ${CHANGE_ID:-} ]]; then 
        name+="-${CHANGE_ID:-}"
    fi 

    if [[ -n ${GIT_COMMIT:-} ]]; then 
            name+="-${GIT_COMMIT:0:4}"
    fi 

    # [remove-kind] Add Jenkins BUILD_ID into Name
    if [[ -n ${JENKINS_BUILD_ID:-} ]]; then 
            name+="-${JENKINS_BUILD_ID}"
    fi 

    export MILVUS_HELM_RELEASE_NAME=${name}
    echo ${name}
}
milvus_ci_release_name