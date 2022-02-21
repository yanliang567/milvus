# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CXX          = /usr/bin/ccache /usr/bin/c++ -std=c++11
CXXCPP       = /usr/bin/ccache /usr/bin/c++ -std=c++11 -E
CPPFLAGS     = -DFINTEGER=int -DUSE_CPU  
CXXFLAGS     = -fPIC -m64 -Wno-sign-compare -DELPP_THREAD_SAFE -fopenmp -Werror=return-type -O3 -DNDEBUG -fPIC -mf16c -O3
CPUFLAGS     = -mpopcnt -msse4
LDFLAGS      =  -L/usr/lib 
LIBS         = -lopenblas   
PYTHONCFLAGS =   -I
SWIGFLAGS    = -DSWIGWORDSIZE64

NVCC         = 
CUDA_ROOT    = 
CUDA_ARCH    = default
NVCCFLAGS    = -I $(CUDA_ROOT)/targets/x86_64-linux/include/ \
-O3 \
-Xcompiler -fPIC \
-Xcudafe --diag_suppress=unrecognized_attribute \
$(CUDA_ARCH) \
-lineinfo \
-ccbin $(CXX)

OS = $(shell uname -s)

SHAREDEXT   = so
SHAREDFLAGS = -shared

ifeq ($(OS),Darwin)
	SHAREDEXT   = dylib
	SHAREDFLAGS = -dynamiclib -undefined dynamic_lookup
        SWIGFLAGS   =
endif

MKDIR_P      = /bin/mkdir -p
PYTHON       = 
SWIG         = 
AR          ?= ar

prefix      ?= /go/src/github.com/milvus-io/milvus/cmake_build/src/index/faiss_ep-prefix/src/faiss_ep
exec_prefix ?= ${prefix}
libdir       = ${exec_prefix}/lib
includedir   = ${prefix}/include
