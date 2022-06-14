// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package paramtable

import (
	"math"
	"strconv"
	"sync"
	"time"

	"github.com/go-basic/ipv4"
	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"
)

const (
	// DefaultServerMaxSendSize defines the maximum size of data per grpc request can send by server side.
	DefaultServerMaxSendSize = math.MaxInt32

	// DefaultServerMaxRecvSize defines the maximum size of data per grpc request can receive by server side.
	DefaultServerMaxRecvSize = math.MaxInt32

	// DefaultClientMaxSendSize defines the maximum size of data per grpc request can send by client side.
	DefaultClientMaxSendSize = 100 * 1024 * 1024

	// DefaultClientMaxRecvSize defines the maximum size of data per grpc request can receive by client side.
	DefaultClientMaxRecvSize = 100 * 1024 * 1024

	// DefaultLogLevel defines the log level of grpc
	DefaultLogLevel = "WARNING"

	// Grpc Timeout related configs
	DefaultDialTimeout      = 5000 * time.Millisecond
	DefaultKeepAliveTime    = 10000 * time.Millisecond
	DefaultKeepAliveTimeout = 20000 * time.Millisecond

	ProxyInternalPort = 19529
	ProxyExternalPort = 19530
)

///////////////////////////////////////////////////////////////////////////////
// --- grpc ---
type grpcConfig struct {
	ServiceParam

	once          sync.Once
	Domain        string
	IP            string
	TLSMode       int
	Port          int
	InternalPort  int
	ServerPemPath string
	ServerKeyPath string
	CaPemPath     string
}

func (p *grpcConfig) init(domain string) {
	p.ServiceParam.Init()
	p.Domain = domain

	p.LoadFromEnv()
	p.LoadFromArgs()
	p.initPort()
	p.initTLSPath()
}

// LoadFromEnv is used to initialize configuration items from env.
func (p *grpcConfig) LoadFromEnv() {
	p.IP = ipv4.LocalIP()
}

// LoadFromArgs is used to initialize configuration items from args.
func (p *grpcConfig) LoadFromArgs() {

}

func (p *grpcConfig) initPort() {
	p.Port = p.ParseIntWithDefault(p.Domain+".port", ProxyExternalPort)
	p.InternalPort = p.ParseIntWithDefault(p.Domain+".internalPort", ProxyInternalPort)
}

func (p *grpcConfig) initTLSPath() {
	p.TLSMode = p.ParseIntWithDefault("common.security.tlsMode", 0)
	p.ServerPemPath = p.Get("tls.serverPemPath")
	p.ServerKeyPath = p.Get("tls.serverKeyPath")
	p.CaPemPath = p.Get("tls.caPemPath")
}

// GetAddress return grpc address
func (p *grpcConfig) GetAddress() string {
	return p.IP + ":" + strconv.Itoa(p.Port)
}

func (p *grpcConfig) GetInternalAddress() string {
	return p.IP + ":" + strconv.Itoa(p.InternalPort)
}

// GrpcServerConfig is configuration for grpc server.
type GrpcServerConfig struct {
	grpcConfig

	ServerMaxSendSize int
	ServerMaxRecvSize int
}

// InitOnce initialize grpc server config once
func (p *GrpcServerConfig) InitOnce(domain string) {
	p.once.Do(func() {
		p.init(domain)
	})
}

func (p *GrpcServerConfig) init(domain string) {
	p.grpcConfig.init(domain)

	p.initServerMaxSendSize()
	p.initServerMaxRecvSize()
}

func (p *GrpcServerConfig) initServerMaxSendSize() {
	var err error

	valueStr, err := p.Load("grpc.serverMaxSendSize")
	if err != nil {
		valueStr, err = p.Load(p.Domain + ".grpc.serverMaxSendSize")
	}
	if err != nil {
		p.ServerMaxSendSize = DefaultServerMaxSendSize
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.serverMaxSendSize, set to default",
				zap.String("role", p.Domain), zap.String("grpc.serverMaxSendSize", valueStr),
				zap.Error(err))
			p.ServerMaxSendSize = DefaultServerMaxSendSize
		} else {
			p.ServerMaxSendSize = value
		}
	}

	log.Debug("initServerMaxSendSize",
		zap.String("role", p.Domain), zap.Int("grpc.serverMaxSendSize", p.ServerMaxSendSize))
}

func (p *GrpcServerConfig) initServerMaxRecvSize() {
	var err error
	valueStr, err := p.Load("grpc.serverMaxRecvSize")
	if err != nil {
		valueStr, err = p.Load(p.Domain + ".grpc.serverMaxRecvSize")
	}
	if err != nil {
		p.ServerMaxRecvSize = DefaultServerMaxRecvSize
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.serverMaxRecvSize, set to default",
				zap.String("role", p.Domain), zap.String("grpc.serverMaxRecvSize", valueStr),
				zap.Error(err))
			p.ServerMaxRecvSize = DefaultServerMaxRecvSize
		} else {
			p.ServerMaxRecvSize = value
		}
	}

	log.Debug("initServerMaxRecvSize",
		zap.String("role", p.Domain), zap.Int("grpc.serverMaxRecvSize", p.ServerMaxRecvSize))
}

// GrpcClientConfig is configuration for grpc client.
type GrpcClientConfig struct {
	grpcConfig

	ClientMaxSendSize int
	ClientMaxRecvSize int

	DialTimeout      time.Duration
	KeepAliveTime    time.Duration
	KeepAliveTimeout time.Duration
}

// InitOnce initialize grpc client config once
func (p *GrpcClientConfig) InitOnce(domain string) {
	p.once.Do(func() {
		p.init(domain)
	})
}

func (p *GrpcClientConfig) init(domain string) {
	p.grpcConfig.init(domain)

	p.initClientMaxSendSize()
	p.initClientMaxRecvSize()
	p.initDialTimeout()
	p.initKeepAliveTimeout()
	p.initKeepAliveTime()
}

func (p *GrpcClientConfig) initClientMaxSendSize() {
	var err error

	valueStr, err := p.Load("grpc.clientMaxSendSize")
	if err != nil {
		valueStr, err = p.Load(p.Domain + ".grpc.clientMaxSendSize")
	}
	if err != nil {
		p.ClientMaxSendSize = DefaultClientMaxSendSize
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.clientMaxSendSize, set to default",
				zap.String("role", p.Domain), zap.String("grpc.clientMaxSendSize", valueStr),
				zap.Error(err))

			p.ClientMaxSendSize = DefaultClientMaxSendSize
		} else {
			p.ClientMaxSendSize = value
		}
	}

	log.Debug("initClientMaxSendSize",
		zap.String("role", p.Domain), zap.Int("grpc.clientMaxSendSize", p.ClientMaxSendSize))
}

func (p *GrpcClientConfig) initClientMaxRecvSize() {
	var err error
	valueStr, err := p.Load("grpc.clientMaxRecvSize")
	if err != nil {
		valueStr, err = p.Load(p.Domain + ".grpc.clientMaxRecvSize")
	}
	if err != nil {
		p.ClientMaxRecvSize = DefaultClientMaxRecvSize
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.clientMaxRecvSize, set to default",
				zap.String("role", p.Domain), zap.String("grpc.clientMaxRecvSize", valueStr),
				zap.Error(err))

			p.ClientMaxRecvSize = DefaultClientMaxRecvSize
		} else {
			p.ClientMaxRecvSize = value
		}
	}

	log.Debug("initClientMaxRecvSize",
		zap.String("role", p.Domain), zap.Int("grpc.clientMaxRecvSize", p.ClientMaxRecvSize))
}

func (p *GrpcClientConfig) initDialTimeout() {
	var err error
	valueStr, err := p.Load("grpc.client.dialTimeout")
	if err != nil {
		p.DialTimeout = DefaultDialTimeout
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.client.dialTimeout, set to default",
				zap.String("role", p.Domain), zap.String("grpc.client.dialTimeout", valueStr),
				zap.Error(err))
			p.DialTimeout = DefaultDialTimeout
		} else {
			p.DialTimeout = time.Duration(value) * time.Millisecond
		}
	}
	log.Debug("Init dial timeout",
		zap.String("role", p.Domain), zap.Duration("grpc.log.dialTimeout", p.DialTimeout))
}

func (p *GrpcClientConfig) initKeepAliveTime() {
	var err error
	valueStr, err := p.Load("grpc.client.keepAliveTime")
	if err != nil {
		p.KeepAliveTime = DefaultKeepAliveTime
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.client.keepAliveTime, set to default",
				zap.String("role", p.Domain), zap.String("grpc.client.keepAliveTime", valueStr),
				zap.Error(err))

			p.KeepAliveTime = DefaultKeepAliveTime
		} else {
			p.KeepAliveTime = time.Duration(value) * time.Millisecond
		}
	}
	log.Debug("Init keep alive time",
		zap.String("role", p.Domain), zap.Duration("grpc.log.keepAliveTime", p.KeepAliveTime))
}

func (p *GrpcClientConfig) initKeepAliveTimeout() {
	var err error
	valueStr, err := p.Load("grpc.client.keepAliveTimeout")
	if err != nil {
		p.KeepAliveTimeout = DefaultKeepAliveTimeout
	} else {
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			log.Warn("Failed to parse grpc.client.keepAliveTimeout, set to default",
				zap.String("role", p.Domain), zap.String("grpc.client.keepAliveTimeout", valueStr),
				zap.Error(err))
			p.KeepAliveTimeout = DefaultKeepAliveTimeout
		} else {
			p.KeepAliveTimeout = time.Duration(value) * time.Millisecond
		}
	}
	log.Debug("Init keep alive timeout",
		zap.String("role", p.Domain), zap.Duration("grpc.log.keepAliveTimeout", p.KeepAliveTimeout))
}
