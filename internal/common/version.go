package common

import "github.com/blang/semver/v4"

// Version current versiong for session
var Version semver.Version

func init() {
	Version, _ = semver.Parse("2.2.0-pre+dev")
}
