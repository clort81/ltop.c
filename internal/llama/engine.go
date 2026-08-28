package llama

import (
	"strings"

	"github.com/pefman/ltop/internal/promparse"
)

// Engine identifies which inference server produced the telemetry.
type Engine string

const (
	EngineUnknown Engine = ""
	EngineLlama   Engine = "llama.cpp"
	EngineSGLang  Engine = "sglang"
)

// DetectEngine identifies the server from a /metrics exposition document.
func DetectEngine(set promparse.Set) Engine {
	switch {
	case set.HasPrefix("sglang:"):
		return EngineSGLang
	case set.HasPrefix("llamacpp:"):
		return EngineLlama
	default:
		return EngineUnknown
	}
}

// ParseMetricsFor projects a metrics scrape into the typed snapshot used by
// the dashboard. Unknown engines fall back to the llama.cpp layout so a
// partial scrape still yields something usable.
func ParseMetricsFor(engine Engine, set promparse.Set) Metrics {
	switch engine {
	case EngineSGLang:
		return ParseSGLangMetrics(set)
	default:
		return ParseMetrics(set)
	}
}

// DisplayName is the short label shown in the header and setup list.
func (e Engine) DisplayName() string {
	switch e {
	case EngineSGLang:
		return "sglang"
	case EngineLlama:
		return "llama.cpp"
	case EngineUnknown:
		return "server"
	default:
		return string(e)
	}
}

// Is reports whether e equals want, comparing case-insensitively for safety
// when Engine values come from older configs or free-form input.
func (e Engine) Is(want Engine) bool {
	return strings.EqualFold(string(e), string(want))
}
