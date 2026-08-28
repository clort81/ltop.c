package llama

import (
	"encoding/json"
	"path"
	"strings"

	"github.com/pefman/ltop/internal/promparse"
)

// SGLang metric names. All series are labeled; totals are summed across labels.
const (
	sgPromptTokens       = "sglang:prompt_tokens_total"
	sgGenerationTokens   = "sglang:generation_tokens_total"
	sgCachedTokens       = "sglang:cached_tokens_total"
	sgRealtimeTokens     = "sglang:realtime_tokens_total"
	sgSpecVerifyCalls    = "sglang:spec_verify_calls_total"
	sgSpecAcceptRate     = "sglang:spec_accept_rate"
	sgSpecAcceptLength   = "sglang:spec_accept_length"
	sgGenThroughput      = "sglang:gen_throughput"
	sgCacheHitRate       = "sglang:cache_hit_rate"
	sgTokenUsage         = "sglang:token_usage"
	sgNumUsedTokens      = "sglang:num_used_tokens"
	sgMaxTotalTokens     = "sglang:max_total_num_tokens"
	sgContextLen         = "sglang:context_len"
	sgNumRunningReqs     = "sglang:num_running_reqs"
	sgNumQueueReqs       = "sglang:num_queue_reqs"
	sgDecodeSumSeqLens   = "sglang:decode_sum_seq_lens"
)

// ServerInfo is the subset of SGLang's /get_server_info used by ltop.
type ServerInfo struct {
	ModelPath                   string  `json:"model_path"`
	ServedModelName             string  `json:"served_model_name"`
	Version                     string  `json:"version"`
	ContextLength               int     `json:"context_length"`
	MaxTotalNumTokens           int     `json:"max_total_num_tokens"`
	MaxRunningRequests          int     `json:"max_running_requests"`
	SpeculativeAlgorithm        string  `json:"speculative_algorithm"`
	SpeculativeNumSteps         int     `json:"speculative_num_steps"`
	SpeculativeNumDraftTokens   int     `json:"speculative_num_draft_tokens"`
	EnableMetrics               bool    `json:"enable_metrics"`
	AttentionBackend            string  `json:"attention_backend"`
	TPSize                      int     `json:"tp_size"`
	Dtype                       string  `json:"dtype"`
	Quantization                *string `json:"quantization"`
}

// PropsFromServerInfo maps SGLang server info onto the Props shape the rest of
// ltop already renders.
func PropsFromServerInfo(info ServerInfo) Props {
	model := info.ModelPath
	if model == "" {
		model = info.ServedModelName
	}
	build := "sglang"
	if info.Version != "" {
		build = "sglang " + info.Version
	}
	return Props{
		Engine:          EngineSGLang,
		ModelPath:       model,
		BuildInfo:       build,
		EndpointMetrics: info.EnableMetrics,
		EndpointSlots:   false,
		TotalSlots:      info.MaxRunningRequests,
	}
}

// ModelFromServerInfo synthesises the /v1/models metadata fields the header
// and -once output expect when SGLang does not publish GGUF-style meta.
func ModelFromServerInfo(info ServerInfo) Model {
	id := info.ServedModelName
	if id == "" {
		id = info.ModelPath
	}
	ctx := info.ContextLength
	m := Model{ID: id}
	m.Meta.NCtx = ctx
	m.Meta.NCtxTrain = ctx
	if info.Quantization != nil && *info.Quantization != "" {
		m.Meta.FType = *info.Quantization
	} else if info.Dtype != "" && !strings.EqualFold(info.Dtype, "auto") {
		m.Meta.FType = info.Dtype
	}
	return m
}

// DecodeServerInfo parses a /get_server_info (or /server_info) JSON body.
func DecodeServerInfo(b []byte) (ServerInfo, error) {
	var info ServerInfo
	if err := json.Unmarshal(b, &info); err != nil {
		return ServerInfo{}, err
	}
	return info, nil
}

// ParseSGLangMetrics projects an SGLang /metrics scrape into Metrics.
//
// SGLang does not export busy-second counters like llama.cpp. Live rates are
// therefore derived from wall-clock deltas of realtime_tokens_total (and the
// gen_throughput gauge) in the collector. Speculative quality comes from the
// accept-rate / accept-length gauges scaled by verify-call counts so the
// existing SpecAcceptanceRate / SpecTokensPerDraft / SpecSpeedup helpers work.
func ParseSGLangMetrics(set promparse.Set) Metrics {
	prompt := set.Sum(sgPromptTokens)
	cached := set.Sum(sgCachedTokens)
	generated := set.Sum(sgGenerationTokens)
	verifyCalls := set.Sum(sgSpecVerifyCalls)
	acceptRate := set.Sum(sgSpecAcceptRate)
	acceptLength := set.Sum(sgSpecAcceptLength)
	genThroughput := set.Sum(sgGenThroughput)

	// realtime decode tokens advance during generation and act as the live
	// activity signal in place of llama.cpp's n_decode_total.
	realtimeDecode := set.SumMatching(sgRealtimeTokens, map[string]string{"mode": "decode"})

	m := Metrics{
		PromptTokensTotal:    prompt,
		PromptCachedTotal:    cached,
		PredictedTokensTotal: generated,
		DecodeTotal:          realtimeDecode,
		TokensMax:            maxFloat(set.Sum(sgDecodeSumSeqLens), set.Sum(sgNumUsedTokens)),

		AvgPredictedTokensPerSec: genThroughput,
		AvgPromptTokensPerSec:    0,

		RequestsProcessing: set.Sum(sgNumRunningReqs),
		RequestsDeferred:   set.Sum(sgNumQueueReqs),
	}

	if hit := set.Sum(sgCacheHitRate); hit > 0 && prompt+cached <= 0 {
		// Prefer counter-derived hit rate; fall back to the gauge when the
		// server has not yet accumulated token counters.
		m.PromptCachedTotal = hit
		m.PromptTokensTotal = 1 - hit
	}

	if verifyCalls > 0 && acceptLength > 0 {
		// accept_length includes the bonus token; SpecSpeedup adds +1 to
		// accepted-per-draft, so store accepted drafts excluding the bonus.
		acceptedPerDraft := acceptLength - 1
		if acceptedPerDraft < 0 {
			acceptedPerDraft = 0
		}
		m.SpecDraftsTotal = verifyCalls
		m.SpecAcceptedTokensTotal = acceptedPerDraft * verifyCalls
		if acceptRate > 0 {
			m.SpecDraftTokensTotal = m.SpecAcceptedTokensTotal / acceptRate
		} else if acceptedPerDraft > 0 {
			m.SpecDraftTokensTotal = m.SpecAcceptedTokensTotal
		}
	}

	return m
}

// SGLangTokenUsage is the fraction of the KV pool currently occupied, in 0..1.
func SGLangTokenUsage(set promparse.Set) float64 {
	if u := set.Sum(sgTokenUsage); u > 0 {
		if u > 1 {
			return 1
		}
		return u
	}
	maxTok := set.Sum(sgMaxTotalTokens)
	if maxTok <= 0 {
		return 0
	}
	used := set.Sum(sgNumUsedTokens)
	if used <= 0 {
		used = set.Sum("sglang:kv_used_tokens")
	}
	u := used / maxTok
	if u > 1 {
		return 1
	}
	return u
}

// SGLangContextLen returns the configured maximum context length from metrics.
func SGLangContextLen(set promparse.Set) int {
	return int(set.Sum(sgContextLen))
}

// ModelNameFromPath returns the last path segment of a Hugging Face-style
// model id, suitable for the dashboard title.
func ModelNameFromPath(modelPath string) string {
	modelPath = strings.TrimSuffix(modelPath, "/")
	if modelPath == "" {
		return ""
	}
	return path.Base(modelPath)
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
