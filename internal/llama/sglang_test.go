package llama

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/pefman/ltop/internal/promparse"
)

func TestDetectEngineSGLang(t *testing.T) {
	f, err := os.Open("../promparse/testdata/sglang_metrics.txt")
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	set, err := promparse.Parse(f)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got := DetectEngine(set); got != EngineSGLang {
		t.Fatalf("DetectEngine = %q, want %q", got, EngineSGLang)
	}
}

func TestParseSGLangMetrics(t *testing.T) {
	f, err := os.Open("../promparse/testdata/sglang_metrics.txt")
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	set, err := promparse.Parse(f)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	m := ParseSGLangMetrics(set)

	if got := m.PromptTokensTotal; math.Abs(got-(228+215519)) > 1e-6 {
		t.Errorf("PromptTokensTotal = %v, want %v", got, 228+215519)
	}
	if got := m.PredictedTokensTotal; math.Abs(got-(88+9200)) > 1e-6 {
		t.Errorf("PredictedTokensTotal = %v, want %v", got, 88+9200)
	}
	if got := m.PromptCachedTotal; got != 116928 {
		t.Errorf("PromptCachedTotal = %v, want 116928", got)
	}
	if got := m.DecodeTotal; got != 9805 {
		t.Errorf("DecodeTotal (realtime decode) = %v, want 9805", got)
	}
	if !m.SpecEnabled() {
		t.Fatal("SpecEnabled = false, want true")
	}
	// accept_rate 0.8833…, accept_length 3.65 → accepted/draft = 2.65, speedup = 3.65
	if got := m.SpecAcceptanceRate(); math.Abs(got-0.8833333333333333) > 1e-6 {
		t.Errorf("SpecAcceptanceRate = %v, want ~0.8833", got)
	}
	if got := m.SpecTokensPerDraft(); math.Abs(got-2.65) > 1e-6 {
		t.Errorf("SpecTokensPerDraft = %v, want 2.65", got)
	}
	if got := m.SpecSpeedup(); math.Abs(got-3.65) > 1e-6 {
		t.Errorf("SpecSpeedup = %v, want 3.65", got)
	}
	if got := m.CacheHitRate(); math.Abs(got-116928.0/(116928.0+228+215519)) > 1e-9 {
		t.Errorf("CacheHitRate = %v", got)
	}
	if got := m.RequestsProcessing; got != 0 {
		t.Errorf("RequestsProcessing = %v, want 0", got)
	}
}

func TestPropsFromServerInfo(t *testing.T) {
	info := ServerInfo{
		ModelPath:          "RadixArk/Qwen3.8-27B-NVFP4",
		ServedModelName:    "RadixArk/Qwen3.8-27B-NVFP4",
		Version:            "0.5.17",
		ContextLength:      32768,
		MaxRunningRequests: 1,
		EnableMetrics:      true,
	}
	p := PropsFromServerInfo(info)
	if p.Engine != EngineSGLang {
		t.Errorf("Engine = %q", p.Engine)
	}
	if got := p.ModelName(); got != "Qwen3.8-27B-NVFP4" {
		t.Errorf("ModelName = %q, want Qwen3.8-27B-NVFP4", got)
	}
	if p.BuildInfo != "sglang 0.5.17" {
		t.Errorf("BuildInfo = %q", p.BuildInfo)
	}
	m := ModelFromServerInfo(info)
	if m.Meta.NCtx != 32768 || m.Meta.NCtxTrain != 32768 {
		t.Errorf("context = %d/%d", m.Meta.NCtx, m.Meta.NCtxTrain)
	}
}

func TestParseMetricsForRoutes(t *testing.T) {
	set, err := promparse.Parse(strings.NewReader("sglang:generation_tokens_total{model_name=\"m\"} 3\n"))
	if err != nil {
		t.Fatal(err)
	}
	m := ParseMetricsFor(EngineSGLang, set)
	if m.PredictedTokensTotal != 3 {
		t.Errorf("PredictedTokensTotal = %v, want 3", m.PredictedTokensTotal)
	}
}
