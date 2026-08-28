package collect

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/pefman/ltop/internal/gpu"
	"github.com/pefman/ltop/internal/llama"
)

// fakeServer serves the llama.cpp / SGLang endpoints ltop reads, with counters
// the test controls.
type fakeServer struct {
	*httptest.Server
	metrics     string
	slots       []llama.Slot
	modelPath   string
	noMetrics   bool
	noSlots     bool
	unhealthy   bool
	requireKey  string
	sglang      bool
	serverInfo  string
	healthBody  string
}

func newFakeServer(t *testing.T) *fakeServer {
	t.Helper()
	f := &fakeServer{modelPath: "/models/a.gguf"}

	mux := http.NewServeMux()
	auth := func(w http.ResponseWriter, r *http.Request) bool {
		if f.requireKey == "" {
			return true
		}
		if r.Header.Get("Authorization") == "Bearer "+f.requireKey {
			return true
		}
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if f.unhealthy {
			http.Error(w, `{"status":"loading model"}`, http.StatusServiceUnavailable)
			return
		}
		if f.healthBody != "" {
			fmt.Fprint(w, f.healthBody)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if f.noMetrics {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		fmt.Fprint(w, f.metrics)
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if f.sglang {
			http.Error(w, `{"detail":"Not Found"}`, http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(llama.Props{ModelPath: f.modelPath, BuildInfo: "test"})
	})
	mux.HandleFunc("/get_server_info", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if !f.sglang {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if f.serverInfo != "" {
			fmt.Fprint(w, f.serverInfo)
			return
		}
		fmt.Fprintf(w, `{"model_path":%q,"served_model_name":%q,"version":"0.5.17","context_length":32768,"enable_metrics":true,"max_running_requests":1}`,
			f.modelPath, f.modelPath)
	})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if f.noSlots || f.sglang {
			http.Error(w, "disabled", http.StatusNotImplemented)
			return
		}
		_ = json.NewEncoder(w).Encode(f.slots)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		if f.sglang {
			fmt.Fprintf(w, `{"data":[{"id":%q,"max_model_len":32768}]}`, f.modelPath)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"m","meta":{"n_ctx":4096}}]}`)
	})

	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Close)
	return f
}

func (f *fakeServer) setCounters(decode, tokens, tokSecs, prompt, promptSecs float64) {
	f.metrics = fmt.Sprintf(
		"llamacpp:n_decode_total %f\n"+
			"llamacpp:tokens_predicted_total %f\n"+
			"llamacpp:tokens_predicted_seconds_total %f\n"+
			"llamacpp:prompt_tokens_total %f\n"+
			"llamacpp:prompt_seconds_total %f\n",
		decode, tokens, tokSecs, prompt, promptSecs)
}

// The first poll cannot produce a rate because rates come from deltas.
func TestFirstPollHasNoRates(t *testing.T) {
	f := newFakeServer(t)
	f.setCounters(100, 1000, 10, 500, 1)

	c := New(f.URL, "")
	snap := c.Poll(context.Background())

	if !snap.Online {
		t.Fatalf("offline: %v", snap.Err)
	}
	if snap.DecodeStepsPerSec != 0 {
		t.Errorf("DecodeStepsPerSec = %v, want 0", snap.DecodeStepsPerSec)
	}
	if snap.HasDecodeMeasured {
		t.Error("HasDecodeMeasured = true on first poll")
	}
}

// Token counters only move when a request completes; the measured rate must be
// held with a growing age rather than dropping to zero.
func TestMeasuredRateIsStickyAndAges(t *testing.T) {
	f := newFakeServer(t)
	c := New(f.URL, "")
	ctx := context.Background()

	f.setCounters(100, 1000, 10, 500, 1)
	c.Poll(ctx)

	// 200 tokens generated in 4 seconds of decode time = 50 tok/s.
	f.setCounters(120, 1200, 14, 500, 1)
	snap := c.Poll(ctx)

	if !snap.HasDecodeMeasured {
		t.Fatal("HasDecodeMeasured = false after counters advanced")
	}
	if got := snap.DecodeTokensPerSec; got != 50 {
		t.Errorf("DecodeTokensPerSec = %v, want 50", got)
	}
	if !snap.Generating {
		t.Error("Generating = false while token counters advanced")
	}

	// Counters freeze, as they do mid-generation.
	f.setCounters(140, 1200, 14, 500, 1)
	snap = c.Poll(ctx)

	if got := snap.DecodeTokensPerSec; got != 50 {
		t.Errorf("rate not held: got %v, want 50", got)
	}
	if snap.DecodeAge <= 0 {
		t.Error("DecodeAge did not advance while counters were frozen")
	}
	if snap.Generating {
		t.Error("Generating = true while token counters were frozen")
	}
	if snap.DecodeStepsPerSec <= 0 {
		t.Error("DecodeStepsPerSec = 0 although n_decode_total advanced")
	}
}

// A restarted server resets its counters; ltop must rebaseline instead of
// reporting a negative rate.
func TestCounterResetRebaselines(t *testing.T) {
	f := newFakeServer(t)
	c := New(f.URL, "")
	ctx := context.Background()

	f.setCounters(100, 1000, 10, 500, 1)
	c.Poll(ctx)
	f.setCounters(120, 1200, 14, 600, 2)
	c.Poll(ctx)

	if len(c.StepHist.Values()) == 0 {
		t.Fatal("no history recorded before restart")
	}

	f.setCounters(2, 5, 0.1, 3, 0.1)
	snap := c.Poll(ctx)

	if !snap.Restarted {
		t.Error("Restarted = false after counters went backwards")
	}
	if snap.DecodeStepsPerSec < 0 || snap.DecodeTokensPerSec < 0 {
		t.Errorf("negative rate after restart: steps=%v tokens=%v",
			snap.DecodeStepsPerSec, snap.DecodeTokensPerSec)
	}
	if len(c.StepHist.Values()) != 1 {
		t.Errorf("history not cleared on restart: %d entries", len(c.StepHist.Values()))
	}
}

func TestOfflineServerYieldsSnapshotNotPanic(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := c.Poll(context.Background())

	if snap.Online {
		t.Error("Online = true for an unreachable server")
	}
	if snap.Err == nil {
		t.Error("Err = nil for an unreachable server")
	}
}

// A server started without --metrics is still online; blanking the dashboard
// would misreport a working server as down.
func TestServerWithoutMetricsStaysOnline(t *testing.T) {
	f := newFakeServer(t)
	f.noMetrics = true
	f.slots = []llama.Slot{{ID: 0, NCtx: 4096, PromptTokens: 2048}}

	snap := New(f.URL, "").Poll(context.Background())

	if !snap.Online {
		t.Fatalf("Online = false without --metrics: %v", snap.Err)
	}
	if snap.HasMetrics {
		t.Error("HasMetrics = true although /metrics returned 404")
	}
	if snap.Err != nil {
		t.Errorf("Err = %v, want nil for a disabled endpoint", snap.Err)
	}
	if !snap.HasSlots || len(snap.Slots) != 1 {
		t.Error("slots not read while metrics were unavailable")
	}
	if snap.ContextPressure != 0.5 {
		t.Errorf("ContextPressure = %v, want 0.5", snap.ContextPressure)
	}
}

// A server started without --slots must likewise stay online.
func TestServerWithoutSlotsStaysOnline(t *testing.T) {
	f := newFakeServer(t)
	f.noSlots = true
	f.setCounters(100, 1000, 10, 500, 1)

	snap := New(f.URL, "").Poll(context.Background())

	if !snap.Online || !snap.HasMetrics {
		t.Fatalf("online=%v metrics=%v err=%v", snap.Online, snap.HasMetrics, snap.Err)
	}
	if snap.HasSlots {
		t.Error("HasSlots = true although /slots returned 501")
	}
	if snap.Err != nil {
		t.Errorf("Err = %v, want nil for a disabled endpoint", snap.Err)
	}
}

func TestLoadingServerReportsLoading(t *testing.T) {
	f := newFakeServer(t)
	f.unhealthy = true
	f.setCounters(100, 1000, 10, 500, 1)

	snap := New(f.URL, "").Poll(context.Background())

	if !snap.Online {
		t.Fatalf("Online = false while loading: %v", snap.Err)
	}
	if !snap.Loading {
		t.Error("Loading = false for a 503 health response")
	}
}

func TestAPIKeyIsSent(t *testing.T) {
	f := newFakeServer(t)
	f.requireKey = "secret-token"
	f.setCounters(100, 1000, 10, 500, 1)

	if snap := New(f.URL, "").Poll(context.Background()); snap.Online || !snap.NeedsAuth {
		t.Errorf("without key: online=%v needsAuth=%v", snap.Online, snap.NeedsAuth)
	}
	if snap := New(f.URL, "secret-token").Poll(context.Background()); !snap.Online || !snap.HasMetrics {
		t.Errorf("with key: online=%v metrics=%v err=%v", snap.Online, snap.HasMetrics, snap.Err)
	}
}

func TestContextPressureUsesFullestSlot(t *testing.T) {
	slots := []llama.Slot{
		{NCtx: 1000, PromptTokens: 100},
		{NCtx: 1000, PromptTokens: 900},
		{NCtx: 1000, PromptTokens: 400},
	}
	if got := contextPressure(slots); got != 0.9 {
		t.Errorf("contextPressure = %v, want 0.9", got)
	}
	if got := contextPressure(nil); got != 0 {
		t.Errorf("contextPressure(nil) = %v, want 0", got)
	}
}

// Efficiency is only meaningful while decoding; a stale rate against idle GPU
// power would otherwise report a rising, meaningless figure.
func TestEfficiencyRequiresActiveDecoding(t *testing.T) {
	base := Snapshot{
		GPUs:               []gpu.Device{{PowerWatts: 300, HasPower: true}},
		DecodeTokensPerSec: 60,
		HasDecodeMeasured:  true,
	}

	active := base
	active.DecodeStepsPerSec = 20
	active.DecodeAge = time.Second
	deriveEfficiency(&active)
	if !active.HasEfficiency {
		t.Fatal("HasEfficiency = false while decoding")
	}
	if got := active.TokensPerJoule; got != 0.2 {
		t.Errorf("TokensPerJoule = %v, want 0.2", got)
	}

	idle := base
	idle.DecodeStepsPerSec = 0
	idle.DecodeAge = time.Second
	deriveEfficiency(&idle)
	if idle.HasEfficiency {
		t.Error("HasEfficiency = true while idle")
	}

	stale := base
	stale.DecodeStepsPerSec = 20
	stale.DecodeAge = time.Minute
	deriveEfficiency(&stale)
	if stale.HasEfficiency {
		t.Error("HasEfficiency = true for a stale measurement")
	}
}

func TestHistoryRingEvictsOldest(t *testing.T) {
	h := NewHistory(3)
	for _, v := range []float64{1, 2, 3, 4, 5} {
		h.Push(v)
	}
	got := h.Values()
	if len(got) != 3 || got[0] != 3 || got[2] != 5 {
		t.Errorf("Values = %v, want [3 4 5]", got)
	}
	if h.Max() != 5 {
		t.Errorf("Max = %v, want 5", h.Max())
	}
	if h.Last() != 5 {
		t.Errorf("Last = %v, want 5", h.Last())
	}

	h.Reset()
	if len(h.Values()) != 0 || h.Max() != 0 || h.Last() != 0 {
		t.Error("Reset did not clear the ring")
	}
}

func TestEnergyAccumulates(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := Snapshot{GPUs: []gpu.Device{{PowerWatts: 360, HasPower: true}}}
	start := time.Now()

	c.accumulateEnergy(&snap, start)
	if snap.EnergyWh != 0 {
		t.Errorf("first sample = %v Wh, want 0 (no interval yet)", snap.EnergyWh)
	}

	// 360 W held for 10 s is exactly 1 Wh.
	c.accumulateEnergy(&snap, start.Add(10*time.Second))
	if !snap.HasEnergy {
		t.Fatal("HasEnergy = false after two samples")
	}
	if math.Abs(snap.EnergyWh-1) > 1e-9 {
		t.Errorf("EnergyWh = %v, want 1", snap.EnergyWh)
	}
}

// A suspended machine reports a huge gap; charging it as energy would invent
// kilowatt-hours that were never drawn.
func TestEnergyIgnoresImplausibleGap(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := Snapshot{GPUs: []gpu.Device{{PowerWatts: 400, HasPower: true}}}
	start := time.Now()

	c.accumulateEnergy(&snap, start)
	c.accumulateEnergy(&snap, start.Add(6*time.Hour))
	if snap.EnergyWh != 0 {
		t.Errorf("EnergyWh = %v after a 6h gap, want 0", snap.EnergyWh)
	}
}

func TestEnergySkippedWithoutPowerReadings(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := Snapshot{GPUs: []gpu.Device{{UtilPercent: 50, HasUtil: true}}}

	c.accumulateEnergy(&snap, time.Now())
	c.accumulateEnergy(&snap, time.Now().Add(time.Second))
	if snap.HasEnergy {
		t.Error("HasEnergy = true without any power readings")
	}
}

func TestEnergyAccumulatesPerGPU(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := Snapshot{GPUs: []gpu.Device{
		{Index: 0, PowerWatts: 360, HasPower: true},
		{Index: 1, PowerWatts: 180, HasPower: true},
	}}
	start := time.Now()

	c.accumulateEnergy(&snap, start)
	c.accumulateEnergy(&snap, start.Add(10*time.Second))

	// 360 W for 10 s is 1 Wh; 180 W for 10 s is 0.5 Wh.
	if math.Abs(snap.GPUEnergyWh[0]-1) > 1e-9 {
		t.Errorf("GPU0 EnergyWh = %v, want 1", snap.GPUEnergyWh[0])
	}
	if math.Abs(snap.GPUEnergyWh[1]-0.5) > 1e-9 {
		t.Errorf("GPU1 EnergyWh = %v, want 0.5", snap.GPUEnergyWh[1])
	}
	if math.Abs(snap.EnergyWh-1.5) > 1e-9 {
		t.Errorf("total EnergyWh = %v, want 1.5", snap.EnergyWh)
	}
}

func TestResetStatsRebasesEnergy(t *testing.T) {
	c := New("http://127.0.0.1:1", "")
	snap := Snapshot{GPUs: []gpu.Device{
		{Index: 0, PowerWatts: 360, HasPower: true},
	}}
	start := time.Now()

	c.accumulateEnergy(&snap, start)
	c.accumulateEnergy(&snap, start.Add(10*time.Second))
	if math.Abs(snap.EnergyWh-1) > 1e-9 {
		t.Fatalf("pre-reset EnergyWh = %v, want 1", snap.EnergyWh)
	}

	c.ResetStats()
	c.accumulateEnergy(&snap, start.Add(10*time.Second))
	if snap.EnergyWh != 0 {
		t.Errorf("EnergyWh just after reset = %v, want 0", snap.EnergyWh)
	}
	if snap.GPUEnergyWh[0] != 0 {
		t.Errorf("GPU0 EnergyWh just after reset = %v, want 0", snap.GPUEnergyWh[0])
	}

	c.accumulateEnergy(&snap, start.Add(20*time.Second))
	if math.Abs(snap.EnergyWh-1) > 1e-9 {
		t.Errorf("EnergyWh after 10s in new window = %v, want 1", snap.EnergyWh)
	}
	if math.Abs(snap.GPUEnergyWh[0]-1) > 1e-9 {
		t.Errorf("GPU0 EnergyWh after 10s in new window = %v, want 1", snap.GPUEnergyWh[0])
	}
}

func TestResetStatsRebasesTotals(t *testing.T) {
	f := newFakeServer(t)
	c := New(f.URL, "")
	ctx := context.Background()

	f.setCounters(100, 1000, 10, 500, 1)
	c.Poll(ctx)
	f.setCounters(120, 1200, 14, 700, 2)
	if snap := c.Poll(ctx); snap.Raw.PredictedTokensTotal != 1200 {
		t.Fatalf("pre-reset total = %v, want 1200", snap.Raw.PredictedTokensTotal)
	}

	c.ResetStats()

	// Same counters immediately after reset means nothing has happened yet.
	snap := c.Poll(ctx)
	if !snap.StatsReset {
		t.Fatal("StatsReset = false after ResetStats")
	}
	if snap.Raw.PredictedTokensTotal != 0 {
		t.Errorf("generated = %v just after reset, want 0", snap.Raw.PredictedTokensTotal)
	}
	if snap.Raw.DecodeTotal != 0 {
		t.Errorf("decodes = %v just after reset, want 0", snap.Raw.DecodeTotal)
	}

	// Further work counts from the baseline, not from server start.
	f.setCounters(140, 1500, 20, 900, 3)
	snap = c.Poll(ctx)
	if got := snap.Raw.PredictedTokensTotal; got != 300 {
		t.Errorf("generated = %v, want 300 (1500-1200)", got)
	}
	if got := snap.Raw.PromptTokensTotal; got != 200 {
		t.Errorf("prefilled = %v, want 200 (900-700)", got)
	}
	if got := snap.Raw.DecodeTotal; got != 20 {
		t.Errorf("decodes = %v, want 20 (140-120)", got)
	}
	if snap.StatsSince <= 0 {
		t.Error("StatsSince did not advance")
	}
}

// A restart zeroes the server's counters, so a baseline taken against the old
// ones would make every total negative.
func TestServerRestartClearsBaseline(t *testing.T) {
	f := newFakeServer(t)
	c := New(f.URL, "")
	ctx := context.Background()

	f.setCounters(100, 1000, 10, 500, 1)
	c.Poll(ctx)
	f.setCounters(120, 1200, 14, 700, 2)
	c.Poll(ctx)
	c.ResetStats()

	f.setCounters(3, 20, 0.5, 10, 0.2)
	snap := c.Poll(ctx)

	if !snap.Restarted {
		t.Error("Restarted = false after counters went backwards")
	}
	if snap.StatsReset {
		t.Error("StatsReset still true after a server restart")
	}

	f.setCounters(5, 40, 1, 20, 0.4)
	snap = c.Poll(ctx)
	if snap.Raw.PredictedTokensTotal != 40 {
		t.Errorf("generated = %v, want the absolute 40 after baseline was dropped",
			snap.Raw.PredictedTokensTotal)
	}
}

func TestSGLangPollMapsMetricsAndIdentity(t *testing.T) {
	f := newFakeServer(t)
	f.sglang = true
	f.healthBody = ""
	f.modelPath = "RadixArk/Qwen3.8-27B-NVFP4"
	f.metrics = `
sglang:prompt_tokens_total{model_name="m"} 100
sglang:generation_tokens_total{model_name="m"} 50
sglang:cached_tokens_total{cache_source="device",model_name="m"} 200
sglang:realtime_tokens_total{mode="decode",model_name="m"} 40
sglang:num_running_reqs{model_name="m"} 1
sglang:num_queue_reqs{model_name="m"} 2
sglang:token_usage{model_name="m"} 0.25
sglang:spec_verify_calls_total{model_name="m"} 10
sglang:spec_accept_rate{model_name="m"} 0.8
sglang:spec_accept_length{model_name="m"} 3
sglang:gen_throughput{model_name="m"} 12
`

	c := New(f.URL, "")
	ctx := context.Background()
	snap := c.Poll(ctx)

	if !snap.Online {
		t.Fatalf("offline: %v", snap.Err)
	}
	if !snap.HasMetrics {
		t.Fatal("HasMetrics = false")
	}
	if snap.HasSlots {
		t.Error("HasSlots = true for SGLang")
	}
	if snap.Props.Engine != llama.EngineSGLang {
		t.Errorf("Engine = %q, want sglang", snap.Props.Engine)
	}
	if got := snap.Props.ModelName(); got != "Qwen3.8-27B-NVFP4" {
		t.Errorf("ModelName = %q", got)
	}
	if snap.Model.Meta.NCtx != 32768 {
		t.Errorf("NCtx = %d, want 32768", snap.Model.Meta.NCtx)
	}
	if snap.ContextPressure != 0.25 {
		t.Errorf("ContextPressure = %v, want 0.25", snap.ContextPressure)
	}
	if snap.Raw.PredictedTokensTotal != 50 || snap.Raw.PromptTokensTotal != 100 {
		t.Errorf("tokens gen=%v prompt=%v", snap.Raw.PredictedTokensTotal, snap.Raw.PromptTokensTotal)
	}
	if !snap.Raw.SpecEnabled() {
		t.Error("SpecEnabled = false")
	}

	// Second poll: realtime decode advances → live rate.
	f.metrics = `
sglang:prompt_tokens_total{model_name="m"} 150
sglang:generation_tokens_total{model_name="m"} 80
sglang:cached_tokens_total{cache_source="device",model_name="m"} 220
sglang:realtime_tokens_total{mode="decode",model_name="m"} 70
sglang:num_running_reqs{model_name="m"} 1
sglang:num_queue_reqs{model_name="m"} 0
sglang:token_usage{model_name="m"} 0.4
sglang:spec_verify_calls_total{model_name="m"} 12
sglang:spec_accept_rate{model_name="m"} 0.8
sglang:spec_accept_length{model_name="m"} 3
sglang:gen_throughput{model_name="m"} 30
`
	time.Sleep(20 * time.Millisecond)
	snap = c.Poll(ctx)
	if snap.DecodeStepsPerSec <= 0 {
		t.Errorf("DecodeStepsPerSec = %v, want live decode activity", snap.DecodeStepsPerSec)
	}
	if !snap.HasDecodeMeasured || snap.DecodeTokensPerSec <= 0 {
		t.Errorf("decode measured=%v rate=%v", snap.HasDecodeMeasured, snap.DecodeTokensPerSec)
	}
	if !snap.HasPrefillMeasured || snap.PrefillTokensPerSec <= 0 {
		t.Errorf("prefill measured=%v rate=%v", snap.HasPrefillMeasured, snap.PrefillTokensPerSec)
	}
}

func TestSGLangEmptyHealthIsOnline(t *testing.T) {
	f := newFakeServer(t)
	f.sglang = true
	f.healthBody = "   "
	f.modelPath = "org/model"
	f.metrics = "sglang:generation_tokens_total{model_name=\"m\"} 1\n"

	snap := New(f.URL, "").Poll(context.Background())
	if !snap.Online {
		t.Fatalf("empty health body should be online: %v", snap.Err)
	}
}
