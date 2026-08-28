package collect

import (
	"context"
	"errors"
	"maps"
	"sync"
	"time"

	"github.com/pefman/ltop/internal/gpu"
	"github.com/pefman/ltop/internal/host"
	"github.com/pefman/ltop/internal/llama"
	"github.com/pefman/ltop/internal/promparse"
)

// propsInterval limits how often /props (or SGLang /get_server_info) is
// re-read; it is static apart from model swaps and sleep transitions.
const propsInterval = 10 * time.Second

// HistorySize is the number of samples retained for sparklines.
const HistorySize = 240

// Collector polls one inference server and maintains derived state.
type Collector struct {
	client *llama.Client
	gpus   *gpu.Multi
	hostS  *host.Sampler

	mu       sync.Mutex
	prev     llama.Metrics
	prevAt   time.Time
	havePrev bool
	props    llama.Props
	model    llama.Model
	propsAt  time.Time
	startAt  time.Time
	engine   llama.Engine

	lastModel           string
	modelUnmatched      bool
	lastDecodeRate      float64
	lastPrefillRate     float64
	lastDecodeAt        time.Time
	lastPrefillAt       time.Time
	energyWh            float64
	energyByGPU         map[int]float64
	lastEnergyAt        time.Time
	baseline            llama.Metrics
	baselineSet         bool
	baselineAt          time.Time
	baselineEnergy      float64
	baselineEnergyByGPU map[int]float64

	DecodeHist  *History
	PrefillHist *History
	StepHist    *History
}

// New returns a collector for the given endpoint.
func New(endpoint, apiKey string) *Collector {
	return &Collector{
		client:      llama.New(endpoint, apiKey),
		gpus:        gpu.NewMulti(),
		hostS:       host.NewSampler(),
		startAt:     time.Now(),
		DecodeHist:  NewHistory(HistorySize),
		PrefillHist: NewHistory(HistorySize),
		StepHist:    NewHistory(HistorySize),
	}
}

// Uptime is how long ltop has been observing this server.
func (c *Collector) Uptime() time.Duration { return time.Since(c.startAt) }

// ResetStats rebases the totals onto the current counter values.
//
// Neither llama.cpp nor SGLang offers a way to zero its own counters, so ltop
// records where they stand and reports the difference from here on, letting a
// single workload be measured without restarting the server.
func (c *Collector) ResetStats() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.baseline, c.baselineSet, c.baselineAt = c.prev, c.havePrev, time.Now()
	c.baselineEnergy = c.energyWh
	c.baselineEnergyByGPU = maps.Clone(c.energyByGPU)
	// prev is deliberately kept: rates and restart detection compare against
	// the server's absolute counters and are unaffected by the display window.
	c.clearHistory()
}

// Endpoint returns the server root being polled.
func (c *Collector) Endpoint() string { return c.client.BaseURL() }

// HasGPU reports whether any GPU backend was detected.
func (c *Collector) HasGPU() bool { return c.gpus.Available() }

// Poll performs one scrape and returns a derived snapshot. A snapshot is
// always returned, with Online false and Err set when the server is
// unreachable, so the caller can render an error state rather than exit.
//
// Liveness comes from /health alone. A server started without --metrics or
// --slots is still online and still worth showing; those panels degrade
// individually rather than blanking the dashboard.
func (c *Collector) Poll(ctx context.Context) Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()

	start := time.Now()
	snap := Snapshot{At: start}

	// Host and GPU are sampled regardless of server reachability so the
	// dashboard stays useful while the inference server is restarting.
	snap.Host = c.hostS.Sample()
	snap.GPUs = c.gpus.Sample(ctx)
	c.accumulateEnergy(&snap, start)

	health, err := c.client.Health(ctx)
	snap.ScrapeR = time.Since(start)
	if err != nil {
		snap.Err = err
		snap.NeedsAuth = errors.Is(err, llama.ErrUnauthorized)
		return snap
	}
	snap.Online = true
	snap.Loading = health.Loading()

	var (
		absolute llama.Metrics
		set      promparse.Set
	)
	if scraped, err := c.client.Metrics(ctx); err == nil {
		set = scraped
		if c.engine == llama.EngineUnknown {
			c.engine = llama.DetectEngine(set)
		}
		absolute = llama.ParseMetricsFor(c.engine, set)
		snap.Raw = absolute
		snap.HasMetrics = true
		if c.baselineSet {
			snap.Raw = absolute.Since(c.baseline)
			snap.StatsReset = true
			snap.StatsSince = start.Sub(c.baselineAt)
		}
	} else if !errors.Is(err, llama.ErrEndpointDisabled) {
		snap.Err = err
	}

	c.refreshProps(ctx, start, &snap)
	snap.Props = c.props
	snap.Model = c.model
	snap.ModelUnmatched = c.modelUnmatched

	if c.engine.Is(llama.EngineSGLang) {
		if set != nil {
			snap.ContextPressure = llama.SGLangTokenUsage(set)
		}
	} else {
		if slots, err := c.client.Slots(ctx); err == nil {
			snap.Slots = slots
			snap.HasSlots = true
		} else if !errors.Is(err, llama.ErrEndpointDisabled) {
			snap.Err = err
		}
		snap.ContextPressure = contextPressure(snap.Slots)
	}

	if snap.HasMetrics {
		c.deriveRates(&snap, start, absolute)
		deriveEfficiency(&snap)

		c.DecodeHist.Push(snap.DecodeTokensPerSec)
		c.PrefillHist.Push(snap.PrefillTokensPerSec)
		c.StepHist.Push(snap.DecodeStepsPerSec)

		c.prev, c.prevAt, c.havePrev = absolute, start, true
	}
	snap.ScrapeR = time.Since(start)
	return snap
}

// deriveRates computes the live decode-step rate and measured throughput.
//
// llama.cpp only folds a request's token and time totals into /metrics once the
// request finishes. Between completions those counters are frozen, so the
// measured tok/s is carried forward with an age rather than falling to zero,
// and n_decode_total supplies the live activity signal in the meantime.
//
// SGLang has no busy-second counters. It advances realtime_tokens_total during
// generation (mapped onto DecodeTotal) and exposes gen_throughput as a gauge;
// when token counters move, rates are taken against wall time instead.
func (c *Collector) deriveRates(snap *Snapshot, now time.Time, cur llama.Metrics) {
	if !c.havePrev {
		return
	}

	if counterWentBackwards(c.prev, cur) {
		snap.Restarted = true
		// A restart zeroes the server's counters, so any baseline taken
		// against the old ones would make every total negative.
		c.baselineSet = false
		snap.StatsReset = false
		c.reset()
		return
	}

	wall := now.Sub(c.prevAt).Seconds()
	if wall <= 0 {
		return
	}

	if steps := cur.DecodeTotal - c.prev.DecodeTotal; steps >= 0 {
		snap.DecodeStepsPerSec = steps / wall
	}

	decodeTokens := cur.PredictedTokensTotal - c.prev.PredictedTokensTotal
	decodeSecs := cur.PredictedSecondsTotal - c.prev.PredictedSecondsTotal
	prefillTokens := cur.PromptTokensTotal - c.prev.PromptTokensTotal
	prefillSecs := cur.PromptSecondsTotal - c.prev.PromptSecondsTotal

	switch {
	case decodeTokens > 0 && decodeSecs > 0:
		// Dividing tokens by the server's own busy seconds rather than by wall
		// time yields the rate achieved while generating (llama.cpp).
		c.lastDecodeRate = decodeTokens / decodeSecs
		c.lastDecodeAt = now
		snap.Generating = true
	case decodeTokens > 0:
		c.lastDecodeRate = decodeTokens / wall
		c.lastDecodeAt = now
		snap.Generating = true
	case cur.AvgPredictedTokensPerSec > 0 && c.engine.Is(llama.EngineSGLang):
		// SGLang's gen_throughput gauge is live while decoding; use it when
		// the cumulative generation counter has not yet moved this interval.
		c.lastDecodeRate = cur.AvgPredictedTokensPerSec
		c.lastDecodeAt = now
		snap.Generating = true
		if snap.DecodeStepsPerSec == 0 {
			snap.DecodeStepsPerSec = cur.AvgPredictedTokensPerSec
		}
	case cur.DecodeTotal > c.prev.DecodeTotal && c.engine.Is(llama.EngineSGLang):
		// realtime decode tokens advanced; treat the wall rate as measured
		// throughput when generation_tokens_total is still catching up.
		c.lastDecodeRate = (cur.DecodeTotal - c.prev.DecodeTotal) / wall
		c.lastDecodeAt = now
		snap.Generating = true
	}

	switch {
	case prefillTokens > 0 && prefillSecs > 0:
		c.lastPrefillRate = prefillTokens / prefillSecs
		c.lastPrefillAt = now
	case prefillTokens > 0:
		c.lastPrefillRate = prefillTokens / wall
		c.lastPrefillAt = now
	}

	if !c.lastDecodeAt.IsZero() {
		snap.DecodeTokensPerSec = c.lastDecodeRate
		snap.DecodeAge = now.Sub(c.lastDecodeAt)
		snap.HasDecodeMeasured = true
	}
	if !c.lastPrefillAt.IsZero() {
		snap.PrefillTokensPerSec = c.lastPrefillRate
		snap.PrefillAge = now.Sub(c.lastPrefillAt)
		snap.HasPrefillMeasured = true
	}
}

func (c *Collector) reset() {
	c.clearHistory()
	c.havePrev = false
}

// clearHistory drops the sparklines and held rates without discarding the
// counter baseline that rate derivation depends on.
func (c *Collector) clearHistory() {
	c.DecodeHist.Reset()
	c.PrefillHist.Reset()
	c.StepHist.Reset()
	c.lastDecodeAt = time.Time{}
	c.lastPrefillAt = time.Time{}
	c.lastDecodeRate = 0
	c.lastPrefillRate = 0
}

func (c *Collector) refreshProps(ctx context.Context, now time.Time, snap *Snapshot) {
	if !c.propsAt.IsZero() && now.Sub(c.propsAt) < propsInterval {
		return
	}

	props, model, unmatched, ok := c.loadServerIdentity(ctx)
	if !ok {
		return
	}
	c.propsAt = now
	c.engine = props.EngineOr()

	if c.lastModel != "" && props.ModelPath != "" && props.ModelPath != c.lastModel {
		snap.ModelChanged = true
		c.reset()
	}
	if props.ModelPath != "" {
		c.lastModel = props.ModelPath
	}
	c.props = props
	c.model = model
	c.modelUnmatched = unmatched
}

// loadServerIdentity reads llama.cpp /props or SGLang /get_server_info and
// pairs it with /v1/models metadata when available.
func (c *Collector) loadServerIdentity(ctx context.Context) (llama.Props, llama.Model, bool, bool) {
	if props, err := c.client.Props(ctx); err == nil && props.ModelPath != "" {
		props.Engine = llama.EngineLlama
		var (
			model     llama.Model
			unmatched bool
		)
		if models, err := c.client.Models(ctx); err == nil {
			var ok bool
			model, ok = llama.MatchModel(models, props)
			unmatched = !ok && len(models) > 0
		}
		return props, model, unmatched, true
	}

	if info, err := c.client.ServerInfo(ctx); err == nil && (info.ModelPath != "" || info.ServedModelName != "") {
		props := llama.PropsFromServerInfo(info)
		model := llama.ModelFromServerInfo(info)
		if models, err := c.client.Models(ctx); err == nil {
			if matched, ok := llama.MatchModel(models, props); ok {
				// Prefer OpenAI id when present; keep context from server info.
				if matched.ID != "" {
					model.ID = matched.ID
				}
				if matched.Meta.NCtx > 0 {
					model.Meta.NCtx = matched.Meta.NCtx
				}
				if matched.Meta.NCtxTrain > 0 {
					model.Meta.NCtxTrain = matched.Meta.NCtxTrain
				} else if model.Meta.NCtxTrain == 0 {
					model.Meta.NCtxTrain = model.Meta.NCtx
				}
				if matched.Meta.FType != "" {
					model.Meta.FType = matched.Meta.FType
				}
				if matched.Meta.NParams > 0 {
					model.Meta.NParams = matched.Meta.NParams
				}
				if matched.Meta.Size > 0 {
					model.Meta.Size = matched.Meta.Size
				}
			}
		}
		return props, model, false, true
	}

	if c.engine.Is(llama.EngineSGLang) && c.props.Engine.Is(llama.EngineSGLang) {
		return c.props, c.model, c.modelUnmatched, true
	}
	return llama.Props{}, llama.Model{}, false, false
}

// counterWentBackwards detects a server restart, which resets all counters.
func counterWentBackwards(prev, cur llama.Metrics) bool {
	return cur.PredictedTokensTotal < prev.PredictedTokensTotal ||
		cur.PromptTokensTotal < prev.PromptTokensTotal ||
		cur.DecodeTotal < prev.DecodeTotal
}

func contextPressure(slots []llama.Slot) float64 {
	worst := 0.0
	for _, s := range slots {
		if u := s.ContextUsed(); u > worst {
			worst = u
		}
	}
	return worst
}

// accumulateEnergy integrates GPU power draw over the interval between polls.
func (c *Collector) accumulateEnergy(snap *Snapshot, now time.Time) {
	watts := 0.0
	readable := false
	for _, d := range snap.GPUs {
		if d.HasPower {
			watts += d.PowerWatts
			readable = true
		}
	}
	if !readable {
		return
	}
	if c.energyByGPU == nil {
		c.energyByGPU = make(map[int]float64)
	}
	if !c.lastEnergyAt.IsZero() {
		hours := now.Sub(c.lastEnergyAt).Hours()
		// Guard against a suspended laptop reporting an enormous interval.
		if hours > 0 && hours < 1 {
			c.energyWh += watts * hours
			for _, d := range snap.GPUs {
				if d.HasPower {
					c.energyByGPU[d.Index] += d.PowerWatts * hours
				}
			}
		}
	}
	c.lastEnergyAt = now
	snap.EnergyWh, snap.HasEnergy = c.energyWh-c.baselineEnergy, true
	snap.GPUEnergyWh = make(map[int]float64, len(c.energyByGPU))
	for idx, wh := range c.energyByGPU {
		snap.GPUEnergyWh[idx] = wh - c.baselineEnergyByGPU[idx]
	}
}

// deriveEfficiency computes tokens generated per joule of GPU energy.
//
// It is only meaningful while the server is actually decoding: once generation
// stops, GPU power falls to idle while the last measured token rate persists,
// which would otherwise produce a steadily rising and meaningless figure.
func deriveEfficiency(snap *Snapshot) {
	const staleAfter = 5 * time.Second
	if snap.DecodeStepsPerSec <= 0 || !snap.HasDecodeMeasured || snap.DecodeAge > staleAfter {
		return
	}

	watts := 0.0
	for _, d := range snap.GPUs {
		if d.HasPower {
			watts += d.PowerWatts
		}
	}
	if watts <= 0 || snap.DecodeTokensPerSec <= 0 {
		return
	}
	snap.TokensPerJoule = snap.DecodeTokensPerSec / watts
	snap.HasEfficiency = true
}
