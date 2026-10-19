/*
 * ltop.c - minimal top(1) for llama.cpp's llama-server.
 * Plain C99 + POSIX: sockets, poll(2), termios, ANSI escapes. Zero dependencies.
 *
 *   cc -O2 -std=c99 -Wall -Wextra -o ltop ltop.c
 *   ltop [-1] [-n secs] [-k apikey] [-G] [http://host:port]
 *
 * Default URL: $LTOP_URL or http://127.0.0.1:8080.  API key: -k or $LTOP_API_KEY.
 * Interactive when stdout is a tty; otherwise (or with -1) prints one
 * "key value" snapshot (two scrapes, -n apart) suitable for grep/awk.
 * Needs llama-server --metrics (rates) and --slots (slot table).
 * Plain http only (put stunnel/socat in front for https).
 * Keys: q quit  p pause  +/- interval  z reset counters  r refresh now
 *
 * Simplified port of https://github.com/pefman/ltop (MIT). 
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE /* Ensures poll() and _SC_NPROCESSORS_ONLN are exposed on glibc */
#include <errno.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define MAXBODY (8 << 20)
#define MAXPOS 32
#define MAXSLOTS 64
#define HIST 240

/* ---------- metrics ---------- */
enum { PROMPT_TOK, PROMPT_CACHED, PROMPT_SEC, PRED_TOK, PRED_SEC, N_DECODE,
       TOK_MAX, SPEC_DRAFT_TOK, SPEC_ACC_TOK, SPEC_DRAFTS, AVG_PROMPT_TPS,
       AVG_PRED_TPS, REQ_PROC, REQ_DEFER, BUSY_PER_DECODE, NM };
static const char *mname[NM] = {
    "prompt_tokens_total", "prompt_tokens_cached_total", "prompt_seconds_total",
    "tokens_predicted_total", "tokens_predicted_seconds_total", "n_decode_total",
    "n_tokens_max", "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total", "spec_decode_num_drafts_total",
    "prompt_tokens_seconds", "predicted_tokens_seconds", "requests_processing",
    "requests_deferred", "n_busy_slots_per_decode" };
static const int counter[NM] = { 1,1,1,1,1,1,0,1,1,1,0,0,0,0,0 };

typedef struct { double v[NM]; double pos[MAXPOS]; int npos; } Metrics;
typedef struct { int id, n_ctx, task, proc, prompt, cached; } Slot;

/* ---------- global state ---------- */
static char host[256] = "127.0.0.1", port[16] = "8080", apikey[256];
static double interval = 1.0;
static volatile sig_atomic_t quit_flag, winch_flag;
static struct termios orig_tio;
static int tty_raw;

static Metrics prev, base;
static int have_prev, have_base;
static double prev_at, start_at;
static double dec_rate, pre_rate, dec_at = -1, pre_at = -1, steps_ps;
static double hist[HIST]; static int hist_n;
static char model[512], build[128]; static int total_slots; static double props_at = -1e9;
static Slot slots[MAXSLOTS]; static int nslots, have_slots, have_metrics, online;
static char errmsg[160]; static double scrape_ms;
static unsigned long long cpu_prev[2]; static double cpu_pct = -1;

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* ---------- HTTP/1.0 GET over a plain socket ---------- */
static int http_get(const char *path, char **out, size_t *outlen) {
    struct addrinfo hints = {0}, *res, *ai;
    int fd = -1, status = -1;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) { 
        snprintf(errmsg, sizeof errmsg, "cannot resolve %s", host); 
        return -1; 
    }
    for (ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = { 5, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { snprintf(errmsg, sizeof errmsg, "connect %s:%s: %s", host, port, strerror(errno)); return -1; }

    char req[1024];
    int n = snprintf(req, sizeof req, "GET %s HTTP/1.0\r\nHost: %s:%s\r\n%s%s%sConnection: close\r\n\r\n",
                     path, host, port, *apikey ? "Authorization: Bearer " : "", apikey, *apikey ? "\r\n" : "");
    if (write(fd, req, n) != n) { close(fd); return -1; }

    size_t cap = 65536, len = 0; 
    char *buf = malloc(cap);
    if (!buf) { close(fd); return -1; }
    
    ssize_t r;
    while ((r = read(fd, buf + len, cap - len - 1)) > 0) {
        len += r;
        if (len + 1 >= cap) {
            if (cap >= MAXBODY) break;
            char *nb = realloc(buf, cap *= 2); 
            if (!nb) { free(buf); buf = NULL; break; } 
            buf = nb;
        }
    }
    close(fd);
    if (!buf) return -1;
    
    buf[len] = 0;
    char *body = strstr(buf, "\r\n\r\n");
    if (strncmp(buf, "HTTP/1.", 7) || !body) { 
        free(buf); 
        snprintf(errmsg, sizeof errmsg, "bad HTTP reply on %s", path); 
        return -1; 
    }
    status = atoi(buf + 9);
    body += 4;
    *outlen = len - (body - buf);
    memmove(buf, body, *outlen + 1);
    *out = buf;
    return status;
}

/* ---------- Prometheus text exposition ---------- */
static void parse_metrics(char *s, Metrics *m) {
    memset(m, 0, sizeof *m);
    char *saveptr;
    /* Use strtok_r for POSIX thread-safety and robustness */
    for (char *line = strtok_r(s, "\n", &saveptr); line; line = strtok_r(NULL, "\n", &saveptr)) {
        if (*line == '#' || strncmp(line, "llamacpp:", 9)) continue;
        char *name = line + 9, *end = name + strcspn(name, "{ ");
        char *val = end, *lab = NULL;
        if (*end == '{') { lab = end; val = strchr(end, '}'); if (!val) continue; val++; }
        size_t nl = end - name;
        double v = strtod(val, NULL);
        if (!strncmp(name, "spec_decode_num_accepted_tokens_per_pos_total", nl) && nl == strlen("spec_decode_num_accepted_tokens_per_pos_total")) {
            char *p = lab ? strstr(lab, "position=\"") : NULL;
            if (p) { int i = atoi(p + 10); if (i >= 0 && i < MAXPOS) { m->pos[i] = v; if (i >= m->npos) m->npos = i + 1; } }
            continue;
        }
        for (int i = 0; i < NM; i++)
            if (strlen(mname[i]) == nl && !strncmp(name, mname[i], nl)) { m->v[i] = v; break; }
    }
}

/* ---------- tiny JSON helpers (enough for /props and /slots) ---------- */
static int json_str(const char *js, const char *key, char *out, size_t n) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat); if (!p) return 0;
    p = strchr(p + strlen(pat), ':'); if (!p) return 0;
    while (*++p == ' ');
    if (*p != '"') { snprintf(out, n, "%ld", strtol(p, NULL, 10)); return 1; }
    size_t i = 0;
    for (p++; *p && *p != '"' && i + 1 < n; p++) { if (*p == '\\' && p[1]) p++; out[i++] = *p; }
    out[i] = 0; return 1;
}

/* Walk the top-level array; read scalar keys only at depth 2 (slot object). */
static int parse_slots(const char *s, Slot *out, int max) {
    int depth = 0, n = 0; Slot cur = {0};
    for (const char *p = s; *p; p++) {
        if (*p == '"') {
            const char *k = ++p;
            for (; *p && *p != '"'; p++) if (*p == '\\' && p[1]) p++;
            if (!*p) break;
            size_t kl = p - k; const char *q = p + 1;
            while (*q == ' ') q++;
            if (depth != 2 || *q != ':') continue;
            q++; while (*q == ' ') q++;
            long v = (*q == 't') ? 1 : (*q == 'f') ? 0 : strtol(q, NULL, 10);
#define K(name, field) if (kl == strlen(name) && !strncmp(k, name, kl)) cur.field = (int)v
            K("id", id); K("n_ctx", n_ctx); K("id_task", task); K("is_processing", proc);
            K("n_prompt_tokens", prompt); K("n_prompt_tokens_cache", cached);
#undef K
        } else if (*p == '{' || *p == '[') {
            if (++depth == 2) memset(&cur, 0, sizeof cur);
        } else if (*p == '}' || *p == ']') {
            if (depth-- == 2 && n < max) out[n++] = cur;
        }
    }
    return n;
}

/* ---------- host sampling (Linux /proc; silently absent elsewhere) ---------- */
static void sample_cpu(void) {
    FILE *f = fopen("/proc/stat", "r"); if (!f) return;
    unsigned long long u, ni, sy, id, io, irq, sirq, st;
    if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &u, &ni, &sy, &id, &io, &irq, &sirq, &st) == 8) {
        unsigned long long tot = u + ni + sy + id + io + irq + sirq + st, idle = id + io;
        if (cpu_prev[0] && tot > cpu_prev[0])
            cpu_pct = 100.0 * (1.0 - (double)(idle - cpu_prev[1]) / (tot - cpu_prev[0]));
        cpu_prev[0] = tot; cpu_prev[1] = idle;
    }
    fclose(f);
}

static int mem_info(double *used, double *total) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return 0;
    char k[64]; unsigned long long v, tot = 0, avail = 0;
    while (fscanf(f, "%63s %llu kB\n", k, &v) == 2) {
        if (!strcmp(k, "MemTotal:")) tot = v;
        else if (!strcmp(k, "MemAvailable:")) avail = v;
    }
    fclose(f);
    *total = tot * 1024.0; *used = (tot - avail) * 1024.0;
    return tot > 0;
}

/* ---------- formatting ---------- */
static const char *si(double v, char *b) {
    if (v >= 1e9) sprintf(b, "%.2fG", v / 1e9); else if (v >= 1e6) sprintf(b, "%.2fM", v / 1e6);
    else if (v >= 1e3) sprintf(b, "%.1fk", v / 1e3); else sprintf(b, "%.0f", v);
    return b;
}
static const char *dur(double s, char *b) {
    long t = (long)s;
    if (t >= 3600) sprintf(b, "%ldh%02ldm", t / 3600, t / 60 % 60);
    else if (t >= 60) sprintf(b, "%ldm%02lds", t / 60, t % 60); else sprintf(b, "%lds", t);
    return b;
}
static char *bar(double frac, int w, char *b) {
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    int f = (int)(frac * w + 0.5); b[0] = '[';
    for (int i = 0; i < w; i++) b[i + 1] = i < f ? '|' : ' ';
    b[w + 1] = ']'; b[w + 2] = 0; return b;
}
static double sdiv(double a, double b) { return b > 0 ? a / b : 0; }

/* ---------- one poll ---------- */
static void poll_server(void) {
    double t = now(); char *body; size_t len; int st;
    errmsg[0] = 0; have_metrics = have_slots = 0;
    sample_cpu();

    st = http_get("/health", &body, &len);
    scrape_ms = (now() - t) * 1000;
    if (st < 0) { online = 0; return; }
    free(body);
    if (st == 401 || st == 403) { online = 0; snprintf(errmsg, sizeof errmsg, "unauthorized (-k apikey)"); return; }
    online = (st == 503) ? 2 : 1;   /* 2 = loading model */

    if (t - props_at > 10 && http_get("/props", &body, &len) == 200) {
        char old[sizeof model]; strcpy(old, model);
        json_str(body, "model_path", model, sizeof model);
        json_str(body, "build_info", build, sizeof build);
        char ts[32]; if (json_str(body, "total_slots", ts, sizeof ts)) total_slots = atoi(ts);
        if (*old && strcmp(old, model)) { have_prev = 0; dec_at = pre_at = -1; hist_n = 0; } /* model swapped */
        props_at = t; free(body);
    }

    if ((st = http_get("/slots", &body, &len)) == 200) { nslots = parse_slots(body, slots, MAXSLOTS); have_slots = 1; }
    if (st >= 0) free(body);

    Metrics cur;
    if ((st = http_get("/metrics", &body, &len)) != 200) { if (st >= 0) free(body); return; }
    parse_metrics(body, &cur); free(body);
    have_metrics = 1;

    if (have_prev && cur.v[N_DECODE] < prev.v[N_DECODE]) { have_prev = have_base = 0; hist_n = 0; dec_at = pre_at = -1; } /* restart */
    if (have_prev) {
        double wall = t - prev_at;
        if (wall > 0) {
            steps_ps = (cur.v[N_DECODE] - prev.v[N_DECODE]) / wall;
            double dt = cur.v[PRED_TOK] - prev.v[PRED_TOK], ds = cur.v[PRED_SEC] - prev.v[PRED_SEC];
            double pt = cur.v[PROMPT_TOK] - prev.v[PROMPT_TOK], ps = cur.v[PROMPT_SEC] - prev.v[PROMPT_SEC];
            if (dt > 0) { dec_rate = dt / (ds > 0 ? ds : wall); dec_at = t; }
            if (pt > 0) { pre_rate = pt / (ps > 0 ? ps : wall); pre_at = t; }
            memmove(hist, hist + (hist_n == HIST), sizeof(double) * (hist_n - (hist_n == HIST)));
            hist[hist_n < HIST ? hist_n++ : HIST - 1] = steps_ps;
        }
    }
    prev = cur; prev_at = t; have_prev = 1;
}

/* counters relative to the 'z' baseline */
static Metrics view(void) {
    Metrics m = prev;
    if (!have_base) return m;
    for (int i = 0; i < NM; i++) if (counter[i]) { m.v[i] -= base.v[i]; if (m.v[i] < 0) m.v[i] = 0; }
    for (int i = 0; i < m.npos; i++) { m.pos[i] -= base.pos[i]; if (m.pos[i] < 0) m.pos[i] = 0; }
    return m;
}

/* ---------- output ---------- */
static char out[65536]; static size_t olen;
static void put(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void put(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(out + olen, sizeof out - olen, fmt, ap); va_end(ap);
    if (n > 0) olen += (size_t)n < sizeof out - olen ? (size_t)n : sizeof out - olen - 1;
}
#define LINE(...) do { put(__VA_ARGS__); put("\033[K\r\n"); } while (0)

static int term_cols(void) {
    struct winsize ws; return ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col ? ws.ws_col : 80;
}

static void render(int paused) {
    char a[64], b[64], c[64], br[80]; double t = now(); int cols = term_cols();
    Metrics m = view();
    const char *mb = strrchr(model, '/'); mb = mb ? mb + 1 : (*model ? model : "?");
    olen = 0; put("\033[H");
    LINE(" ltop  %s  %s   %s%s", mb, build, online == 1 ? "* online" : online == 2 ? "* loading" : "x OFFLINE",
         paused ? "  [paused]" : "");
    LINE("  http://%s:%s  slots %d  up %s  scrape %.0fms  every %.1fs", host, port, total_slots,
         dur(t - start_at, a), scrape_ms, interval);
    if (*errmsg) LINE("  ! %s", errmsg);
    put("\033[K\r\n");
    if (cpu_pct >= 0) {
        double l = 0; FILE *f = fopen("/proc/loadavg", "r"); if (f) { if (fscanf(f, "%lf", &l) != 1) l = 0; fclose(f); }
        LINE("  CPU       %s %5.1f%%  %ld cores  load %.2f", bar(cpu_pct / 100, 24, br), cpu_pct, sysconf(_SC_NPROCESSORS_ONLN), l);
    }
    double mu, mt;
    if (mem_info(&mu, &mt)) LINE("  RAM       %s %5.1f%%  %.1fGiB of %.1fGiB", bar(mu / mt, 24, br), 100 * mu / mt, mu / 1073741824, mt / 1073741824);
    if (getenv("LTOP_NO_GPU") == NULL) {
        FILE *g = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name "
                        "--format=csv,noheader,nounits 2>/dev/null", "r");
        char ln[256]; int i = 0;
        while (g && fgets(ln, sizeof ln, g)) {
            double u, vu, vt, tc, w; char nm[128] = "";
            if (sscanf(ln, "%lf, %lf, %lf, %lf, %lf, %127[^\n]", &u, &vu, &vt, &tc, &w, nm) >= 5)
                LINE("  GPU%-2d     %s %5.0f%%  %.1f/%.1fGiB  %.0fC  %.0fW  %s", i++, bar(u / 100, 24, br), u, vu / 1024, vt / 1024, tc, w, nm);
        }
        if (g) pclose(g);
    }
    put("\033[K\r\n");
    if (!online) goto foot;
    if (!have_metrics) { LINE("  (no /metrics - start llama-server with --metrics)"); goto slots; }

    {   int w = cols - 36; if (w > HIST) w = HIST; if (w < 0) w = 0;
        double mx = 1e-9; for (int i = 0; i < hist_n; i++) if (hist[i] > mx) mx = hist[i];
        /* Each UTF-8 block character (▁▂▃▄▅▆▇█) is exactly 3 bytes. */
        static const char ramp[] = "▁▂▃▄▅▆▇█"; 
        char sp[HIST * 3 + 1]; int k = 0;
        for (int i = hist_n > w ? hist_n - w : 0; i < hist_n; i++) {
            int g = (int)(hist[i] / mx * 7 + 0.5); if (g > 7) g = 7; if (g < 0) g = 0;
            memcpy(sp + k, ramp + 3 * g, 3); k += 3;
        }
        sp[k] = 0;
        LINE("  DECODE    %8.1f steps/s  %s", steps_ps, sp); }
    if (dec_at >= 0) LINE("    tok/s   %8.1f  measured %s ago   lifetime %.1f", dec_rate, dur(t - dec_at, a), sdiv(m.v[PRED_TOK], m.v[PRED_SEC]));
    else             LINE("    tok/s       --    (updates when a request completes)   lifetime %.1f", sdiv(m.v[PRED_TOK], m.v[PRED_SEC]));
    if (pre_at >= 0) LINE("  PREFILL   %8.0f tok/s  measured %s ago   lifetime %.0f", pre_rate, dur(t - pre_at, a), sdiv(m.v[PROMPT_TOK], m.v[PROMPT_SEC]));
    else             LINE("  PREFILL       --    lifetime %.0f", sdiv(m.v[PROMPT_TOK], m.v[PROMPT_SEC]));
    LINE("  QUEUE     %.0f processing  %.0f deferred  %.2f slots/decode", m.v[REQ_PROC], m.v[REQ_DEFER], m.v[BUSY_PER_DECODE]);
    put("\033[K\r\n");
    { double hit = sdiv(m.v[PROMPT_CACHED], m.v[PROMPT_CACHED] + m.v[PROMPT_TOK]);
      LINE("  KV CACHE  %s %5.1f%%  %s reused / %s prefilled", bar(hit, 24, br), hit * 100, si(m.v[PROMPT_CACHED], a), si(m.v[PROMPT_TOK], b)); }
    if (m.v[SPEC_DRAFT_TOK] > 0 || m.v[SPEC_DRAFTS] > 0) {
        double acc = sdiv(m.v[SPEC_ACC_TOK], m.v[SPEC_DRAFT_TOK]), per = sdiv(m.v[SPEC_ACC_TOK], m.v[SPEC_DRAFTS]);
        LINE("  SPEC      %s %5.1f%%  %.2fx est speedup  %.2f accepted/draft", bar(acc, 24, br), acc * 100, m.v[SPEC_DRAFTS] > 0 ? per + 1 : 1, per);
        if (m.npos && m.v[SPEC_DRAFTS] > 0) {
            put("    draft  "); for (int i = 0; i < m.npos; i++) put(" pos%d %.0f%%", i, 100 * m.pos[i] / m.v[SPEC_DRAFTS]); put("\033[K\r\n");
        }
    }
slots:
    put("\033[K\r\n");
    if (!have_slots) LINE("  (no /slots - start llama-server with --slots)");
    else {
        LINE("  SLOT STATE      TASK     PROMPT     CACHED     CTX");
        for (int i = 0; i < nslots; i++) {
            double u = sdiv(slots[i].prompt, slots[i].n_ctx); if (u > 1) u = 1;
            LINE("  %-4d %-8s %7d %10d %10d  %5.1f%%  %s", slots[i].id, slots[i].proc ? "running" : "idle",
                 slots[i].task, slots[i].prompt, slots[i].cached, u * 100, bar(u, 12, br));
        }
    }
    if (have_metrics) {
        double rate = sdiv(m.v[PROMPT_TOK], m.v[PROMPT_SEC]); char d[64];
        put("\033[K\r\n");
        LINE("  STATS%s  %s generated  %s prefilled  %s cached  %s decodes  ~%s saved", have_base ? "*" : " ",
             si(m.v[PRED_TOK], a), si(m.v[PROMPT_TOK], b), si(m.v[PROMPT_CACHED], c), si(m.v[N_DECODE], br),
             dur(rate > 0 ? m.v[PROMPT_CACHED] / rate : 0, d));
    }
foot:
    put("\033[K\r\n");
    LINE("  q quit   p pause   +/- interval   z reset counters   r refresh");
    put("\033[J");
    for (size_t off = 0; off < olen;) { ssize_t w = write(STDOUT_FILENO, out + off, olen - off); if (w <= 0) break; off += w; }
}

/* plain "key value" lines for scripts */
static void print_once(void) {
    Metrics m = view();
    printf("online %d\nmodel %s\nbuild %s\n", online, model, build);
    if (have_metrics) {
        for (int i = 0; i < NM; i++) printf("%s %.6g\n", mname[i], m.v[i]);
        for (int i = 0; i < m.npos; i++) printf("spec_accepted_pos%d %.6g\n", i, m.pos[i]);
        printf("decode_steps_per_s %.3f\n", steps_ps);
        if (dec_at >= 0) printf("decode_tok_per_s %.3f\n", dec_rate);
        if (pre_at >= 0) printf("prefill_tok_per_s %.3f\n", pre_rate);
        printf("kv_cache_hit %.4f\n", sdiv(m.v[PROMPT_CACHED], m.v[PROMPT_CACHED] + m.v[PROMPT_TOK]));
    }
    for (int i = 0; have_slots && i < nslots; i++)
        printf("slot%d %s task=%d prompt=%d cached=%d n_ctx=%d\n", slots[i].id, slots[i].proc ? "running" : "idle",
               slots[i].task, slots[i].prompt, slots[i].cached, slots[i].n_ctx);
    if (*errmsg) printf("error %s\n", errmsg);
}

/* ---------- terminal ---------- */
static void term_restore(void) {
    if (!tty_raw) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_tio);
    (void)!write(STDOUT_FILENO, "\033[?25h\033[?1049l", 14);
    tty_raw = 0;
}
static void on_sig(int s) { if (s == SIGWINCH) winch_flag = 1; else quit_flag = 1; }

static void set_url(const char *u) {
    if (!strncmp(u, "https://", 8)) { fprintf(stderr, "ltop: https not supported; use a local http endpoint\n"); exit(2); }
    if (!strncmp(u, "http://", 7)) u += 7;
    char tmp[300]; snprintf(tmp, sizeof tmp, "%s", u);
    tmp[strcspn(tmp, "/")] = 0;                      /* drop /v1 or any path */
    char *colon = strrchr(tmp, ':'), *rb = strrchr(tmp, ']');
    if (colon && (!rb || colon > rb)) { *colon = 0; snprintf(port, sizeof port, "%s", colon + 1); }
    char *h = tmp; if (*h == '[') { h++; if ((rb = strchr(h, ']'))) *rb = 0; }
    snprintf(host, sizeof host, "%s", h);
}

int main(int argc, char **argv) {
    int once = 0, opt;
    if (getenv("LTOP_URL")) set_url(getenv("LTOP_URL"));
    if (getenv("LTOP_API_KEY")) snprintf(apikey, sizeof apikey, "%s", getenv("LTOP_API_KEY"));
    while ((opt = getopt(argc, argv, "1n:k:Gh")) != -1) switch (opt) {
        case '1': once = 1; break;
        case 'n': interval = atof(optarg); if (interval < 0.2) interval = 0.2; break;
        case 'k': snprintf(apikey, sizeof apikey, "%s", optarg); break;
        case 'G': setenv("LTOP_NO_GPU", "1", 1); break;
        default: fprintf(stderr, "usage: %s [-1] [-n secs] [-k apikey] [-G] [http://host:port]\n", argv[0]); return opt == 'h' ? 0 : 2;
    }
    if (optind < argc) set_url(argv[optind]);
    start_at = now();
    signal(SIGPIPE, SIG_IGN);

    if (once || !isatty(STDOUT_FILENO) || !isatty(STDIN_FILENO)) {
        poll_server();
        if (online && have_metrics) { struct timespec ts = { (time_t)interval, (long)((interval - (time_t)interval) * 1e9) }; nanosleep(&ts, NULL); poll_server(); }
        print_once();
        return online ? 0 : 1;
    }

    struct sigaction sa = {0}; sa.sa_handler = on_sig; sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL); sigaction(SIGHUP, &sa, NULL); sigaction(SIGWINCH, &sa, NULL);
    tcgetattr(STDIN_FILENO, &orig_tio);
    struct termios raw = orig_tio;
    raw.c_lflag &= ~(ICANON | ECHO); raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw); tty_raw = 1; atexit(term_restore);
    (void)!write(STDOUT_FILENO, "\033[?1049h\033[?25l\033[2J", 18);

    int paused = 0; double next = 0;
    while (!quit_flag) {
        double t = now();
        if (!paused && t >= next) { poll_server(); next = t + interval; render(paused); }
        else if (winch_flag) { winch_flag = 0; render(paused); }
        int ms = paused ? -1 : (int)((next - now()) * 1000); if (ms < 0 && !paused) ms = 0;
        struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
        if (poll(&pfd, 1, ms) > 0 && (pfd.revents & POLLIN)) {
            char k; if (read(STDIN_FILENO, &k, 1) != 1) continue;
            switch (k) {
                case 'q': case 'Q': quit_flag = 1; break;
                case 'p': paused = !paused; render(paused); break;
                case '+': case '=': interval += interval < 1 ? 0.25 : 1; render(paused); break;
                case '-': interval -= interval <= 1 ? 0.25 : 1; if (interval < 0.25) interval = 0.25; render(paused); break;
                case 'z': if (have_prev) { base = prev; have_base = 1; hist_n = 0; dec_at = pre_at = -1; } render(paused); break;
                case 'r': next = 0; break;
            }
        }
    }
    term_restore();
    return 0;
}