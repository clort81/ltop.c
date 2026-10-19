# ltop.c

A dependency-free, single-file POSIX C monitor for `llama.cpp`'s `llama-server` metrics. 

![ltop.c](ltop.c.gif)

## Philosophy

Modern software development has forgotten how to write a simple command-line utility. A tool that merely fetches a few HTTP endpoints and prints formatted text should not require a massive toolchain, a megabyte-sized binary, or a sprawling graph of transitive dependencies. 

This project is a corrective measure. It is written in strict C99, relies exclusively on standard POSIX APIs, and adheres to the Unix philosophy: write programs that do one thing well, work together, and handle text streams. It embodies 1980s coding idioms—procedural design, deterministic memory usage, and absolute transparency—rejecting the modern habit of treating simple CLI tools as enterprise-grade application frameworks.

# Zero Dependencies:

- Avoids linking against heavy libraries (libcurl, libcjson), resulting in a tiny binary (~30KB vs ~300KB+) and instant startup time. It can be dropped on any POSIX system and compiled immediately.

# Zero-Allocation JSON Parsing:

Just what is needed for the job. The custom parse_slots state machine is written in minimal C. It parses the JSON in a single O(N) pass with zero heap allocations. In contrast, libcjson builds an entire tree of dynamically allocated nodes, which is overkill and slower for this predictable, flat data structure.

# Elegant Sparkline Rendering:

Shiny!

## Features

- **Zero Dependencies**: No `libcurl`, no `libcjson`, no `ncurses`, no Go runtime. 
- **Single File**: The entire program resides in one readable `ltop.c` file. No build systems, no Makefiles, no package managers.
- **Microscopic Footprint**: Compiles to a ~30KB statically optimizable binary in a fraction of a second.
- **Pure VT100**: Uses direct ANSI escape sequences for terminal control and rendering. No terminfo database lookups or heavy terminal libraries.
- **Scriptable**: Automatically detects non-interactive TTYs and outputs clean, space-separated key-value pairs suitable for `awk`, `grep`, or `prometheus-node-exporter` textfile collection.

## The Case Against the "Modern" Implementation

This project is a simplified, hardened port of the original Go-based `ltop`. While the original is functional, it exhibits several modern anti-patterns that are unacceptable for a lightweight system utility:

| Metric | Original Go Implementation | This POSIX C Implementation |
| :--- | :--- | :--- |
| **Build Toolchain** | Requires Go compiler, module resolution, and network access to fetch dependencies. | Standard `cc`. Compiles offline in milliseconds. |
| **Binary Size** | 15MB - 50MB+ (includes entire Go runtime and garbage collector). | ~30KB (pure machine code, no runtime overhead). |
| **Dependencies** | Dozens of transitive dependencies for HTTP, JSON, and terminal rendering. | Exactly zero. The OS provides everything needed. |
| **JSON Parsing** | Heavy reflection-based unmarshaling, allocating hundreds of temporary objects per request. | Custom, zero-allocation state machine. Single `O(N)` pass over the string. |
| **Scope Creep** | Includes "config wizards", "self-updaters", "server discovery", and complex color libraries. | Strictly monitors metrics. Does one thing, perfectly. |

Adding a "self-updater" or a "configuration wizard" to a tool whose only job is to read an HTTP endpoint is a failure of scope. This version strips away the cruft and focuses entirely on performance, reliability, and clarity.

## Technical Highlights

1. **Zero-Allocation JSON Parsing**: The `/slots` endpoint is parsed using a custom, depth-tracking state machine. It reads scalar values directly from the string buffer in a single pass without a single call to `malloc`.
2. **Raw POSIX Sockets**: HTTP/1.0 requests are constructed and sent via standard `socket()`, `connect()`, and `write()` calls. This eliminates the need for `libcurl` and its associated initialization overhead and hidden state.
3. **Deterministic UTF-8 Sparklines**: The decode speed sparkline exploits the fact that the Unicode block characters (`▁▂▃▄▅▆▇█`) are contiguous in the UTF-8 table and each is exactly 3 bytes. Rendering is achieved via direct pointer arithmetic (`memcpy(sp + k, ramp + 3 * g, 3)`), avoiding array-of-pointers indirection and ensuring maximum cache efficiency.
4. **Robust Terminal Handling**: Uses `poll(2)` with precise microsecond-aware timing for non-blocking input, ensuring the UI remains responsive without busy-waiting or spawning threads.

## Build

Ensure you have a standard C compiler (GCC or Clang) installed. No development packages beyond the standard C library are required.

```bash
cc -O2 -std=c99 -Wall -Wextra -o ltop ltop.c
```

## Usage

# Interactive mode (default when attached to a TTY)
./ltop

# Point to a specific host and port
./ltop http://192.168.1.50:8080

# Non-interactive mode: outputs a single snapshot for scripting
./ltop -1

# Non-interactive mode: outputs two snapshots, 2 seconds apart, to calculate rates
./ltop -1 -n 2

# Use an API key if your server requires authentication
./ltop -k your_secret_api_key

### Interactive Controls

When running in a terminal, the following keys are active:
- `q` or `Q`: Quit immediately.
- `p`: Pause/resume screen updates.
- `+` or `=`: Increase the polling interval.
- `-`: Decrease the polling interval (minimum 0.25s).
- `z`: Reset the baseline counters (useful for measuring a specific generation run).
- `r`: Force an immediate refresh.

## Design Constraints

- **No HTTPS**: This tool speaks raw HTTP/1.0. If your `llama-server` is behind a reverse proxy requiring HTTPS, use `stunnel`, `socat`, or `nginx` to terminate TLS locally, and point `ltop` at `http://127.0.0.1`. A monitoring tool should not be burdened with cryptographic library dependencies.
- **Linux-Centric Host Metrics**: CPU and RAM sampling read directly from `/proc/stat` and `/proc/meminfo`. On non-Linux POSIX systems, these specific lines will silently omit, while the core `llama-server` metrics will continue to function perfectly.
- **No Object-Oriented Patterns**: The code is strictly procedural. State is managed via explicit structs and global static variables where appropriate for a single-instance CLI tool, avoiding the abstraction overhead and indirection of OOP paradigms.

## License

MIT. Derived from the original pefman/ltop concept, but rewritten from the ground up to adhere to strict Unix engineering principles.

