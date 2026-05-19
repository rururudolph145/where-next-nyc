# Concurrency in Python — Applied to Data Ingestion

## 1. What Is Concurrency?

Concurrency means making progress on multiple tasks during overlapping time periods. It doesn't necessarily mean things run at the exact same instant — it means your program doesn't sit idle waiting for one thing to finish before starting the next.

There are three main concurrency models in Python:

### Threading (`concurrent.futures.ThreadPoolExecutor`, `threading`)

Multiple threads share the same process and memory. Python's **GIL** (Global Interpreter Lock) prevents threads from executing Python bytecode in true parallel, but the GIL is *released during I/O operations* (network calls, disk reads). This makes threading ideal for **I/O-bound** work like API calls.

```
Thread 1:  [request >>>>>>>]  [parse]  [request >>>>>>>]  [parse]
Thread 2:       [request >>>>>>>]  [parse]  [request >>>>>>>]  [parse]
Thread 3:            [request >>>>>>>]  [parse]  [request >>>>>>>]  [parse]
               ↑ network wait — GIL released, other threads run
```

**Good for:** HTTP requests, file I/O, database queries
**Bad for:** CPU-heavy number crunching (all threads fight over the GIL)

### Multiprocessing (`concurrent.futures.ProcessPoolExecutor`, `multiprocessing`)

Spawns separate OS processes, each with its own Python interpreter and GIL. Achieves true parallelism for **CPU-bound** work, but has higher overhead (process creation, inter-process communication via serialisation).

```
Process 1:  [parse + transform ████████]
Process 2:  [parse + transform ████████]     ← true parallel on separate CPU cores
Process 3:  [parse + transform ████████]
```

**Good for:** DataFrame transformations, model training, heavy computation
**Bad for:** Tasks that need to share mutable state (requires explicit IPC)

### Async I/O (`asyncio`, `aiohttp`)

A single-threaded event loop that switches between tasks at `await` points. The most memory-efficient option for high-volume I/O, but requires an async-compatible ecosystem (can't just drop `requests` in — you need `aiohttp` or `httpx`).

```
Event loop:  [req1 start] [req2 start] [req3 start] [req1 done→parse] [req2 done→parse] ...
             ← one thread, no GIL issues, very low memory overhead →
```

**Good for:** Hundreds/thousands of concurrent HTTP calls
**Bad for:** CPU-bound work (blocks the event loop)

---

## 2. Where Concurrency Applies in Our Ingestion Pipeline

Look at the current `fetch_data` method in `app/ingestion/ingest.py`:

```python
# Current: sequential — each month waits for the previous one to finish
def fetch_data(self) -> list[str]:
    paths: list[str] = []
    year, month = s_year, s_month
    while (year, month) <= (e_year, e_month):
        paths.append(self.fetch_month(year, month))  # blocks until complete
        month += 1
        ...
    return paths
```

If we fetch Jan–Jun 2022 (6 months), and each month takes ~2 minutes of mostly *waiting for HTTP responses*, total wall time is ~12 minutes. The CPU is idle for the vast majority of that time.

### The bottleneck is I/O, not CPU

Each `fetch_month` call spends its time like this:

```
[resolve_dataset_id: HTTP call ~0.5s]
[_fetch_to_parquet page 1: HTTP ~2s] [cast_batch: CPU ~0.1s] [write: disk ~0.05s]
[_fetch_to_parquet page 2: HTTP ~2s] [cast_batch: CPU ~0.1s] [write: disk ~0.05s]
...
```

~95% of the time is network wait. This is the textbook case for threading.

---

## 3. Applying `ThreadPoolExecutor` to `fetch_data`

`concurrent.futures.ThreadPoolExecutor` is the simplest way to parallelise this. Here's how it would look:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_data(self, max_workers: int = 4) -> list[str]:
    """Fetch all configured months concurrently."""
    start_ym: str = self.config["ingest"]["start_year_month"]
    end_ym: str = self.config["ingest"]["end_year_month"]
    s_year, s_month = (int(x) for x in start_ym.split("-"))
    e_year, e_month = (int(x) for x in end_ym.split("-"))

    # Build list of (year, month) to fetch
    months: list[tuple[int, int]] = []
    year, month = s_year, s_month
    while (year, month) <= (e_year, e_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    # Submit all months to the thread pool
    paths: list[str] = [""] * len(months)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(self.fetch_month, y, m): i
            for i, (y, m) in enumerate(months)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            paths[idx] = future.result()  # raises if fetch_month raised

    return paths
```

### What changed

| Aspect | Before (sequential) | After (threaded) |
|---|---|---|
| 6 months, ~2 min each | ~12 min total | ~3 min total (4 workers) |
| Complexity | Simple loop | `ThreadPoolExecutor` + futures |
| Error handling | Exception stops the loop | `future.result()` re-raises per-month |
| API load | 1 concurrent connection | Up to `max_workers` connections |

### Why this is safe

- Each `fetch_month` writes to its **own file** (`data/bronze/2022_01/2022_01.parquet`), so no write conflicts.
- `requests.Session` is **not** thread-safe, but we can create one session per thread, or use a thread-local session. Alternatively, each `fetch_month` could create its own session.
- The Socrata API is a public API — moderate parallelism (3–4 workers) is fine, but don't set `max_workers=50` or you'll get rate-limited (HTTP 429).

---

## 4. Thread Safety Considerations

When sharing objects across threads, you need to think about race conditions:

### Safe (no shared mutable state)
```python
# Each thread gets its own output path and writes to its own file.
# self.config is read-only after __init__. This is fine.
path = self.fetch_month(year, month)
```

### Unsafe (shared mutable state)
```python
# BAD: multiple threads appending to the same list
paths = []
def worker(y, m):
    paths.append(self.fetch_month(y, m))  # race condition!
```

### Fix: use thread-safe structures or collect via futures
```python
# GOOD: each future returns its result; the main thread assembles them
future_to_idx = {pool.submit(self.fetch_month, y, m): i for i, (y, m) in enumerate(months)}
for future in as_completed(future_to_idx):
    paths[future_to_idx[future]] = future.result()
```

### Making `requests.Session` thread-safe

Our `_make_session()` creates a single session stored as `self._session`. For thread safety, either:

1. **Create a session per thread** using `threading.local()`:
   ```python
   import threading

   _thread_local = threading.local()

   def _get_session(self) -> requests.Session:
       if not hasattr(_thread_local, "session"):
           _thread_local.session = _make_session()
       return _thread_local.session
   ```

2. **Create a session per call** (simpler, slightly more overhead):
   ```python
   def fetch_month(self, year: int, month: int) -> str:
       session = _make_session()  # new session for this call
       ...
   ```

---

## 5. Applying Concurrency to the `pipeline.run_all` Runner

The pipeline in `pipeline/run_all.py` runs ingest → clean sequentially per month. We could parallelise this too, but we need to be more careful:

```python
# Current sequential pipeline
for year, month in months:
    bronze_path = ingestor.fetch_month(year, month)   # I/O-bound
    df = pd.read_parquet(bronze_path)                  # I/O-bound
    cleaned = cleaner.clean(df)                        # CPU-bound
    cleaned.to_parquet(silver_path)                    # I/O-bound
```

### Option A: Parallelise only the fetch (simplest)

Fetch all months concurrently, then clean them sequentially:

```python
# Phase 1: fetch all months in parallel (I/O-bound → threading)
with ThreadPoolExecutor(max_workers=4) as pool:
    bronze_paths = list(pool.map(
        lambda ym: ingestor.fetch_month(*ym), months
    ))

# Phase 2: clean sequentially (CPU-bound, simpler to debug)
for (year, month), bronze_path in zip(months, bronze_paths):
    df = pd.read_parquet(bronze_path)
    cleaned = cleaner.clean(df)
    cleaned.to_parquet(silver_path)
```

### Option B: Pipeline parallelism (fetch next while cleaning current)

Use a producer-consumer pattern where one thread fetches while another cleans:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

work_queue: queue.Queue[tuple[str, str]] = queue.Queue()

def fetch_worker(ingestor, months):
    """Producer: fetch months and push paths into the queue."""
    for year, month in months:
        path = ingestor.fetch_month(year, month)
        work_queue.put((f"{year}_{month:02d}", path))
    work_queue.put(("DONE", ""))  # sentinel

def clean_worker(cleaner):
    """Consumer: pull paths from queue, clean, and save."""
    while True:
        tag, bronze_path = work_queue.get()
        if tag == "DONE":
            break
        df = pd.read_parquet(bronze_path)
        cleaned = cleaner.clean(df)
        cleaned.to_parquet(f"data/silver/yellow_tripdata_cleaned_{tag}.parquet")
```

This keeps the network and CPU busy at the same time. In practice, **Option A is usually sufficient** — the fetch is the slow part by far.

---

## 6. When NOT to Use Concurrency

- **Debugging**: Concurrent bugs (race conditions, deadlocks) are hard to reproduce. Start sequential, add concurrency after correctness is proven.
- **Small workloads**: Fetching 1 month? Threading overhead isn't worth it.
- **Rate-limited APIs**: If the API caps you at 2 req/s, 10 threads won't help — they'll just get 429 errors.
- **Shared mutable state**: If tasks need to read/write the same DataFrame or file, you'll need locks, and the complexity may not be worth it.

---

## 7. Quick Reference

| Scenario | Best tool | Why |
|---|---|---|
| Fetch multiple months from API | `ThreadPoolExecutor` | I/O-bound, simple API |
| Fetch hundreds of days | `asyncio` + `aiohttp` | Many concurrent connections, low memory |
| Clean/transform DataFrames in parallel | `ProcessPoolExecutor` | CPU-bound, bypasses GIL |
| Fetch 1 month | Sequential | No benefit from concurrency |

### Key stdlib imports

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading   # for locks, thread-local storage
import asyncio     # for async/await event loop
import queue       # for thread-safe producer/consumer
```
