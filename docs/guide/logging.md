# Logging

> **TL;DR** — Pyneapple uses [loguru](https://github.com/Delgan/loguru) for structured logging. On import the default level is `WARNING` and no output is produced in worker processes. Covers the default behavior, `configure_logging()`, the `PYNEAPPLE_QUIET` environment variable, the `--verbose` CLI flag, and writing logs to a file.

---

## Default behavior

Importing `pyneapple` removes loguru's built-in handler and installs a `WARNING`-level stderr sink. This means:

- `WARNING`, `ERROR`, and `CRITICAL` messages appear on stderr.
- `INFO` and `DEBUG` messages are suppressed unless you call `configure_logging()`.
- Worker processes spawned by joblib (parallel fitting) inherit `PYNEAPPLE_QUIET=1` from the parent and produce no console output.

---

## `configure_logging`

```python
import pyneapple

pyneapple.configure_logging(level="DEBUG")
```

Removes the current stderr sink and adds a new one at the requested level. Call this once at the top of a script before any fitting code.

### Log levels

| Level | When to use |
|---|---|
| `"DEBUG"` | Verbose — every model build, per-step IDEAL progress, solver details |
| `"INFO"` | Normal script use — config loaded, fitter built, fit complete |
| `"WARNING"` | Default — only unexpected but recoverable situations |
| `"ERROR"` | Errors that abort the current operation |
| `"CRITICAL"` | Fatal errors |

### Custom format

Pass any [loguru `add()` keyword](https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.add) as extra kwargs:

```python
pyneapple.configure_logging(
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)
```

---

## Silence all output — `PYNEAPPLE_QUIET`

Set the environment variable `PYNEAPPLE_QUIET=1` before importing `pyneapple` to suppress every log message, including warnings:

```bash
PYNEAPPLE_QUIET=1 python fit.py
```

Or from Python before the import:

```python
import os
os.environ["PYNEAPPLE_QUIET"] = "1"
import pyneapple  # no stderr sink is added
```

This is the mechanism used internally for worker processes. It is also useful in notebooks where any stderr output would appear inline.

---

## `--verbose` CLI flag

All Pyneapple CLI commands accept `--verbose` to enable `DEBUG`-level output:

```bash
pyneapple-pixelwise --image dwi.nii.gz --bval dwi.bval --config config.toml --verbose
```

Equivalent to calling `pyneapple.configure_logging(level="DEBUG")` at the start of a script.

---

## Writing logs to a file

`configure_logging()` only manages the stderr sink. Add a separate file sink directly via loguru:

```python
from loguru import logger
import pyneapple

pyneapple.configure_logging(level="INFO")          # keep stderr at INFO
logger.add("fit.log", level="DEBUG", rotation="10 MB")  # full debug log to file
```

Loguru sinks are independent — the stderr level and the file level can differ.

> [!NOTE]
> `logger.add()` is additive. If you call it multiple times without `logger.remove()` in between you will accumulate duplicate sinks.
