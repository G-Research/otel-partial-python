# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `partial-span-processor`, a Python package that extends the OpenTelemetry Python SDK to support partial spans. It implements a `PartialSpanProcessor` that exports heartbeat events for long-running spans and stop events when spans complete, allowing for better monitoring of active traces.

## Development Commands

### Building
```bash
python -m build
```

### Installation (Local)
```bash
pip install dist/partial_span_processor-0.0.x-py3-none-any.whl
```

### Running Tests
```bash
python -m unittest discover tests/
```

### Running Single Test
```bash
python -m unittest tests.partial_span_processor.test_partial_span_processor.TestPartialSpanProcessor.test_on_start
```

### Running Example
```bash
python example.py
```

## Architecture

### Core Components

**PartialSpanProcessor** (`src/partial_span_processor/__init__.py`): Main component that implements OpenTelemetry's `SpanProcessor` interface. Features:
- Tracks active spans in memory
- Manages two queues for heartbeat scheduling: delayed and ready
- Uses background worker thread for processing heartbeat events
- Exports span data as logs via configurable `LogExporter`

**PeekableQueue** (`src/partial_span_processor/peekable_queue.py`): Thread-safe queue extension that allows peeking at the first element without removing it.

**InMemoryLogExporter** (`tests/partial_span_processor/in_memory_log_exporter.py`): Test utility for capturing exported logs in memory.

### Key Design Patterns

- **Threading**: Uses background worker thread with condition variables for timed processing
- **Queue Management**: Dual-queue system separating initial delay from regular heartbeat intervals
- **Span Lifecycle**: Hooks into OpenTelemetry's `on_start` and `on_end` events
- **Data Serialization**: Converts span data to JSON format compatible with partial collector

### Configuration Parameters

- `heartbeat_interval_millis`: Frequency of heartbeat events (default: 5000ms)
- `initial_heartbeat_delay_millis`: Delay before first heartbeat (default: 5000ms)  
- `process_interval_millis`: Worker thread processing interval (default: 5000ms)

### Testing

Tests use `unittest` framework with mocking for span objects and datetime manipulation. The test suite covers parameter validation, span lifecycle events, and queue processing logic.

## Publishing

Package is published to PyPI via GitHub Actions workflow triggered by version tags. Version must be updated in `pyproject.toml` before creating a tag with format `vX.Y.Z`.