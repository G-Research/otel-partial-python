# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from typing import TYPE_CHECKING

from google.protobuf import json_format
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.sdk._logs import LogData, LogRecord
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import TraceFlags

from .peekable_queue import PeekableQueue

if TYPE_CHECKING:
  from opentelemetry import context as context_api
  from opentelemetry.sdk._logs.export import LogExporter
  from opentelemetry.sdk.resources import Resource

WORKER_THREAD_NAME = "OtelPartialSpanProcessor"
DEFAULT_HEARTBEAT_INTERVAL_MILLIS = 5000
DEFAULT_INITIAL_HEARTBEAT_DELAY_MILLIS = 5000
DEFAULT_PROCESS_INTERVAL_MILLIS = 5000

_logger = logging.getLogger(__name__)

def validate_parameters(log_exporter, heartbeat_interval_millis,
    initial_heartbeat_delay_millis, process_interval_millis):
  if log_exporter is None:
    msg = "log_exporter must not be None"
    raise ValueError(msg)

  if heartbeat_interval_millis <= 0:
    msg = "heartbeat_interval_millis must be greater than 0"
    raise ValueError(msg)

  if initial_heartbeat_delay_millis < 0:
    msg = "initial_heartbeat_delay_millis must be greater or equal to 0"
    raise ValueError(msg)

  if process_interval_millis <= 0:
    msg = "process_interval_millis must be greater than 0"
    raise ValueError(msg)


class PartialSpanProcessor(SpanProcessor):

  def __init__(
      self,
      log_exporter: LogExporter,
      heartbeat_interval_millis: int = DEFAULT_HEARTBEAT_INTERVAL_MILLIS,
      initial_heartbeat_delay_millis: int = DEFAULT_INITIAL_HEARTBEAT_DELAY_MILLIS,
      process_interval_millis: int = DEFAULT_PROCESS_INTERVAL_MILLIS,
      resource: Resource | None = None,
  ) -> None:
    validate_parameters(log_exporter, heartbeat_interval_millis,
                        initial_heartbeat_delay_millis, process_interval_millis)

    self.log_exporter = log_exporter
    self.heartbeat_interval_millis = heartbeat_interval_millis
    self.initial_heartbeat_delay_millis = initial_heartbeat_delay_millis
    self.process_interval_millis = process_interval_millis
    self.resource = resource

    self.active_spans = {}
    self.delayed_heartbeat_spans: PeekableQueue[tuple[int, datetime.datetime]] = \
      PeekableQueue()
    self.delayed_heartbeat_spans_lookup: set[int] = set()
    self.ready_heartbeat_spans: PeekableQueue[
      tuple[int, datetime.datetime]] = PeekableQueue()
    self.lock = threading.Lock()

    self.done = False
    self.condition = threading.Condition(threading.Lock())
    self.worker_thread = threading.Thread(
      name=WORKER_THREAD_NAME, target=self.worker, daemon=True,
    )
    self.worker_thread.start()

  def worker(self) -> None:
    while not self.done:
      with self.condition:
        self.condition.wait(self.process_interval_millis / 1000)
        if self.done:
          break

        self.process_delayed_heartbeat_spans()
        self.process_ready_heartbeat_spans()

  def on_start(self, span: Span,
      parent_context: context_api.Context | None = None) -> None:
    with self.lock:
      self.active_spans[span.context.span_id] = span
      self.delayed_heartbeat_spans_lookup.add(span.context.span_id)

      next_heartbeat_time = datetime.datetime.now() + datetime.timedelta(
        milliseconds=self.initial_heartbeat_delay_millis)
      self.delayed_heartbeat_spans.put(
        (span.context.span_id, next_heartbeat_time))

  def on_end(self, span: ReadableSpan) -> None:
    is_delayed_heartbeat_pending = False
    with self.lock:
      self.active_spans.pop(span.context.span_id)

      if span.context.span_id in self.delayed_heartbeat_spans_lookup:
        is_delayed_heartbeat_pending = True
        self.delayed_heartbeat_spans_lookup.remove(span.context.span_id)

    if is_delayed_heartbeat_pending:
      return

    self.export_log(span, get_stop_attributes())

  def export_log(self, span, attributes: dict[str, str]) -> None:
    log_data = self.get_log_data(span, attributes)
    try:
      self.log_exporter.export([log_data])
    except Exception:
      _logger.exception("Exception while exporting logs.")

  def shutdown(self) -> None:
    # signal the worker thread to finish and then wait for it
    self.done = True
    with self.condition:
      self.condition.notify_all()
    self.worker_thread.join()

  def get_heartbeat_attributes(self) -> dict[str, str]:
    return {
      "partial.event": "heartbeat",
      "partial.frequency": str(self.heartbeat_interval_millis) + "ms",
      "partial.body.type": "json/v1",
    }

  def get_log_data(self, span: Span, attributes: dict[str, str]) -> LogData:
    instrumentation_scope = span.instrumentation_scope if hasattr(span,
                                                                  "instrumentation_scope") else None
    span_context = span.get_span_context()
    parent = span.parent

    enc_spans = encode_spans([span]).resource_spans
    traces_data = trace_pb2.TracesData()
    traces_data.resource_spans.extend(enc_spans)
    serialized_traces_data = json_format.MessageToJson(traces_data)

    # FIXME/HACK replace serialized traceId, spanId, and parentSpanId (if present) values as string comparison
    # possible issue is when there are multiple spans in the same trace.
    # currently that should not be the case.
    # trace_id and span_id are stored as int.
    # when serializing it gets serialized to bytes.
    # that is not inline with partial collector.
    traces_dict = json.loads(serialized_traces_data)
    for resource_span in traces_dict.get("resourceSpans", []):
      for scope_span in resource_span.get("scopeSpans", []):
        for span in scope_span.get("spans", []):
          span["traceId"] = f"{span_context.trace_id:032x}"
          span["spanId"] = f"{span_context.span_id:016x}"
          if parent:
            span["parentSpanId"] = f"{parent.span_id:016x}"

    serialized_traces_data = json.dumps(traces_dict, separators=(",", ":"))

    log_record = LogRecord(
      timestamp=time.time_ns(),
      observed_timestamp=time.time_ns(),
      trace_id=span_context.trace_id,
      span_id=span_context.span_id,
      trace_flags=TraceFlags().get_default(),
      severity_text="INFO",
      severity_number=SeverityNumber.INFO,
      body=serialized_traces_data,
      resource=self.resource,
      attributes=attributes,
    )
    return LogData(
      log_record=log_record, instrumentation_scope=instrumentation_scope,
    )

  def process_delayed_heartbeat_spans(self) -> None:
    spans_to_be_logged = []
    with (self.lock):
      now = datetime.datetime.now()
      while True:
        if self.delayed_heartbeat_spans.empty():
          break

        (span_id, next_heartbeat_time) = self.delayed_heartbeat_spans.peek()
        if next_heartbeat_time > now:
          break

        self.delayed_heartbeat_spans_lookup.discard(span_id)
        self.delayed_heartbeat_spans.get()

        span = self.active_spans.get(span_id)
        if span:
          spans_to_be_logged.append(span)

          next_heartbeat_time = now + datetime.timedelta(
            milliseconds=self.heartbeat_interval_millis)
          self.ready_heartbeat_spans.put((span_id, next_heartbeat_time))

    for span in spans_to_be_logged:
      self.export_log(span, self.get_heartbeat_attributes())

  def process_ready_heartbeat_spans(self) -> None:
    spans_to_be_logged = []
    now = datetime.datetime.now()
    with self.lock:
      while True:
        if self.ready_heartbeat_spans.empty():
          break

        (span_id, next_heartbeat_time) = self.ready_heartbeat_spans.peek()
        if next_heartbeat_time > now:
          break

        self.ready_heartbeat_spans.get()

        span = self.active_spans.get(span_id)
        if span:
          spans_to_be_logged.append(span)

          next_heartbeat_time = now + datetime.timedelta(
            milliseconds=self.heartbeat_interval_millis)
          self.ready_heartbeat_spans.put((span_id, next_heartbeat_time))

    for span in spans_to_be_logged:
      self.export_log(span, self.get_heartbeat_attributes())


def get_stop_attributes() -> dict[str, str]:
  return {
    "partial.event": "stop",
    "partial.body.type": "json/v1",
  }
