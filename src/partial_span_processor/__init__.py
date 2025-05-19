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

import json
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING

from google.protobuf import json_format
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.sdk._logs import LogData, LogRecord
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import TraceFlags

if TYPE_CHECKING:
  from opentelemetry import context as context_api
  from opentelemetry.sdk._logs.export import LogExporter
  from opentelemetry.sdk.resources import Resource

WORKER_THREAD_NAME = "OtelPartialSpanProcessor"


class PartialSpanProcessor(SpanProcessor):

  def __init__(
      self,
      log_exporter: LogExporter,
      heartbeat_interval_millis: int,
      resource: Resource | None = None,
  ) -> None:
    if heartbeat_interval_millis <= 0:
      msg = "heartbeat_interval_ms must be greater than 0"
      raise ValueError(msg)
    self.log_exporter = log_exporter
    self.heartbeat_interval_millis = heartbeat_interval_millis
    self.resource = resource

    self.active_spans = {}
    self.ended_spans = Queue()
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
        self.condition.wait(self.heartbeat_interval_millis / 1000)
        if self.done:
          break

      # Remove ended spans from active spans
      with self.lock:
        while not self.ended_spans.empty():
          span_key, span = self.ended_spans.get()
          if span_key in self.active_spans:
            del self.active_spans[span_key]

      self.heartbeat()

  def heartbeat(self) -> None:
    with self.lock:
      for span in list(self.active_spans.values()):
        attributes = self.get_heartbeat_attributes()
        log_data = self.get_log_data(span, attributes)
        self.log_exporter.export([log_data])

  def on_start(self, span: Span,
      parent_context: context_api.Context | None = None) -> None:
    attributes = self.get_heartbeat_attributes()
    log_data = self.get_log_data(span, attributes)
    self.log_exporter.export([log_data])

    span_key = (span.context.trace_id, span.context.span_id)
    with self.lock:
      self.active_spans[span_key] = span

  def on_end(self, span: ReadableSpan) -> None:
    attributes = get_stop_attributes()
    log_data = self.get_log_data(span, attributes)
    self.log_exporter.export([log_data])

    span_key = (span.context.trace_id, span.context.span_id)
    self.ended_spans.put((span_key, span))

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
          span["traceId"] = hex(span_context.trace_id)[2:]
          span["spanId"] = hex(span_context.span_id)[2:]
          if parent:
            span["parentSpanId"] = hex(parent.span_id)[2:]

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


def get_stop_attributes() -> dict[str, str]:
  return {
    "partial.event": "stop",
    "partial.body.type": "json/v1",
  }
