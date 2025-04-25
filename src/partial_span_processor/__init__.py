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

import base64
import threading
import time
from queue import Queue
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.sdk._logs import LogData, LogRecord
from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.trace import (
  SpanProcessor,
  Span,
  ReadableSpan
)
from opentelemetry.trace import TraceFlags

WORKER_THREAD_NAME = "OtelPartialSpanProcessor"


class PartialSpanProcessor(SpanProcessor):

  def __init__(
      self,
      log_exporter: LogExporter,
      heartbeat_interval_millis: int
  ):
    if heartbeat_interval_millis <= 0:
      raise ValueError("heartbeat_interval_ms must be greater than 0")
    self.log_exporter = log_exporter
    self.heartbeat_interval_millis = heartbeat_interval_millis

    self.active_spans = {}
    self.ended_spans = Queue()
    self.lock = threading.Lock()

    self.done = False
    self.condition = threading.Condition(threading.Lock())
    self.worker_thread = threading.Thread(
      name=WORKER_THREAD_NAME, target=self.worker, daemon=True
    )
    self.worker_thread.start()

  def worker(self):
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

  def heartbeat(self):
    with self.lock:
      for span_key, span in list(self.active_spans.items()):
        attributes = self.get_heartbeat_attributes()
        log_data = get_log_data(span, attributes)
        self.log_exporter.export([log_data])

  def on_start(self, span: "Span",
      parent_context: Optional[context_api.Context] = None) -> None:
    attributes = self.get_heartbeat_attributes()
    log_data = get_log_data(span, attributes)
    self.log_exporter.export([log_data])

    span_key = (span.context.trace_id, span.context.span_id)
    with self.lock:
      self.active_spans[span_key] = span

  def on_end(self, span: ReadableSpan) -> None:
    attributes = get_stop_attributes()
    log_data = get_log_data(span, attributes)
    self.log_exporter.export([log_data])

    span_key = (span.context.trace_id, span.context.span_id)
    self.ended_spans.put((span_key, span))

  def shutdown(self) -> None:
    # signal the worker thread to finish and then wait for it
    self.done = True
    with self.condition:
      self.condition.notify_all()
    self.worker_thread.join()

  def get_heartbeat_attributes(self):
    return {
      "partial.event": "heartbeat",
      "partial.frequency": str(self.heartbeat_interval_millis) + "ms",
    }


def get_stop_attributes():
  return {
    "partial.event": "stop",
  }


def get_log_data(span, attributes):
  span_context = Span.get_span_context(span)

  enc_spans = encode_spans([span]).resource_spans
  traces_data = trace_pb2.TracesData()
  traces_data.resource_spans.extend(enc_spans)
  serialized_traces_data = traces_data.SerializeToString()

  log_record = LogRecord(
    timestamp=time.time_ns(),
    observed_timestamp=time.time_ns(),
    trace_id=span_context.trace_id,
    span_id=span_context.span_id,
    trace_flags=TraceFlags().get_default(),
    severity_text="INFO",
    severity_number=SeverityNumber.INFO,
    body=base64.b64encode(serialized_traces_data).decode('utf-8'),
    attributes=attributes,
  )
  log_data = LogData(
    log_record=log_record, instrumentation_scope=span.instrumentation_scope
  )
  return log_data
