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

import unittest
from time import sleep

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, SpanContext, TraceFlags

from src.partial_span_processor import PartialSpanProcessor
from tests.partial_span_processor.in_memory_log_exporter import InMemoryLogExporter


class TestPartialSpanProcessor(unittest.TestCase):
  def setUp(self) -> None:
    # Set up an in-memory log exporter and processor
    self.log_exporter = InMemoryLogExporter()
    self.processor = PartialSpanProcessor(
      log_exporter=self.log_exporter,
      heartbeat_interval_millis=1000,  # 1 second
      heartbeat_delay_millis=0, # no initial delay
      resource=Resource(attributes={"service.name": "test"}),
    )

  def tearDown(self) -> None:
    # Shut down the processor
    self.processor.shutdown()

  def create_mock_span(self, trace_id: int = 1, span_id: int = 1) -> Span:
    # Create a mock tracer
    tracer_provider = TracerProvider(resource=Resource.create({}))
    tracer = tracer_provider.get_tracer("test_tracer")

    # Start a span using the tracer
    with tracer.start_as_current_span("test_span") as span:
      # Set the span context manually for testing purposes
      span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
      )
      span._context = span_context  # Modify the span's context for testing
      return span

  def test_on_start(self) -> None:
    # Test the on_start method
    span = self.create_mock_span()
    self.processor.on_start(span)

    # Verify the span is added to active_spans
    span_key = (span.context.trace_id, span.context.span_id)
    assert span_key in self.processor.active_spans

    # Verify a log is emitted
    logs = self.log_exporter.get_finished_logs()
    assert len(logs) == 1
    assert logs[0].log_record.attributes["partial.event"] == "heartbeat"
    assert logs[0].log_record.resource.attributes["service.name"] == "test"

  def test_on_end(self) -> None:
    # Test the on_end method
    span = self.create_mock_span()
    self.processor.on_start(span)
    self.processor.on_end(span)

    assert len(self.processor.active_spans) == 0

    # Verify a log is emitted
    logs = self.log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert logs[1].log_record.attributes["partial.event"] == "stop"
    assert logs[0].log_record.resource.attributes["service.name"] == "test"

  def test_heartbeat_without_delay(self) -> None:
    # Test the heartbeat method
    span = self.create_mock_span()
    self.processor.on_start(span)

    # Wait for the heartbeat interval
    sleep(1.5)
    logs = self.log_exporter.get_finished_logs()

    # Verify heartbeat logs are emitted
    assert len(logs) >= 2
    assert logs[1].log_record.attributes["partial.event"] == "heartbeat"
    assert logs[0].log_record.resource.attributes["service.name"] == "test"

  def test_heartbeat_with_delay(self) -> None:
    processor = PartialSpanProcessor(
      log_exporter=self.log_exporter,
      heartbeat_interval_millis=1000,
      heartbeat_delay_millis=1500,
      resource=Resource(attributes={"service.name": "test"}),
    )

    span = self.create_mock_span()
    processor.on_start(span)

    sleep(1.5)
    logs = self.log_exporter.get_finished_logs()
    assert len(logs) == 1

    sleep(1)

    logs = self.log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert logs[1].log_record.attributes["partial.event"] == "heartbeat"
    assert logs[0].log_record.resource.attributes["service.name"] == "test"

  def test_shutdown(self) -> None:
    # Test the shutdown method
    self.processor.shutdown()

    # Verify the worker thread is stopped
    assert self.processor.done

  def test_worker_thread(self) -> None:
    # Test the worker thread processes ended spans
    span = self.create_mock_span()
    self.processor.on_start(span)
    self.processor.on_end(span)

    # Wait for the worker thread to process the ended span
    sleep(1.5)

    # Verify the span is removed from active_spans
    span_key = (span.context.trace_id, span.context.span_id)
    assert span_key not in self.processor.active_spans


if __name__ == "__main__":
  unittest.main()
