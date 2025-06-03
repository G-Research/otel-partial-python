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

import datetime
import unittest
from unittest import mock
from unittest.mock import patch

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, SpanContext, TraceFlags

from src.partial_span_processor import PartialSpanProcessor
from tests.partial_span_processor.in_memory_log_exporter import \
  InMemoryLogExporter


class TestPartialSpanProcessor(unittest.TestCase):
  def setUp(self) -> None:
    self.log_exporter = InMemoryLogExporter()
    self.processor = PartialSpanProcessor(
      log_exporter=self.log_exporter,
      heartbeat_interval_millis=1000,
      initial_heartbeat_delay_millis=1000,
      process_interval_millis=1000,
      resource=Resource(attributes={"service.name": "test"}),
    )

  def tearDown(self) -> None:
    self.processor.shutdown()

  @staticmethod
  def create_mock_span(trace_id: int = 1, span_id: int = 1) -> Span:
    tracer_provider = TracerProvider(resource=Resource.create({}))
    tracer = tracer_provider.get_tracer("test_tracer")

    with tracer.start_as_current_span("test_span") as span:
      span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
      )
      span._context = span_context
      return span

  def test_shutdown(self) -> None:
    self.processor.shutdown()

    self.assertTrue(self.processor.done)

  def test_invalid_log_exporter(self):
    with self.assertRaises(ValueError) as context:
      PartialSpanProcessor(
        log_exporter=None,
        heartbeat_interval_millis=1000,
        initial_heartbeat_delay_millis=1000,
        process_interval_millis=1000,
      )
    self.assertEqual(str(context.exception), "log_exporter must not be None")

  def test_invalid_heartbeat_interval(self):
    with self.assertRaises(ValueError) as context:
      PartialSpanProcessor(
        log_exporter=InMemoryLogExporter(),
        heartbeat_interval_millis=0,
        initial_heartbeat_delay_millis=1000,
        process_interval_millis=1000,
      )
    self.assertEqual(str(context.exception),
                     "heartbeat_interval_millis must be greater than 0")

  def test_invalid_initial_heartbeat_delay(self):
    with self.assertRaises(ValueError) as context:
      PartialSpanProcessor(
        log_exporter=InMemoryLogExporter(),
        heartbeat_interval_millis=1000,
        initial_heartbeat_delay_millis=-1,
        process_interval_millis=1000,
      )
    self.assertEqual(str(context.exception),
                     "initial_heartbeat_delay_millis must be greater or equal to 0")

  def test_invalid_process_interval(self):
    with self.assertRaises(ValueError) as context:
      PartialSpanProcessor(
        log_exporter=InMemoryLogExporter(),
        heartbeat_interval_millis=1000,
        initial_heartbeat_delay_millis=1000,
        process_interval_millis=0,
      )
    self.assertEqual(str(context.exception),
                     "process_interval_millis must be greater than 0")

  def test_on_start(self):
    span = TestPartialSpanProcessor.create_mock_span()
    expected_span_id = span.get_span_context().span_id
    now = datetime.datetime.now()
    self.processor.on_start(span)

    self.assertIn(expected_span_id, self.processor.active_spans)
    self.assertIn(expected_span_id,
                  self.processor.delayed_heartbeat_spans_lookup)
    self.assertEqual(self.processor.delayed_heartbeat_spans.qsize(), 1)
    (
      span_id,
      next_heartbeat_time) = self.processor.delayed_heartbeat_spans.get()
    self.assertEqual(expected_span_id, span_id)
    self.assertGreater(next_heartbeat_time, now)
    self.assertEqual(self.log_exporter.get_finished_logs(), ())

  def test_on_end_when_initial_heartbeat_not_sent(self):
    span = TestPartialSpanProcessor.create_mock_span()
    span_id = span.get_span_context().span_id

    self.processor.active_spans[span_id] = span
    self.processor.delayed_heartbeat_spans_lookup.add(span_id)
    self.processor.delayed_heartbeat_spans.put((span_id, unittest.mock.ANY))

    self.processor.on_end(span)

    self.assertNotIn(span_id, self.processor.active_spans)
    self.assertNotIn(span_id,
                     self.processor.delayed_heartbeat_spans_lookup)
    self.assertFalse(self.processor.delayed_heartbeat_spans.empty())
    self.assertEqual(self.log_exporter.get_finished_logs(), ())

  def test_on_end_when_initial_heartbeat_sent(self):
    span = TestPartialSpanProcessor.create_mock_span()
    span_id = span.get_span_context().span_id

    self.processor.active_spans[span_id] = span

    self.processor.on_end(span)

    self.assertNotIn(span_id, self.processor.active_spans)

    logs = self.log_exporter.get_finished_logs()
    self.assertEqual(len(logs), 1)
    self.assertEqual(logs[0].log_record.attributes["partial.event"], "stop")
    self.assertEqual(logs[0].log_record.attributes["partial.body.type"],
                     "json/v1")
    self.assertEqual(logs[0].log_record.resource.attributes["service.name"],
                     "test")

  def test_process_delayed_heartbeat_spans(self):
    span = TestPartialSpanProcessor.create_mock_span()
    span_id = span.get_span_context().span_id

    self.processor.active_spans[span_id] = span
    now = datetime.datetime.now()
    self.processor.delayed_heartbeat_spans.put((span_id, now))
    self.processor.delayed_heartbeat_spans_lookup.add(span_id)

    with patch("datetime.datetime") as mock_datetime:
      mock_datetime.now.return_value = now
      self.processor.process_delayed_heartbeat_spans()

    self.assertNotIn(span_id, self.processor.delayed_heartbeat_spans_lookup)
    self.assertTrue(self.processor.delayed_heartbeat_spans.empty())

    next_heartbeat_time = now + datetime.timedelta(
      milliseconds=self.processor.heartbeat_interval_millis)
    self.assertFalse(self.processor.ready_heartbeat_spans.empty())
    self.assertEqual(self.processor.ready_heartbeat_spans.get(),
                     (span_id, next_heartbeat_time))

  def test_process_ready_heartbeat_spans(self):
    span = TestPartialSpanProcessor.create_mock_span()
    span_id = span.get_span_context().span_id

    self.processor.active_spans[span_id] = span
    now = datetime.datetime.now()
    next_heartbeat_time = now
    self.processor.ready_heartbeat_spans.put((span_id, next_heartbeat_time))

    with patch("datetime.datetime") as mock_datetime:
      mock_datetime.now.return_value = now
      self.processor.process_ready_heartbeat_spans()

    updated_next_heartbeat_time = now + datetime.timedelta(
      milliseconds=self.processor.heartbeat_interval_millis)
    self.assertTrue(self.processor.ready_heartbeat_spans.qsize() == 1)
    self.assertEqual(self.processor.ready_heartbeat_spans.get(),
                     (span_id, updated_next_heartbeat_time))

    logs = self.log_exporter.get_finished_logs()
    self.assertEqual(len(logs), 1)
    self.assertEqual(logs[0].log_record.attributes["partial.event"],
                     "heartbeat")


if __name__ == "__main__":
  unittest.main()
