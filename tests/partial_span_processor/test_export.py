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

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging import WARNING
from platform import python_implementation, system
from unittest import mock

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.sdk import trace
from opentelemetry.sdk._logs._internal.export import SimpleLogRecordProcessor
from opentelemetry.sdk.environment_variables import (
  OTEL_BSP_EXPORT_TIMEOUT,
  OTEL_BSP_MAX_EXPORT_BATCH_SIZE,
  OTEL_BSP_MAX_QUEUE_SIZE,
  OTEL_BSP_SCHEDULE_DELAY,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace.export import logger
from pytest import mark

from src.partial_span_processor import PartialSpanProcessor
from tests.partial_span_processor.concurrency_test import ConcurrencyTestBase
from tests.partial_span_processor.in_memory_log_exporter import \
  InMemoryLogExporter


class MySpanExporter(export.SpanExporter):
  """Very simple span exporter used for testing."""

  def __init__(
      self,
      destination,
      max_export_batch_size=None,
      export_timeout_millis=0.0,
      export_event: threading.Event = None,
  ):
    self.destination = destination
    self.max_export_batch_size = max_export_batch_size
    self.is_shutdown = False
    self.export_timeout = export_timeout_millis / 1e3
    self.export_event = export_event

  def export(self, spans: trace.Span) -> export.SpanExportResult:
    if (
        self.max_export_batch_size is not None
        and len(spans) > self.max_export_batch_size
    ):
      raise ValueError("Batch is too big")
    time.sleep(self.export_timeout)
    self.destination.extend(span.name for span in spans)
    if self.export_event:
      self.export_event.set()
    return export.SpanExportResult.SUCCESS

  def shutdown(self):
    self.is_shutdown = True


def _create_start_and_end_span(name, span_processor, resource):
  span = trace._Span(
    name,
    trace_api.SpanContext(
      0xDEADBEEF,
      0xDEADBEEF,
      is_remote=False,
      trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
    ),
    span_processor=span_processor,
    resource=resource,
  )
  span.start()
  span.end()


class TestPartialSpanProcessor(ConcurrencyTestBase):
  @mock.patch.dict(
    "os.environ",
    {
      OTEL_BSP_MAX_QUEUE_SIZE: "10",
      OTEL_BSP_SCHEDULE_DELAY: "2",
      OTEL_BSP_MAX_EXPORT_BATCH_SIZE: "3",
      OTEL_BSP_EXPORT_TIMEOUT: "4",
    },
  )
  def test_args_env_var(self):
    partial_span_processor = PartialSpanProcessor(
      MySpanExporter(destination=[]),
      SimpleLogRecordProcessor(InMemoryLogExporter())
    )

    self.assertEqual(partial_span_processor.max_queue_size, 10)
    self.assertEqual(partial_span_processor.schedule_delay_millis, 2)
    self.assertEqual(partial_span_processor.max_export_batch_size, 3)
    self.assertEqual(partial_span_processor.export_timeout_millis, 4)

  def test_args_env_var_defaults(self):
    partial_span_processor = PartialSpanProcessor(
      MySpanExporter(destination=[]),
      SimpleLogRecordProcessor(InMemoryLogExporter())
    )

    self.assertEqual(partial_span_processor.max_queue_size, 2048)
    self.assertEqual(partial_span_processor.schedule_delay_millis, 5000)
    self.assertEqual(partial_span_processor.max_export_batch_size, 512)
    self.assertEqual(partial_span_processor.export_timeout_millis, 30000)

  @mock.patch.dict(
    "os.environ",
    {
      OTEL_BSP_MAX_QUEUE_SIZE: "a",
      OTEL_BSP_SCHEDULE_DELAY: " ",
      OTEL_BSP_MAX_EXPORT_BATCH_SIZE: "One",
      OTEL_BSP_EXPORT_TIMEOUT: "@",
    },
  )
  def test_args_env_var_value_error(self):
    logger.disabled = True
    batch_span_processor = PartialSpanProcessor(
      MySpanExporter(destination=[]),
      SimpleLogRecordProcessor(InMemoryLogExporter())
    )
    logger.disabled = False

    self.assertEqual(batch_span_processor.max_queue_size, 2048)
    self.assertEqual(batch_span_processor.schedule_delay_millis, 5000)
    self.assertEqual(batch_span_processor.max_export_batch_size, 512)
    self.assertEqual(batch_span_processor.export_timeout_millis, 30000)

  def test_on_start_accepts_parent_context(self):
    # pylint: disable=no-self-use
    my_exporter = MySpanExporter(destination=[])
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = mock.Mock(
      wraps=PartialSpanProcessor(my_exporter, log_processor)
    )
    tracer_provider = trace.TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    tracer = tracer_provider.get_tracer(__name__)

    context = Context()
    span = tracer.start_span("foo", context=context)

    span_processor.on_start.assert_called_once_with(
      span, parent_context=context
    )

  def test_heartbeat_attributes(self):
    # pylint: disable=no-self-use
    my_exporter = MySpanExporter(destination=[])
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    schedule_delay_millis = 1234
    span_processor = mock.Mock(
      wraps=PartialSpanProcessor(my_exporter, log_processor,
                                 schedule_delay_millis=schedule_delay_millis)
    )
    tracer_provider = trace.TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    tracer = tracer_provider.get_tracer(__name__)

    context = Context()
    span = tracer.start_span("foo", context=context)

    span_processor.on_start.assert_called_once_with(
      span, parent_context=context
    )

    expected_values = {
      'partial.event': 'heartbeat',
      'partial.frequency': str(schedule_delay_millis) + "ms",
    }

    logs = log_exporter.get_finished_logs()
    self.assertEqual(len(logs), 1)

    for log in logs:
      for key, value in log.log_record.attributes.items():
        self.assertEqual(value, expected_values[key])

  def test_shutdown(self):
    spans_names_list = []

    my_exporter = MySpanExporter(destination=spans_names_list)
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(my_exporter, log_processor)

    span_names = ["xxx", "bar", "foo"]

    resource = Resource.create({})
    for name in span_names:
      _create_start_and_end_span(name, span_processor, resource)

    span_processor.shutdown()
    self.assertTrue(my_exporter.is_shutdown)

    # check that spans are exported without an explicitly call to
    # force_flush()
    self.assertListEqual(span_names, spans_names_list)

  def test_stop_attributes(self):
    spans_names_list = []

    my_exporter = MySpanExporter(destination=spans_names_list)
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(my_exporter, log_processor,
                                          schedule_delay_millis=10000)

    span_names = ["xxx"]

    resource = Resource.create({})
    for name in span_names:
      _create_start_and_end_span(name, span_processor, resource)

    span_processor.shutdown()
    self.assertTrue(my_exporter.is_shutdown)

    # check that spans are exported without an explicitly call to
    # force_flush()
    self.assertListEqual(span_names, spans_names_list)

    expected_values = {
      'partial.event': 'stop',
    }

    logs = log_exporter.get_finished_logs()
    for log in logs:
      print(log.log_record.attributes)
    self.assertEqual(len(logs), 3)

    for key, value in logs[2].log_record.attributes.items():
      self.assertEqual(value, expected_values[key])

  def test_flush(self):
    spans_names_list = []

    my_exporter = MySpanExporter(destination=spans_names_list)
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(my_exporter, log_processor)

    span_names0 = ["xxx", "bar", "foo"]
    span_names1 = ["yyy", "baz", "fox"]

    resource = Resource.create({})
    for name in span_names0:
      _create_start_and_end_span(name, span_processor, resource)

    self.assertTrue(span_processor.force_flush())
    self.assertListEqual(span_names0, spans_names_list)

    # create some more spans to check that span processor still works
    for name in span_names1:
      _create_start_and_end_span(name, span_processor, resource)

    self.assertTrue(span_processor.force_flush())
    self.assertListEqual(span_names0 + span_names1, spans_names_list)

    span_processor.shutdown()

  def test_flush_empty(self):
    spans_names_list = []

    my_exporter = MySpanExporter(destination=spans_names_list)
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(my_exporter, log_processor)

    self.assertTrue(span_processor.force_flush())

  def test_flush_from_multiple_threads(self):
    num_threads = 50
    num_spans = 10

    span_list = []

    my_exporter = MySpanExporter(destination=span_list)
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter, max_queue_size=512, max_export_batch_size=128,
      log_processor=log_processor
    )

    resource = Resource.create({})

    def create_spans_and_flush(tno: int):
      for span_idx in range(num_spans):
        _create_start_and_end_span(
          f"Span {tno}-{span_idx}", span_processor, resource
        )
      self.assertTrue(span_processor.force_flush())

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      future_list = []
      for thread_no in range(num_threads):
        future = executor.submit(create_spans_and_flush, thread_no)
        future_list.append(future)

      executor.shutdown()

    self.assertEqual(num_threads * num_spans, len(span_list))

  def test_flush_timeout(self):
    spans_names_list = []

    my_exporter = MySpanExporter(
      destination=spans_names_list, export_timeout_millis=500
    )
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(my_exporter, log_processor)

    resource = Resource.create({})
    _create_start_and_end_span("foo", span_processor, resource)

    # check that the timeout is not meet
    with self.assertLogs(level=WARNING):
      self.assertFalse(span_processor.force_flush(100))
    span_processor.shutdown()

  def test_batch_span_processor_lossless(self):
    """Test that no spans are lost when sending max_queue_size spans"""
    spans_names_list = []

    my_exporter = MySpanExporter(
      destination=spans_names_list, max_export_batch_size=128
    )
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter, max_queue_size=512, max_export_batch_size=128,
      log_processor=log_processor
    )

    resource = Resource.create({})
    for _ in range(512):
      _create_start_and_end_span("foo", span_processor, resource)

    time.sleep(1)
    self.assertTrue(span_processor.force_flush())
    self.assertEqual(len(spans_names_list), 512)
    span_processor.shutdown()

  def test_batch_span_processor_many_spans(self):
    """Test that no spans are lost when sending many spans"""
    spans_names_list = []

    my_exporter = MySpanExporter(
      destination=spans_names_list, max_export_batch_size=128
    )
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter,
      max_queue_size=256,
      max_export_batch_size=64,
      schedule_delay_millis=100,
      log_processor=log_processor
    )

    resource = Resource.create({})
    for _ in range(4):
      for _ in range(256):
        _create_start_and_end_span("foo", span_processor, resource)

      time.sleep(0.1)  # give some time for the exporter to upload spans

    self.assertTrue(span_processor.force_flush())
    self.assertEqual(len(spans_names_list), 1024)
    span_processor.shutdown()

  def test_batch_span_processor_not_sampled(self):
    tracer_provider = trace.TracerProvider(
      sampler=trace.sampling.ALWAYS_OFF
    )
    tracer = tracer_provider.get_tracer(__name__)
    spans_names_list = []

    my_exporter = MySpanExporter(
      destination=spans_names_list, max_export_batch_size=128
    )
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter,
      max_queue_size=256,
      max_export_batch_size=64,
      schedule_delay_millis=100,
      log_processor=log_processor
    )
    tracer_provider.add_span_processor(span_processor)
    with tracer.start_as_current_span("foo"):
      pass
    time.sleep(0.05)  # give some time for the exporter to upload spans

    self.assertTrue(span_processor.force_flush())
    self.assertEqual(len(spans_names_list), 0)
    span_processor.shutdown()

  def _check_fork_trace(self, exporter, expected):
    time.sleep(0.5)  # give some time for the exporter to upload spans
    spans = exporter.get_finished_spans()
    for span in spans:
      self.assertIn(span.name, expected)

  # FIXME does not work on macos
  # @unittest.skipUnless(
  #   hasattr(os, "fork"),
  #   "needs *nix",
  # )
  # def test_batch_span_processor_fork(self):
  #   # pylint: disable=invalid-name
  #   tracer_provider = trace.TracerProvider()
  #   tracer = tracer_provider.get_tracer(__name__)
  #
  #   exporter = InMemorySpanExporter()
  #   span_processor = export.BatchSpanProcessor(
  #     exporter,
  #     max_queue_size=256,
  #     max_export_batch_size=64,
  #     schedule_delay_millis=10,
  #   )
  #   tracer_provider.add_span_processor(span_processor)
  #   with tracer.start_as_current_span("foo"):
  #     pass
  #   time.sleep(0.5)  # give some time for the exporter to upload spans
  #
  #   self.assertTrue(span_processor.force_flush())
  #   self.assertEqual(len(exporter.get_finished_spans()), 1)
  #   exporter.clear()
  #
  #   def child(conn):
  #     def _target():
  #       with tracer.start_as_current_span("span") as s:
  #         s.set_attribute("i", "1")
  #         with tracer.start_as_current_span("temp"):
  #           pass
  #
  #     self.run_with_many_threads(_target, 100)
  #
  #     time.sleep(0.5)
  #
  #     spans = exporter.get_finished_spans()
  #     conn.send(len(spans) == 200)
  #     conn.close()
  #
  #   parent_conn, child_conn = multiprocessing.Pipe()
  #   p = multiprocessing.Process(target=child, args=(child_conn,))
  #   p.start()
  #   self.assertTrue(parent_conn.recv())
  #   p.join()
  #
  #   span_processor.shutdown()

  @mark.skipif(
    python_implementation() == "PyPy" or system() == "Windows",
    reason="This test randomly fails with huge delta in Windows or PyPy",
  )
  def test_batch_span_processor_scheduled_delay(self):
    """Test that spans are exported each schedule_delay_millis"""
    spans_names_list = []

    export_event = threading.Event()
    my_exporter = MySpanExporter(
      destination=spans_names_list, export_event=export_event
    )
    start_time = time.time()
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter,
      schedule_delay_millis=500,
      log_processor=log_processor
    )

    # create single span
    resource = Resource.create({})
    _create_start_and_end_span("foo", span_processor, resource)

    self.assertTrue(export_event.wait(2))
    export_time = time.time()
    self.assertEqual(len(spans_names_list), 1)
    self.assertAlmostEqual((export_time - start_time) * 1e3, 500, delta=25)

    span_processor.shutdown()

  @mark.skipif(
    python_implementation() == "PyPy" and system() == "Windows",
    reason="This test randomly fails in Windows with PyPy",
  )
  def test_batch_span_processor_reset_timeout(self):
    """Test that the scheduled timeout is reset on cycles without spans"""
    spans_names_list = []

    export_event = threading.Event()
    my_exporter = MySpanExporter(
      destination=spans_names_list,
      export_event=export_event,
      export_timeout_millis=50,
    )

    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    span_processor = PartialSpanProcessor(
      my_exporter,
      schedule_delay_millis=50,
      log_processor=log_processor
    )

    with mock.patch.object(span_processor.condition, "wait") as mock_wait:
      resource = Resource.create({})
      _create_start_and_end_span("foo", span_processor, resource)
      self.assertTrue(export_event.wait(2))

      # give some time for exporter to loop
      # since wait is mocked it should return immediately
      time.sleep(0.1)
      mock_wait_calls = list(mock_wait.mock_calls)

      # find the index of the call that processed the singular span
      for idx, wait_call in enumerate(mock_wait_calls):
        _, args, __ = wait_call
        if args[0] <= 0:
          after_calls = mock_wait_calls[idx + 1:]
          break

      self.assertTrue(
        all(args[0] >= 0.05 for _, args, __ in after_calls)
      )

    span_processor.shutdown()

  def test_batch_span_processor_parameters(self):
    log_exporter = InMemoryLogExporter()
    log_processor = SimpleLogRecordProcessor(log_exporter)
    # zero max_queue_size
    self.assertRaises(
      ValueError, PartialSpanProcessor, None, max_queue_size=0,
      log_processor=log_processor
    )

    # negative max_queue_size
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      max_queue_size=-500,
      log_processor=log_processor
    )

    # zero schedule_delay_millis
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      schedule_delay_millis=0,
      log_processor=log_processor
    )

    # negative schedule_delay_millis
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      schedule_delay_millis=-500,
      log_processor=log_processor
    )

    # zero max_export_batch_size
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      max_export_batch_size=0,
      log_processor=log_processor
    )

    # negative max_export_batch_size
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      max_export_batch_size=-500,
      log_processor=log_processor
    )

    # max_export_batch_size > max_queue_size:
    self.assertRaises(
      ValueError,
      PartialSpanProcessor,
      None,
      max_queue_size=256,
      max_export_batch_size=512,
      log_processor=log_processor
    )
