import unittest
import warnings

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanContext, TraceFlags

from src.partial_span_processor import PartialSpanProcessor
from tests.partial_span_processor.in_memory_log_exporter import InMemoryLogExporter


class TestLogRecordDeprecation(unittest.TestCase):
    def setUp(self):
        self.processor = PartialSpanProcessor(
            log_exporter=InMemoryLogExporter(),
            heartbeat_interval_millis=1000,
            initial_heartbeat_delay_millis=1000,
            process_interval_millis=1000,
            resource=Resource(attributes={"service.name": "test"}),
        )

    def tearDown(self):
        self.processor.shutdown()

    def test_get_log_data_produces_no_deprecation_warnings(self):
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span("test_span") as span:
            span._context = SpanContext(
                trace_id=0x1234567890abcdef1234567890abcdef,
                span_id=0x1234567890abcdef,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.processor.get_log_data(span, {"partial.event": "heartbeat"})

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(deprecation_warnings), 0)
