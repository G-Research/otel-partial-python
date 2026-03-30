import time
import unittest
import warnings

from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.sdk._logs import LogRecord
from opentelemetry.trace import TraceFlags


class TestLogRecordDeprecation(unittest.TestCase):
    def test_trace_id_span_id_trace_flags_params_are_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LogRecord(
                timestamp=time.time_ns(),
                observed_timestamp=time.time_ns(),
                trace_id=0x1234567890abcdef1234567890abcdef,
                span_id=0x1234567890abcdef,
                trace_flags=TraceFlags().get_default(),
                severity_text="INFO",
                severity_number=SeverityNumber.INFO,
                body="test",
            )
        self.assertEqual(len(w), 1)
        self.assertIn("deprecated", str(w[0].message).lower())
        self.assertIn("context", str(w[0].message).lower())