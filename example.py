from time import sleep

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs._internal.export import SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider

from partial_span_processor import PartialSpanProcessor

# Configure OTLP exporters
log_exporter = OTLPLogExporter(endpoint="http://localhost:4318/v1/logs")  # http

span_processor = PartialSpanProcessor(
  SimpleLogRecordProcessor(log_exporter), log_emit_interval=5000)

# Create a TracerProvider
provider = TracerProvider()
provider.add_span_processor(span_processor)

# Set the global TracerProvider
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Start a span (logs heartbeat and stop events)
with tracer.start_as_current_span("partial_span_1"):
  print("partial_span_1 is running")
  sleep(10)
