from time import sleep

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
  OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from partial_span_processor import PartialSpanProcessor

# Configure OTLP exporters
log_exporter = OTLPLogExporter(endpoint="http://localhost:4318/v1/logs")  # http
span_exporter = OTLPSpanExporter(endpoint="localhost:4317",
                                 insecure=True)  # grpc

# Configure span processors
partial_span_processor = PartialSpanProcessor(log_exporter, 5000)
batch_span_processor = BatchSpanProcessor(span_exporter)

# Create a TracerProvider
provider = TracerProvider()
# Order in which processors are added is important
provider.add_span_processor(partial_span_processor)
provider.add_span_processor(batch_span_processor)

# Set the global TracerProvider
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Start a span (logs heartbeat and stop events)
with tracer.start_as_current_span("partial_span_1"):
  print("partial_span_1 is running")
  sleep(10)
