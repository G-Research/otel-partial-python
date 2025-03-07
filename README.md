# partial-span-processor
OTEL Python SDK extension supporting partial spans

## build
`python -m build`

`pip install dist/partial_span_processor-0.0.x-py3-none-any.whl` to install locally

## publishing
> requirement: `twine`

`python3 -m twine upload --repository pypi dist/* `

## usage
```python
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
  OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs._internal.export import SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from partial_span_processor import PartialSpanProcessor

# Create a TracerProvider
provider = TracerProvider()

# Configure OTLP exporters
span_exporter = OTLPSpanExporter(endpoint="localhost:4317",
                                 insecure=True)  # grpc
log_exporter = OTLPLogExporter(endpoint="http://localhost:4318/v1/logs")  # http

span_processor = PartialSpanProcessor(span_exporter,
                                      SimpleLogRecordProcessor(log_exporter))
provider.add_span_processor(span_processor)

# Set the global TracerProvider
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Start a span (logs heartbeat and stop events)
with tracer.start_as_current_span("partial_span_1"):
  print("partial_span_1 is running")
  with tracer.start_as_current_span("partial_span_2"):
    print("partial_span_2 is running")
    with tracer.start_as_current_span("partial_span_3"):
      print("partial_span_3 is running")
      time.sleep(150)

provider.shutdown()
```

## usage config
* install python version 3.13.1
* create a working dir `test`
* create venv `python3 -m venv venv`
* copy above example script and name it `test.py`
* create `requirements.txt` with content
```
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-logging
partial-span-processor
```
* install requirements `pip install -r requirements.txt`
* run script `python test.py`