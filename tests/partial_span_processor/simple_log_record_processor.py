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

from opentelemetry.sdk._logs import LogRecordProcessor, LogData

class SimpleLogRecordProcessor(LogRecordProcessor):
    def __init__(self, exporter):
        self.exporter = exporter

    def emit(self, log_data: LogData) -> None:
        # Export the log data using the provided exporter
        self.exporter.export([log_data])

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # No-op for this simple implementation
        return True

    def shutdown(self) -> None:
        # Shutdown the exporter
        self.exporter.shutdown()