# partial-span-processor
OTEL Python SDK extension supporting partial spans

## build
`python -m build`

`pip install dist/partial_span_processor-0.0.x-py3-none-any.whl` to install locally

## usage
Check example.py script in the root of the repo.

## usage config
* install python version >= 3.8
* create a working dir `test`
* create venv `python3 -m venv venv`
* copy example script `example.py`
* create `requirements.txt` with content
```
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-logging
partial-span-processor
```
* install requirements `pip install -r requirements.txt`
* run script `python example.py`

## publishing
Python package is published to PyPI via the `release.yml` GitHub Action workflow (approval required) following [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) pattern. 
The workflow is triggered when a new tag is pushed to the repository. Only tags with the format `vX.Y.Z` will trigger the workflow. It's the responsibility of the approver to check that the tag points to a commit on the `master` branch and that its name matches the version in `pyproject.toml`.

Checklist:
- Bump the version in `pyproject.toml` via pull request
- Create a tag on `master` with the format `vX.Y.Z` and push it
- Review the workflow approval request - the tag should point to a commit on the `master` branch!
- Success

Link to PyPI: https://pypi.org/project/partial-span-processor/