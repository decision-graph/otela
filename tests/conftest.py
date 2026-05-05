from pathlib import Path

import pytest

# Synthetic spec fixtures double as the user-facing examples — they live at
# the repo root under `examples/` so the README quickstart can link to them
# at a short, conventional path.
#
# Real-trace fixtures (generated from production SDKs by
# scripts/generate_fixtures.py) stay under tests/fixtures/real/ and are
# used only by tests/test_real_traces.py. Keeping the two trees separate
# means the directory-load tests don't accidentally ingest a real fixture
# and break with off-by-N row counts.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def openinference_path() -> Path:
    return EXAMPLES / "sample.json"


@pytest.fixture
def otel_genai_path() -> Path:
    return EXAMPLES / "sample_genai.json"


@pytest.fixture
def fixtures_dir() -> Path:
    return EXAMPLES
