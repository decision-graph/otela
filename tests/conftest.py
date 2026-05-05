from pathlib import Path

import pytest

# Synthetic spec fixtures live under fixtures/synthetic/. Real-trace
# fixtures (generated from production SDKs) live under fixtures/real/
# and are used only by tests/test_real_traces.py — keeping them in a
# separate tree means the directory-load tests below don't accidentally
# ingest them and break with off-by-N row counts.
SYNTHETIC = Path(__file__).parent / "fixtures" / "synthetic"


@pytest.fixture
def openinference_path() -> Path:
    return SYNTHETIC / "openinference_sample.json"


@pytest.fixture
def otel_genai_path() -> Path:
    return SYNTHETIC / "otel_genai_sample.json"


@pytest.fixture
def fixtures_dir() -> Path:
    return SYNTHETIC
