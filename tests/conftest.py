from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def openinference_path() -> Path:
    return FIXTURES / "openinference_sample.json"


@pytest.fixture
def otel_genai_path() -> Path:
    return FIXTURES / "otel_genai_sample.json"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
