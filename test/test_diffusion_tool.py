import pytest
from pathlib import Path
from pyneapple.utils.free_diffusion_tool import FreeDiffusionTool

@pytest.fixture
def vector_file_siemens():
    yield Path("test_DiffVector.dvs")
    if Path("test_DiffVector.dvs").is_file():
        Path("test_DiffVector.dvs").unlink()


def test_diffusion_tool_siemens_ve11c(vector_file_siemens):
