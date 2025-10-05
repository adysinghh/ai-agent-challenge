# tests/conftest.py
from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption("--bank", action="store", default="icici",
                     help="Bank folder under data/ (e.g., icici, sbi)")
    parser.addoption("--pdf-name", action="store", default=None,
                     help="PDF filename inside data/<bank> (defaults to '<bank> sample.pdf')")

@pytest.fixture
def paths(request):
    bank = request.config.getoption("--bank").lower()
    pdf_name = request.config.getoption("--pdf-name") or f"{bank} sample.pdf"

    base = Path("data") / bank
    csv = base / "result.csv"
    pdf = base / pdf_name
    parser_path = Path("custom_parser") / f"{bank}_parser.py"

    return {
        "BANK": bank,
        "CSV": csv,
        "PDF": pdf,
        "PARSER": parser_path,
    }
