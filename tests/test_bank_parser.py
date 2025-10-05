# tests/test_bank_parser.py
import pandas as pd
from pathlib import Path
import importlib.util

# -------- helpers --------
def import_parser(parser_path: Path):
    spec = importlib.util.spec_from_file_location("user_parser", parser_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# -------- pytest test --------
# tests/test_bank_parser.py (pytest path-based version)
def test_parse_equals_csv(paths):
    mod = import_parser(paths["PARSER"])
    got = mod.parse(str(paths["PDF"]))
    exp = pd.read_csv(paths["CSV"])

    # Optional: sanity messages (do NOT mutate `got`/`exp`)
    assert list(got.columns) == list(exp.columns), "Column order/name mismatch"
    assert got.index.equals(exp.index), "Index mismatch"

    # Strict contract check
    assert got.equals(exp), "DataFrame content/type mismatch"


# -------- optional: run as a plain script --------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="icici",
                    help="Bank folder under data/ (e.g., icici, sbi)")
    ap.add_argument("--pdf-name", default=None,
                    help="PDF filename in data/<bank>. Defaults to '<bank> sample.pdf'")
    args = ap.parse_args()

    bank = args.bank.lower()
    pdf_name = args.pdf_name or f"{bank} sample.pdf"

    base = Path("data") / bank
    csv_path = base / "result.csv"
    pdf_path = base / pdf_name
    parser_path = Path("custom_parser") / f"{bank}_parser.py"

    # Minimal assertion-style run
    assert parser_path.exists(), f"Parser not found: {parser_path}"
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    mod = import_parser(parser_path)
    assert hasattr(mod, "parse"), "parse(pdf_path) missing in parser module"
    got = mod.parse(str(pdf_path))
    exp = pd.read_csv(csv_path)
    assert list(got.columns) == list(exp.columns), "Column order/name mismatch"
    assert got.index.equals(exp.index), "Index mismatch"
    ok = got.equals(exp)

    print(f"[{bank}] equals={ok} rows={len(got)}")
    sys.exit(0 if ok else 1)
