import os, re, json, argparse, shutil, traceback, importlib.util
from pathlib import Path
from typing import List, Tuple
import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# ------------------------------ Paths & Constants ------------------------------
TRACE = Path("trace"); TRACE.mkdir(exist_ok=True, parents=True)
CUSTOM_DIR = Path("custom_parser"); CUSTOM_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = Path("data")

REQUIRED_COLS = ["Date","Description","Debit Amt","Credit Amt","Balance"]
BAN_IMPORTS = [
    r"\bsubprocess\b", r"\bos\.system\b", r"\brequests\b", r"\burllib\b",
    r"\bhttpx\b", r"\bsocket\b", r"\bos\.popen\b"
]

# ------------------------------ LLM Provider Wrapper ---------------------------
def llm_call(prompt:str, temperature:float=0.2) -> str:
    """
    Provider-agnostic LLM call. Select provider with env LLM_PROVIDER in {gemini, groq, openai}.
    Set keys: GOOGLE_API_KEY or GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY.
    Optional models: GEMINI_MODEL, GROQ_MODEL, OPENAI_MODEL.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY")
        genai.configure(api_key=key)
        model_id = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_id)
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return (resp.text or "").strip()
    elif provider == "groq":
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key: raise RuntimeError("Missing GROQ_API_KEY")
        client = Groq(api_key=key)
        model_id = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        out = client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            messages=[{"role":"user","content":prompt}],
        )
        return (out.choices[0].message.content or "").strip()
    elif provider == "openai":
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("Missing OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        out = client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            messages=[{"role":"user","content":prompt}],
        )
        return (out.choices[0].message.content or "").strip()
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")

# ------------------------------ Trace Helpers ----------------------------------
def twrite(name:str, content):
    TRACE.mkdir(exist_ok=True, parents=True)
    p = TRACE/name
    if isinstance(content, str):
        p.write_text(content, encoding="utf-8")
    else:
        p.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return p

def import_parser(bank:str):
    parser_file = CUSTOM_DIR/f"{bank}_parser.py"
    spec = importlib.util.spec_from_file_location("user_parser", parser_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ------------------------------ Plan + Plan-SIM (CodeSIM-lite) -----------------
def sample_text_rows(pdf_path:str, n:int=5) -> List[str]:
    if not pdfplumber:
        return [f"SAMPLE_ROW_{i}" for i in range(n)]
    rows: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for line in txt.splitlines():
                s = line.strip()
                if s:
                    rows.append(s)
            if len(rows) >= 200:  # enough
                break
    # Prefer lines that look like transactions
    tx = [L for L in rows if re.match(r"^\d{2}[-/]\d{2}[-/]\d{4}\b", L)]
    if len(tx) >= n: return tx[:n]
    return rows[:n] if rows else [f"SAMPLE_ROW_{i}" for i in range(n)]

def make_plan(bank:str, pdf_path:str) -> dict:
    plan = {
        "date_regexes": [r"^\d{2}-\d{2}-\d{4}\b", r"^\d{2}/\d{2}/\d{4}\b"],
        "amount_clean": {"strip_commas": True, "paren_negatives": True, "currency_symbols": ["₹", ","]},
        "debit_credit": "only_one_non_empty",
        "header_filters": ["Date Description","Karbon","ChatGPT"],
    }
    twrite("plan.json", {"bank":bank, "plan":plan})
    return plan

def plan_simulate(plan:dict, samples:List[str]) -> Tuple[bool, dict]:
    results = []
    for s in samples:
        date_ok = any(re.match(rx, s) for rx in plan["date_regexes"])
        nums = re.findall(r"[-]?\d[\d,]*\.?\d*", s)
        # simple polarity clues
        s_low = s.lower()
        credit_hint = any(k in s_low for k in [" credit", " cr ", "cr.", "salary", "deposit", "interest", "transfer from"])
        debit_hint  = any(k in s_low for k in [" debit", " dr ", "dr.", "upi", "pos", "atm", "charges"])
        mapping_ok = (len(nums) >= 2)
        results.append({
            "line": s[:160],
            "date_match": bool(date_ok),
            "numbers_found": len(nums),
            "credit_hint": credit_hint,
            "debit_hint": debit_hint,
            "debit_credit_rule_ok": mapping_ok  # we only validate existence here
        })
    ok = all(r["date_match"] and r["numbers_found"] >= 2 for r in results)
    payload = {"ok": ok, "checked": len(samples), "results": results}
    twrite("sim_plan.txt", payload)
    return ok, payload


# ------------------------------ Test & Debug-SIM --------------------------------
def strict_compare(bank: str, pdf_path: Path, csv_path: Path):
    exp = pd.read_csv(csv_path)
    parser = import_parser(bank)
    got = parser.parse(str(pdf_path))
    try:
        got = got[exp.columns.tolist()].reset_index(drop=True)
    except Exception:
        pass
    exp = exp.reset_index(drop=True)
    equal = got.equals(exp)
    reason = "" if equal else "DataFrame.equals returned False or column order mismatch."
    return equal, got, exp, reason

def debug_sim(got:pd.DataFrame, exp:pd.DataFrame, max_rows:int=3) -> dict:
    deltas = []
    rows = min(max_rows, max(len(got), len(exp)))
    for i in range(rows):
        g = (got.iloc[i].to_dict() if i < len(got) else None)
        e = (exp.iloc[i].to_dict() if i < len(exp) else None)
        if g != e:
            deltas.append({"row": i, "got": g, "exp": e})
    summary = {"deltas": deltas, "got_shape": list(got.shape), "exp_shape": list(exp.shape), "columns_equal": list(got.columns)==list(exp.columns)}
    twrite("sim_debug.txt", summary)
    return summary

# ------------------------------ Code Sanitization --------------------------------
def strip_code_fences(text:str) -> str:
    text = text.strip()
    # pull code between ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S|re.I)
    if m: return m.group(1).strip()
    # remove stray fences
    return text.strip("` \n\r\t")

def ensure_imports(code:str) -> str:
    required = ["import pandas as pd", "import pdfplumber", "import re"]
    for imp in required:
        if imp not in code:
            code = imp + "\n" + code
    return code

def ban_unsafe(code: str) -> str:
    # Comment out any line containing a banned pattern
    lines = code.splitlines()
    banned = [re.compile(p) for p in BAN_IMPORTS]
    safe = []
    for ln in lines:
        if any(p.search(ln) for p in banned):
            safe.append("# BANNED: " + ln)
        else:
            safe.append(ln)
    return "\n".join(safe)


def ensure_signature(code:str) -> str:
    if "def parse(" not in code:
        # add minimal stub to satisfy loader; LLM should supply anyway
        code += "\n\ndef parse(pdf_path: str):\n    import pandas as pd\n    return pd.DataFrame(columns=%r)\n" % REQUIRED_COLS
    return code

def fix_str_on_numeric(code: str) -> str:
    # Make .str calls safe on known numeric columns by forcing astype(str)
    pattern = re.compile(
        r"(\b\w+\s*\[\s*['\"](?:Debit Amt|Credit Amt|Credit Amt|Balance)['\"]\s*\])\.str\b"
    )
    # Note: 'Credit Amt' appears twice above? Replace with (Debit Amt|Credit Amt|Balance)
    pattern = re.compile(
        r"(\b\w+\s*\[\s*['\"](?:Debit Amt|Credit Amt|Balance)['\"]\s*\])\.str\b"
    )
    return pattern.sub(r"\1.astype(str).str", code)

def sanitize(code:str) -> str:
    code = strip_code_fences(code)
    code = ensure_imports(code)
    code = fix_str_on_numeric(code)  # <-- NEW: auto-fix fragile .str usage
    code = ensure_signature(code)
    code = ban_unsafe(code)
    return code


# ------------------------------ Prompts ------------------------------------------
GEN_TEMPLATE = """You are an expert Python developer. Write a COMPLETE module that implements:

def parse(pdf_path: str) -> pandas.DataFrame

Constraints:
- Use pdfplumber for PDF reading (tables and/or text).
- Return EXACT columns (and order): {cols}
- Multi-page PDFs: iterate all pages; skip header/footer lines (e.g., {header_filters})
- Date parsing: handle formats DD-MM-YYYY or DD/MM/YYYY; keep Date as string "DD-MM-YYYY".
- Numbers: strip commas, support negatives including parentheses; no currency symbols.
- Exactly one of 'Debit Amt' or 'Credit Amt' is non-empty per row; empty cells must be NaN (via pandas.to_numeric(..., errors='coerce')), not empty string '' or 0.
- Do NOT import network or shell libraries.
- Be robust to table extraction failures: fall back to text + regex if needed.
- Return only valid rows; avoid duplicate headers.
- Convert numeric columns with pandas.to_numeric(..., errors='coerce') so blanks become NaN.
- **Reset the DataFrame index before returning:** `df = df.reset_index(drop=True)`.
- Never call .str on possibly numeric Series. If you need string ops, use .astype(str).str... or a helper:
    def clean_amount(x):
        import re, pandas as pd
        if x is None: return pd.NA
        if isinstance(x, (int, float)): return x
        s = str(x).strip().replace(',', '')
        s = re.sub(r'^\((.*)\)$', r'-\1', s)  # (123.45) -> -123.45
        return pd.to_numeric(s, errors='coerce')
- Apply clean_amount to ['Debit Amt','Credit Amt','Balance'] and keep them numeric (float).


IMPORTANT: Output ONLY Python code of the module, no backticks, no prose.
"""

CRITIC_TEMPLATE = """You will FIX a Python module that defines parse(pdf_path)->DataFrame.

Columns must be exactly: {cols}
Failure summary:
- columns_equal={columns_equal}, expected shape={exp_shape}, got shape={got_shape}
- First deltas (row-level mismatches) follow as JSON:
{deltas_json}

Fixes required (keep minimal changes):
- Ensure exact column order and row count equal to expected.
- Ensure Date formatted 'DD-MM-YYYY' strings (not datetime, not YYYY-MM-DD).
- Ensure numeric cleaning (commas, parentheses).
- Ensure only one of Debit/Credit populated; the other must be NaN (use pandas.to_numeric(..., errors='coerce')).
- Remove header/footer lines and duplicate headers across pages.
- Convert numeric columns with pandas.to_numeric(..., errors='coerce') so blanks become NaN to match the CSV.
- **Ensure index is RangeIndex(0..N-1): call `df = df.reset_index(drop=True)` before return.**
- Avoid AttributeError from .str: use .astype(str).str... or clean_amount(x) for numeric columns.



OUTPUT: The FULL corrected module code ONLY (no backticks, no commentary).
"""

# ------------------------------ Generation & Repair ------------------------------
def write_code(bank:str, code:str):
    p = CUSTOM_DIR / f"{bank}_parser.py"
    p.write_text(code, encoding="utf-8")
    return p

# generate_candidates
def generate_candidates(bank: str, cols: List[str], plan: dict, pdf_path: Path, k: int = 3):
    samples = sample_text_rows(str(pdf_path), n=4)
    plan_bits = "Header filters to skip: " + ", ".join(plan["header_filters"]) + \
                ". Sample PDF lines:\n" + "\n".join(f"- {s}" for s in samples)
    base = GEN_TEMPLATE.format(cols=cols, header_filters=plan["header_filters"])
    prompt = base + "\n\n" + plan_bits

    refx = load_reflections(3)
    if refx:
        prompt += "\n\nReflexion (follow these rules from previous failures):\n" + refx

    temps = [0.1, 0.3, 0.6][:max(1, k)]
    out = []
    for t in temps:
        raw = llm_call(prompt, temperature=t)
        code = sanitize(raw)
        write_code(bank, code)  # write to standard path for convenience

        cand_path = CUSTOM_DIR / f"{bank}_parser_t{str(t).replace('.','_')}.py"
        cand_path.write_text(code, encoding="utf-8")
        out.append((cand_path, code))
    return out



def write_observe(equal: bool, got: pd.DataFrame, exp: pd.DataFrame, note: str = ""):
    payload = {
        "equal": equal,
        "columns_equal": list(got.columns) == list(exp.columns),
        "got_shape": list(got.shape),
        "exp_shape": list(exp.shape),
        "note": note,
    }
    twrite("observe.json", payload)


def eval_candidate(bank: str, cand_path: Path, pdf_path: Path, csv_path: Path) -> Tuple[int, dict]:
    std = CUSTOM_DIR / f"{bank}_parser.py"
    std.write_text(cand_path.read_text(encoding="utf-8"), encoding="utf-8")
    equal, got, exp, _ = strict_compare(bank, pdf_path, csv_path)
    if equal:
        return 100, {"equal": True}
    score = 0
    if list(got.columns) == list(exp.columns): score += 30
    score += max(0, 40 - abs(len(got) - len(exp)))
    try:
        g2 = got.fillna("").astype(str); e2 = exp.fillna("").astype(str)
        common = (g2.reset_index(drop=True).head(10) == e2.reset_index(drop=True).head(10)).sum().sum()
        score += min(30, int(common/5))
    except Exception:
        pass
    return score, {"equal": False, "got_shape": list(got.shape), "exp_shape": list(exp.shape)}


def best_of_k(bank: str, cols: List[str], plan: dict, pdf_path: Path, csv_path: Path, k: int) -> Tuple[str, Path]:    
    cands = generate_candidates(bank, cols, plan, pdf_path, k)
    scores_log, best_score, best = [], -1, None
    for cand_path, code in cands:
        (CUSTOM_DIR / f"{bank}_parser.py").write_text(code, encoding="utf-8")
        score, meta = eval_candidate(bank, cand_path, pdf_path, csv_path)  # csv_path must be in scope (see main)
        scores_log.append({"file": cand_path.name, "score": score, **meta})
        if score > best_score:
            best_score, best = score, (code, cand_path)
    twrite("candidate_scores.json", scores_log)
    final_code, final_path = best
    write_code(bank, final_code)
    twrite("candidate_selected.txt", final_path.name)
    twrite("final_parser.py", final_code)
    return final_code, final_path



def critic_patch(bank:str, prev_code:str, dbg:dict, cols:List[str]) -> str:
    prompt = CRITIC_TEMPLATE.format(
        cols=cols,
        columns_equal=dbg.get("columns_equal"),
        exp_shape=dbg.get("exp_shape"),
        got_shape=dbg.get("got_shape"),
        deltas_json=json.dumps(dbg.get("deltas", [])[:3], indent=2)
    ) + "\n\nCurrent module:\n```python\n" + prev_code + "\n```"

    # reflection context
    refx = load_reflections(3)
    if refx:
        prompt = ("Reflexion rules to obey:\n" + refx + "\n\n") + prompt

    raw = llm_call(prompt, temperature=0.2)
    return sanitize(raw)

# ------------------------------ Reflexion (tiny) --------------------------------
def add_reflection(rule:str):
    f = TRACE/"reflections.jsonl"
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rule": rule}) + "\n")

def load_reflections(k:int=3) -> str:
    f = TRACE/"reflections.jsonl"
    if not f.exists(): return ""
    lines = f.read_text(encoding="utf-8").strip().splitlines()[-k:]
    try:
        rules = [json.loads(x).get("rule","") for x in lines if x.strip()]
        rules = [r for r in rules if r]
    except Exception:
        rules = []
    return "\n".join(f"- {r}" for r in rules[:k])


# ------------------------------ Main Loop ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="bank folder name label (used only for file naming)")
    ap.add_argument("--pdf", required=True, help="path to input statement PDF")
    ap.add_argument("--csv", required=True, help="path to ground-truth CSV to match")
    ap.add_argument("--trace_all", action="store_true", help="also dump sim_debug on GREEN")
    ap.add_argument("--p", type=int, default=1, help="max plan revisions (Plan-SIM)")
    ap.add_argument("--d", type=int, default=2, help="max debug trials (Debug-SIM)")
    ap.add_argument("--best_of", type=int, default=3, help="num candidates on attempt #1")
    args = ap.parse_args()

    bank = args.target.lower()
    pdf_path = Path(args.pdf)
    csv_path = Path(args.csv)
    if not pdf_path.exists():
        raise FileNotFoundError(f"--pdf not found: {pdf_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"--csv not found: {csv_path}")

    # clear trace
    if TRACE.exists():
        shutil.rmtree(TRACE)
    TRACE.mkdir(parents=True, exist_ok=True)


    # PLAN → PLAN-SIM
    print("PLAN: drafting rules")
    plan = make_plan(bank, str(pdf_path))
    samples = sample_text_rows(str(pdf_path), n=5)
    ok, sim = plan_simulate(plan, samples)
    print(f"PLAN-SIM: ok={ok} checked={sim['checked']}")
    if not ok and args.p > 0:
        print("PLAN-SIM: retrying plan once")
        ok, sim = plan_simulate(plan, samples)
        print(f"PLAN-SIM(retry): ok={ok}")

    # EXPECTED SCHEMA
    exp = pd.read_csv(csv_path)
    cols = exp.columns.tolist()

    # ATTEMPT 1 → best-of-k
    print(f"ACT: generating best-of-{args.best_of} candidates")
    code, cand_path = best_of_k(bank, cols, plan, pdf_path, csv_path, args.best_of)
    print(f"ACT: selected {cand_path.name}")

    # TEST
    equal, got, exp, reason = strict_compare(bank, pdf_path, csv_path)
    print(f"OBSERVE: {'GREEN' if equal else 'RED'}")
    if equal:
        print("SUCCESS: strict equality achieved.")
        return

    # DEBUG-SIM & CRITIC→PATCH loops
    dbg = debug_sim(got, exp)
    attempts = 0
    while not equal and attempts < args.d:
        attempts += 1
        print(f"DEBUG-SIM: attempt {attempts} → CRITIC→PATCH")
        patched = critic_patch(bank, code, dbg, cols)
        write_code(bank, patched)
        equal, got, exp, reason = strict_compare(bank, pdf_path, csv_path)
        print(f"OBSERVE: {'GREEN' if equal else 'RED'}")
        write_observe(equal, got, exp, note="post-best-of-k")

        if equal:
            # Optionally force a debug trace for your demo (zero or tiny deltas)
            if args.trace_all:
                dbg = debug_sim(got, exp)  # will likely write empty deltas
            print("SUCCESS: strict equality achieved.")
            return

        # refresh debug info and add a reflection rule
        dbg = debug_sim(got, exp)
        add_reflection("ensure per-page header rows are dropped; format Date as DD-MM-YYYY; empty debit/credit should be NaN")
        code = patched

    if not equal:
        print("FAILED: after debug trials; see trace/ for details")
        # leave best attempt on disk

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        twrite("fatal_error.txt", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise