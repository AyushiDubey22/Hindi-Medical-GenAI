import os
import time
import math
import csv
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from dotenv import load_dotenv

# Try to import the SDK the same way your project does
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------- CONFIG --------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input CSVs (should have columns 'note_id' and 'cleaned_text' or 'text')
DISCHARGE_INPUT = RAW_DIR / "discharge.csv"
RADIOLOGY_INPUT = RAW_DIR / "radiology_sample.csv"

# Output
OUTPUT_CSV = PROCESSED_DIR / "synthetic_all_hindi.csv"
FLAGGED_CSV = PROCESSED_DIR / "synthetic_flagged_for_review.csv"

# Generation parameters
BATCH_SIZE = 10                 # generate this many before checkpointing
SAMPLES_PER_FILE = 500          # if you want to split outputs larger than this
REQUEST_DELAY = 3.0             # seconds between requests (tune per your quota)
MAX_RETRIES = 4
RETRY_BACKOFF = 3.0             # seconds * attempt number

# Validation thresholds
MIN_LENGTH_CHARS = 120
DEVANAGARI_MIN_FRACTION = 0.02  # minimal fraction of Devanagari chars to accept
TRUNCATION_SUFFIXES = ("...", "…")

# Prompt template — includes placeholder for reference_text
GEN_PROMPT_TMPL = """You are an expert medical document writer. Create a brand-new, original HINDI (Devanagari) discharge or radiology note
in Indian hospital style. Do NOT translate the reference — use it only for inspiration. The generated note MUST be purely synthetic and non-identifiable.
Required: include sections for मरीज़ की जानकारी, मुख्य शिकायत, वर्तमान बीमारी का इतिहास, शारीरिक जांच, निदान, उपचार योजना.
Write ONLY in Devanagari Hindi. Use entirely generic names, ages, and locations (e.g., 'रोगी', '50 वर्ष', 'दिल्ली').

Reference (for inspiration only — do NOT copy):
{reference_text}
"""

# -------- SDK SETUP --------
if not API_KEY:
    raise SystemExit("ERROR: No GOOGLE_API_KEY found in .env")

if genai is None:
    raise SystemExit("ERROR: google.generativeai SDK not importable. Install google-generativeai")

genai.configure(api_key=API_KEY)

# Try to get a model wrapper if available
model_wrapper = None
try:
    model_wrapper = genai.GenerativeModel(MODEL_NAME)
except Exception:
    # fallback is fine; we'll call top-level genai.generate if needed
    model_wrapper = None
    print(f"Warning: could not instantiate GenerativeModel({MODEL_NAME}) — will use fallback generate.")

# -------- UTILITIES --------
def contains_devanagari(s: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in str(s))

def devanagari_fraction(s: str) -> float:
    s = str(s)
    if not s:
        return 0.0
    dev = sum(1 for ch in s if '\u0900' <= ch <= '\u097F')
    return dev / max(1, len(s))

def is_truncated(s: str) -> bool:
    s = (s or "").strip()
    return any(s.endswith(suf) for suf in TRUNCATION_SUFFIXES) or len(s) < MIN_LENGTH_CHARS

def simple_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# -------- GENERATION & RETRY --------
def _extract_text_from_response(resp):
    """
    Try multiple strategies to extract user-visible text from different SDK response shapes.
    Returns string (possibly empty).
    """
    # 1) quick attribute
    try:
        txt = getattr(resp, "text", None)
        if txt:
            return str(txt)
    except Exception:
        pass

    # 2) try common fields: candidates / outputs
    try:
        # candidates (list of candidate objects)
        cands = getattr(resp, "candidates", None)
        if cands:
            # pick first candidate
            cand = cands[0]
            # candidate may have .content which is list of parts
            content = getattr(cand, "content", None)
            if content:
                # if list-like
                if isinstance(content, (list, tuple)) and len(content) > 0:
                    first = content[0]
                    t = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                    if t:
                        return str(t)
                else:
                    t = getattr(content, "text", None) or (content.get("text") if isinstance(content, dict) else None)
                    if t:
                        return str(t)
            # fallback to candidate.text or string
            t = getattr(cand, "text", None) or (cand.get("text") if isinstance(cand, dict) else None)
            if t:
                return str(t)
    except Exception:
        pass

    try:
        outs = getattr(resp, "outputs", None)
        if outs:
            out = outs[0]
            t = getattr(out, "text", None) or (out.get("text") if isinstance(out, dict) else None)
            if t:
                return str(t)
    except Exception:
        pass

    # 3) if resp is dict-like with nested structure
    try:
        if isinstance(resp, dict):
            # deep search for 'text' keys
            def find_text(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == "text" and isinstance(v, str):
                            return v
                        res = find_text(v)
                        if res:
                            return res
                elif isinstance(d, list):
                    for el in d:
                        res = find_text(el)
                        if res:
                            return res
                return None
            t = find_text(resp)
            if t:
                return str(t)
    except Exception:
        pass

    # 4) last resort: stringify object
    try:
        return str(resp)
    except Exception:
        return ""

def call_generate(prompt: str):
    """
    Robust generation call with retries and careful parsing of different SDK response shapes.
    Returns the generated text (or raises the last exception).
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if model_wrapper is not None:
                resp = model_wrapper.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.8, max_output_tokens=1500)
                )
            else:
                if hasattr(genai, "generate"):
                    resp = genai.generate(model=MODEL_NAME, prompt=prompt)
                elif hasattr(genai, "generate_text"):
                    resp = genai.generate_text(model=MODEL_NAME, prompt=prompt)
                else:
                    raise RuntimeError("No generate method available in SDK")

            text = _extract_text_from_response(resp)
            text = (text or "").strip()

            # If empty, attempt to see if finish_reason or candidate info exists for logging
            if not text:
                # try to inspect finish_reason for better logging (non-fatal)
                fr = None
                try:
                    if hasattr(resp, "candidates") and getattr(resp, "candidates"):
                        fr = getattr(resp.candidates[0], "finish_reason", None)
                    elif hasattr(resp, "outputs") and getattr(resp, "outputs"):
                        fr = getattr(resp.outputs[0], "finish_reason", None)
                except Exception:
                    fr = None
                raise RuntimeError(f"No generated text found (finish_reason={fr})")

            return text

        except Exception as e:
            last_exc = e
            print(f"Generate error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                sleep_t = RETRY_BACKOFF * attempt
                print(f" Sleeping {sleep_t:.1f}s then retrying...")
                time.sleep(sleep_t)
                continue
            # final failure: raise so caller flags this sample
            raise last_exc

# -------- CHECKPOINTING --------
def save_checkpoint(rows, checkpoint_name):
    path = CHECKPOINT_DIR / checkpoint_name
    cols = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved checkpoint: {path}")
    return path

def load_existing_outputs(path):
    if not path.exists():
        return []
    return pd.read_csv(path, dtype=str).to_dict(orient="records")

# -------- MAIN FLOW --------
def generate_from_input(input_csv: Path, note_type_label: str, target_samples: int):
    # load input
    if not input_csv.exists():
        print(f"Input not found: {input_csv}; skipping.")
        return []

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    # We'll sample or cycle if needed
    rows_out = []
    existing = load_existing_outputs(OUTPUT_CSV) if OUTPUT_CSV.exists() else []
    existing_hashes = set(r.get("reference_hash") for r in existing if r.get("reference_hash"))
    start_index = 0

    # iterate until we get target_samples
    idx = 0
    generated = 0
    while generated < target_samples:
        src_row = df.iloc[idx % len(df)]
        reference_text = src_row.get("cleaned_text") or src_row.get("text") or ""
        if not reference_text:
            idx += 1
            continue
        ref_hash = simple_hash(reference_text)
        # skip if already generated for this reference
        if ref_hash in existing_hashes:
            idx += 1
            continue

        prompt = GEN_PROMPT_TMPL.format(reference_text=reference_text[:1200])
        try:
            t0 = time.time()
            out_text = call_generate(prompt)
            duration = time.time() - t0
        except Exception as e:
            # Do not crash whole run — flag this sample and continue
            print("Generation failed after retries:", e)
            out_text = ""
            duration = 0.0

        # basic validation
        dev_frac = devanagari_fraction(out_text)
        truncated = is_truncated(out_text)
        has_dev = dev_frac >= DEVANAGARI_MIN_FRACTION and contains_devanagari(out_text)
        status = "success" if out_text and has_dev and not truncated else "flagged"

        row = {
            "synthetic_id": f"{note_type_label}_{generated:06d}",
            "reference_note_id": src_row.get("note_id", f"src_{idx}"),
            "note_type": note_type_label,
            "reference_preview": reference_text[:250],
            "reference_hash": ref_hash,
            "synthetic_hindi_text": out_text,
            "status": status,
            "devanagari_fraction": round(dev_frac, 4),
            "truncated": truncated,
            "generated_length": len(out_text or ""),
            "generation_time_sec": round(duration, 2)
        }
        rows_out.append(row)
        generated += 1
        idx += 1

        # print progress
        print(f"[{generated}/{target_samples}] {row['synthetic_id']} status={status} len={row['generated_length']} dev_frac={row['devanagari_fraction']} time={row['generation_time_sec']}s")

        # checkpoint every BATCH_SIZE
        if generated % BATCH_SIZE == 0 or generated == target_samples:
            # append to master output CSV
            append_rows_to_output(rows_out)
            # also save flagged rows to separate file
            flagged = [r for r in rows_out if r["status"] != "success"]
            if flagged:
                append_rows_to_flagged(flagged)
            rows_out = []  # reset batch
        time.sleep(REQUEST_DELAY)

    return True

def append_rows_to_output(rows):
    # if file exists, append; otherwise create with headers
    if not rows:
        return
    file_exists = OUTPUT_CSV.exists()
    df = pd.DataFrame(rows)
    if file_exists:
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(OUTPUT_CSV, mode="w", header=True, index=False, encoding="utf-8-sig")
    print(f"Appended {len(rows)} rows to {OUTPUT_CSV}")

def append_rows_to_flagged(rows):
    if not rows: return
    file_exists = FLAGGED_CSV.exists()
    df = pd.DataFrame(rows)
    if file_exists:
        df.to_csv(FLAGGED_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(FLAGGED_CSV, mode="w", header=True, index=False, encoding="utf-8-sig")
    print(f"Appended {len(rows)} flagged rows to {FLAGGED_CSV}")

# -------- VALIDATION & BIAS ANALYSIS --------
def run_validation_and_bias_report(output_csv=OUTPUT_CSV, flagged_csv=FLAGGED_CSV):
    if not output_csv.exists():
        print("No output file to analyze.")
        return

    df = pd.read_csv(output_csv, dtype=str).fillna("")
    print(f"\nLoaded {len(df)} generated rows for analysis.")

    # Basic quality metrics
    total = len(df)
    success = (df["status"] == "success").sum()
    avg_len = df["generated_length"].astype(int).mean()
    non_dev = (df["devanagari_fraction"].astype(float) < DEVANAGARI_MIN_FRACTION).sum()
    truncated = df["truncated"].astype(str).str.lower().isin(["true","1"]).sum()
    flagged = (df["status"] != "success").sum()

    print("\nQUALITY METRICS")
    print("----------------")
    print(f"Total generated: {total}")
    print(f"Success: {success} ({success/total*100:.1f}%)")
    print(f"Flagged: {flagged} ({flagged/total*100:.1f}%)")
    print(f"Avg length: {avg_len:.0f} chars")
    print(f"No/low Devanagari (dev_frac < {DEVANAGARI_MIN_FRACTION}): {non_dev}")
    print(f"Truncated / short: {truncated}")

    # Duplicate detection (simple hash of generated text)
    df["gen_hash"] = df["synthetic_hindi_text"].astype(str).apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:10])
    dup_counts = df["gen_hash"].value_counts()
    duplicates = dup_counts[dup_counts > 1].sum()
    print(f"Duplicate outputs (exact hash dup): {duplicates}")

    # Extract simple demographics from text via heuristics (gender, age)
    gender_counts = Counter()
    ages = []
    for s in df["synthetic_hindi_text"].astype(str):
        low = s
        if "पुरुष" in low:
            gender_counts["male"] += 1
        if "महिला" in low or "स्त्री" in low:
            gender_counts["female"] += 1
        # capture age patterns: 'उम्र: 52' or '52 वर्ष'
        import re
        m = re.search(r'(\d{2,3})\s*(वर्ष|साल)', s)
        if m:
            ages.append(int(m.group(1)))

    print("\nDEMOGRAPHICS (heuristic)")
    print("------------------------")
    print("Gender counts (heuristic):", gender_counts)
    if ages:
        import statistics
        print(f"Age samples: n={len(ages)} mean={statistics.mean(ages):.1f} median={statistics.median(ages):.1f} min={min(ages)} max={max(ages)}")
    else:
        print("No ages detected via heuristic.")

    # Name frequency (naive: look for patterns like 'नाम: X' or bullets)
    name_freq = Counter()
    import re
    for s in df["synthetic_hindi_text"].astype(str):
        m = re.search(r'नाम[:\s]*([^\n\*\-]{2,40})', s)
        if m:
            name = m.group(1).strip()
            name_freq[name] += 1
    print("\nTop names (sample):", name_freq.most_common(10))

    # Save a short report
    report_path = PROCESSED_DIR / "generation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Generation Report\n")
        f.write("=================\n")
        f.write(f"Total generated: {total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Flagged: {flagged}\n")
        f.write(f"Avg length: {avg_len:.0f}\n")
        f.write(f"Duplicates (exact hash): {duplicates}\n")
        f.write(f"Gender heuristic: {dict(gender_counts)}\n")
        if ages:
            f.write(f"Age mean: {statistics.mean(ages):.1f}\n")
        f.write("\nNote: This is a heuristic report. Manual review required for clinical safety.\n")
    print(f"\nSaved report to {report_path}")

# -------- ENTRY POINT --------
def main():
    # parameters you can edit
    target_per_type = 200   # set 100, 200, 500, 1000 as you like (watch cost)
    print("Starting large synthetic generation run")
    print(f"Model: {MODEL_NAME}  Target per type: {target_per_type}")

    # Generate for discharge and radiology each (or adapt)
    print("\nGenerating for discharge summaries...")
    generate_from_input(DISCHARGE_INPUT, note_type_label="DS", target_samples=target_per_type)

    print("\nGenerating for radiology reports...")
    generate_from_input(RADIOLOGY_INPUT, note_type_label="RR", target_samples=target_per_type)

    print("\nGeneration finished. Running validation & bias report...")
    run_validation_and_bias_report()

if __name__ == "__main__":
    main()
