"""
Simple Hindi Medical Synthetic Data Generator - FIXED VERSION
NO Object-Oriented Programming - Just simple functions!
"""

import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# ========================================
# STEP 1: Setup
# ========================================

print("\n" + "="*70)
print(" Setting up Gemini API...")
print("="*70)

# Ensure we resolve paths relative to script
SCRIPT_DIR = Path(__file__).resolve().parent        # src folder
PROJECT_ROOT = SCRIPT_DIR.parent                    # project root (Dissertation)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Load environment
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash')  # use env override or a newer default

if not api_key:
    print(" ERROR: No GOOGLE_API_KEY found in .env file!")
    print("\n To fix:")
    print("   1. Create a .env file in your project root")
    print("   2. Add this line: GOOGLE_API_KEY=your_key_here")
    print("   3. Get your key from: https://aistudio.google.com/app/apikey")
    sys.exit(1)

# Configure Gemini
genai.configure(api_key=api_key)

def try_list_models():
    """
    Attempt to list models available to your API key using a couple of possible SDK methods.
    Returns a list of dict-like entries or strings describing models.
    """
    models = []
    # Try older google.generativeai.list_models if it exists
    try:
        if hasattr(genai, "list_models"):
            print(" Calling genai.list_models() ...")
            for m in genai.list_models():
                # m may be a simple string or an object/dict
                try:
                    name = getattr(m, "name", None) or m.get("name", None) or str(m)
                except Exception:
                    name = str(m)
                models.append(name)
            return models
    except Exception as e:
        print("  list_models() failed:", str(e)[:200])

    # Try genai.models.list (some SDK versions)
    try:
        if hasattr(genai, "models") and hasattr(genai.models, "list"):
            print(" Calling genai.models.list() ...")
            for m in genai.models.list():
                try:
                    name = getattr(m, "name", None) or m.get("name", None) or str(m)
                except Exception:
                    name = str(m)
                models.append(name)
            return models
    except Exception as e:
        print("  genai.models.list() failed:", str(e)[:200])

    # As a last resort, return empty
    return models


# Try to instantiate or otherwise validate the chosen model
print(f"\nAttempting to use model: {MODEL_NAME}")
model = None
try:
    # Many versions of google.generativeai expose a GenerativeModel class
    # which can be used to call generate_content.
    model = genai.GenerativeModel(MODEL_NAME)
    print(" Gemini API configured successfully!")
    print(" Using model:", MODEL_NAME)
except Exception as e:
    # If the explicit model wrapper fails (404 or similar), try listing available models
    print(" Error setting up model:", str(e)[:300])
    print("\n Attempting to list available models for your API key...\n")
    available = try_list_models()
    if available:
        print(" Available models (sample):")
        for m in available[:40]:
            print("  -", m)
        print("\n Suggestion: set environment variable MODEL_NAME to one of the model names above")
        print(" Example (Windows PowerShell):")
        print(r'  $env:MODEL_NAME="gemini-2.5-flash"')
    else:
        print(" Could not retrieve model list automatically. This may be due to SDK differences.")
        print(" Check your SDK version and API key access / billing, or try upgrading the SDK:")
        print("   pip install --upgrade google-generativeai")
    sys.exit(1)


# ========================================
# STEP 2: Define Generation Prompt
# ========================================

GENERATION_PROMPT = """You are a medical documentation expert creating SYNTHETIC Hindi medical records for an Indian healthcare application.

**CRITICAL INSTRUCTION:** Do NOT translate the text below. Instead, USE IT ONLY AS INSPIRATION to create a COMPLETELY NEW, ORIGINAL Hindi medical record with DIFFERENT patient details.

**Your Task:**
Generate a BRAND NEW, ORIGINAL Hindi medical record that:
1. Uses SIMILAR medical scenario (similar age group, condition type)
2. Has COMPLETELY DIFFERENT specific details (different patient name, different exact symptoms, different lab values)
3. Follows standard Indian hospital documentation format
4. Uses natural Hindi medical terminology (Hindi terms + English medical terms in Devanagari when needed)
5. Includes realistic Indian context (Indian names like राज शर्मा, प्रिया पटेल, common local practices)
6. Maintains medical accuracy and completeness

**Required sections in Hindi:**
- मरीज़ की जानकारी (Patient Information) - Include name, age, gender
- मुख्य शिकायत (Chief Complaint)
- वर्तमान बीमारी का इतिहास (History of Present Illness)
- शारीरिक जांच (Physical Examination)
- निदान (Diagnosis)
- उपचार योजना (Treatment Plan)

**IMPORTANT:** Write ONLY in Hindi (Devanagari script). Create a COMPLETE, ORIGINAL medical note. Do NOT translate - CREATE something new!

Reference (for context / inspiration only):
{reference_text}
"""

# ========================================
# STEP 3: Function to Generate ONE Text
# ========================================

def generate_one_synthetic_text(reference_text, attempt_num=1):
    """
    Generate ONE synthetic Hindi text from reference
    
    Input: English reference text
    Output: New Hindi synthetic text
    """
    
    # Check if text is empty
    if not reference_text or str(reference_text).strip() == '':
        return '', 'empty_input'
    
    try:
        # Prepare the prompt
        short_reference = str(reference_text)[:1200]  # Use first 1200 chars
        prompt = GENERATION_PROMPT.format(reference_text=short_reference)
        
        # Generate with Gemini
        print(f"Generating (attempt {attempt_num})...", end=' ', flush=True)
        
        # There are multiple possible call shapes depending on SDK version;
        # prefer model.generate_content if model is a GenerativeModel wrapper,
        # otherwise try genai.generate or genai.generate_text style calls.
        response = None
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,  # More creative
                    max_output_tokens=1500,  # Longer output
                )
            )
            hindi_text = getattr(response, "text", None) or str(response)
        except Exception as e_inner:
            # Fallback: try a top-level generate() if available
            print("\n primary generate failed, trying fallback generate() ...", end=' ', flush=True)
            try:
                if hasattr(genai, "generate") or hasattr(genai, "generate_text"):
                    # try generate
                    if hasattr(genai, "generate"):
                        resp = genai.generate(model=MODEL_NAME, prompt=prompt)
                        hindi_text = resp.text if hasattr(resp, "text") else str(resp)
                    else:
                        resp = genai.generate_text(model=MODEL_NAME, prompt=prompt)
                        hindi_text = resp.text if hasattr(resp, "text") else str(resp)
                else:
                    raise RuntimeError("No fallback generate method available in SDK")
            except Exception as e_fallback:
                print("\n Fallback generate failed:", str(e_fallback)[:300])
                raise e_inner  # raise original to be caught by outer except
        
        hindi_text = hindi_text.strip() if hindi_text else ''
        
        # Validation checks
        if not hindi_text:
            return '', 'empty_response'
        
        if len(hindi_text) < 100:
            return '', 'output_too_short'
        
        # Check if it contains Hindi (Devanagari characters)
        has_hindi = any('\u0900' <= char <= '\u097F' for char in hindi_text)
        if not has_hindi:
            print(" No Hindi detected, retrying...")
            if attempt_num < 2:
                time.sleep(2)
                return generate_one_synthetic_text(reference_text, attempt_num + 1)
            return '', 'no_hindi_found'
        
        # Check it's not just translating (look for common Hindi medical terms)
        hindi_medical_terms = ['मरीज़', 'रोगी', 'जांच', 'उपचार', 'निदान', 'शिकायत']
        has_medical_terms = any(term in hindi_text for term in hindi_medical_terms)
        
        if not has_medical_terms:
            print(" Quality check failed, retrying...")
            if attempt_num < 2:
                time.sleep(2)
                return generate_one_synthetic_text(reference_text, attempt_num + 1)
        
        return hindi_text, 'success'
        
    except Exception as e:
        error_msg = str(e)
        print(f" Error: {error_msg[:200]}...")
        return '', f'error: {error_msg}'


# ========================================
# STEP 4: Process Multiple Texts
# ========================================

def generate_synthetic_dataset(input_file, output_file, num_samples=10):
    """
    Generate synthetic Hindi dataset from CSV file
    
    Input: Path to CSV with 'cleaned_text' column
    Output: New CSV with synthetic Hindi texts
    """
    
    print(f"\n{'='*70}")
    print(f" Processing: {os.path.basename(input_file)}")
    print(f"{'='*70}")
    
    # Resolve input path relative to project
    input_path = (PROJECT_ROOT / input_file).resolve() if not Path(input_file).is_absolute() else Path(input_file)
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_path)
        print(f" Loaded {len(df)} records from {input_path}")
    except FileNotFoundError:
        print(f" Error: File not found - {input_path}")
        return None
    except Exception as e:
        print(f" Error loading file: {e}")
        return None
    
    # Take only first num_samples rows
    df_subset = df.head(num_samples)
    
    print(f"\n Generating {num_samples} synthetic Hindi texts...")
    print(f"  Estimated time: ~{num_samples * 3:.0f} seconds\n")
    
    # Store results
    results = []
    success_count = 0
    
    # Process each text one by one
    for index in range(len(df_subset)):
        print(f"\n[{index + 1}/{num_samples}] ", end='')
        
        # Get the row
        row = df_subset.iloc[index]
        reference_text = row.get('cleaned_text', '')
        
        if not reference_text:
            # Try 'text' column if cleaned_text doesn't exist
            reference_text = row.get('text', '')
        
        # Generate synthetic text
        start_time = time.time()
        hindi_text, status = generate_one_synthetic_text(reference_text)
        duration = time.time() - start_time
        
        # Store result
        results.append({
            'synthetic_id': f'SYN_{index:04d}',
            'reference_note_id': row.get('note_id', f'NOTE_{index}'),
            'note_type': row.get('note_type', 'unknown'),
            'original_preview': str(reference_text)[:200] + '...',
            'synthetic_hindi_text': hindi_text,
            'status': status,
            'original_length': len(str(reference_text)),
            'generated_length': len(hindi_text),
            'generation_time_sec': round(duration, 2)
        })
        
        if status == 'success':
            success_count += 1
            print(f" Success! ({len(hindi_text)} chars, {duration:.1f}s)")
        else:
            print(f" Failed: {status}")
        
        # Wait to avoid rate limits (Gemini: ~60 requests/min)
        time.sleep(2)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_path = (PROJECT_ROOT / output_file).resolve() if not Path(output_file).is_absolute() else Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    try:
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n Saved to: {output_path}")
    except Exception as e:
        print(f"\n Error saving file: {e}")
        return None
    
    # Print statistics
    print(f"\n{'='*70}")
    print(" Statistics")
    print(f"{'='*70}")
    print(f" Success: {success_count}/{num_samples} ({success_count/num_samples*100:.0f}%)")
    
    if success_count > 0:
        avg_length = results_df[results_df['status'] == 'success']['generated_length'].mean()
        print(f" Average length: {avg_length:.0f} characters")
        
        # Show a sample
        sample = results_df[results_df['status'] == 'success'].iloc[0]['synthetic_hindi_text']
        print(f"\n Sample output (first 200 chars):")
        print(f"   {sample[:200]}...")
    
    return results_df


def main():
    """
    Main function - this runs when you execute the script
    """
    
    print("\n" + "="*70)
    print(" Hindi Synthetic Medical Data Generator")
    print("="*70)
    print(" This creates BRAND NEW Hindi medical records")
    print("   (Not translations - completely synthetic data!)")
    print("="*70)
    
    # Create processed output folder (resolved)
    processed_folder = PROCESSED_DIR
    processed_folder.mkdir(parents=True, exist_ok=True)
    print(f" Using processed output folder: {processed_folder}")
    
    # ===== Process Discharge Summaries =====
    print("\n" + "="*70)
    print("1️  DISCHARGE SUMMARIES")
    print("="*70)
    
    discharge_input = "data/raw/discharge.csv"
    discharge_output = "data/processed/synthetic_discharge_hindi_FIXED.csv"
    
    if (PROJECT_ROOT / discharge_input).exists():
        discharge_results = generate_synthetic_dataset(
            input_file=discharge_input,
            output_file=discharge_output,
            num_samples=5  # Start with 5 to test
        )
    else:
        print(f"  Skipping - file not found: {(PROJECT_ROOT / discharge_input).resolve()}")
        discharge_results = None
    
    # ===== Process Radiology Reports =====
    print("\n" + "="*70)
    print("2️  RADIOLOGY REPORTS")
    print("="*70)
    
    radiology_input = "data/raw/radiology_sample.csv"
    radiology_output = "data/processed/synthetic_radiology_hindi_FIXED.csv"
    
    if (PROJECT_ROOT / radiology_input).exists():
        radiology_results = generate_synthetic_dataset(
            input_file=radiology_input,
            output_file=radiology_output,
            num_samples=5  # Start with 5 to test
        )
    else:
        print(f"  Skipping - file not found: {(PROJECT_ROOT / radiology_input).resolve()}")
        radiology_results = None
    
    # ===== Final Summary =====
    print("\n")
    print(" GENERATION COMPLETE!")
    
    if discharge_results is not None or radiology_results is not None:
        print("\n Your synthetic Hindi medical texts are ready!")
        print(f"\n Check these files:")
        if discharge_results is not None:
            print(f"   • {discharge_output}")
        if radiology_results is not None:
            print(f"   • {radiology_output}")
        
        print("\n Next steps:")
        print("   1. Open the CSV files and review the Hindi text")
        print("   2. Check if the text is:")
        print("      - Actually in Hindi (Devanagari script)")
        print("      - Original (not a translation)")
        print("      - Medically accurate")
        print("   3. If quality is good, increase num_samples to 10, 50, 100!")
        print("   4. For your dissertation, you now have SYNTHETIC data!")
    else:
        print("\n  No data was generated. Check the errors above.")


if __name__ == "__main__":
    main()
