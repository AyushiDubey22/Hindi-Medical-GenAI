"""
Production-Ready Hindi Medical Synthetic Data Generator
- Handles 100+ samples reliably
- Robust error handling and recovery
- Incremental saves (won't lose progress)
- Content filtering bypass strategies
- Quality validation
"""

import pandas as pd
import google.generativeai as genai
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import json

# ========================================
# CONFIGURATION
# ========================================

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash')

# Generation settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
SAVE_INTERVAL = 10  # Save progress every N samples
RATE_LIMIT_DELAY = 2  # seconds between requests

# ========================================
# LOGGING SETUP
# ========================================

log_file = LOGS_DIR / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_message(message, level="INFO"):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

# ========================================
# SETUP GEMINI
# ========================================

log_message("="*70)
log_message("🔧 Setting up Gemini API...")
log_message("="*70)

if not api_key:
    log_message("ERROR: No GOOGLE_API_KEY found in .env file!", "ERROR")
    sys.exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    log_message(f"Gemini configured: {MODEL_NAME}")
except Exception as e:
    log_message(f" Error setting up model: {e}", "ERROR")
    sys.exit(1)

# ========================================
# IMPROVED PROMPT (Less Likely to be Filtered)
# ========================================

GENERATION_PROMPT = """आप एक भारतीय चिकित्सा दस्तावेज़ीकरण विशेषज्ञ हैं जो नई हिंदी चिकित्सा रिपोर्ट बना रहे हैं।

**महत्वपूर्ण निर्देश:** नीचे दिए गए अंग्रेजी पाठ का अनुवाद न करें। इसका उपयोग केवल संदर्भ के लिए करें और एक पूरी तरह से नई, मूल हिंदी चिकित्सा रिपोर्ट बनाएं।

**संदर्भ (केवल विचार के लिए):**

**आपका कार्य:**
एक नई, मौलिक हिंदी चिकित्सा रिपोर्ट बनाएं जिसमें:
1. समान प्रकार की चिकित्सा स्थिति हो (जैसे: बुखार, पेट दर्द, सर्जरी आदि)
2. पूरी तरह से अलग विवरण हों (अलग मरीज़ का नाम, अलग लक्षण, अलग मूल्य)
3. भारतीय अस्पताल के मानक प्रारूप का पालन करे
4. प्राकृतिक हिंदी चिकित्सा शब्दावली का उपयोग करे
5. यथार्थवादी भारतीय संदर्भ शामिल हो (भारतीय नाम, स्थानीय प्रथाएं)

**आवश्यक अनुभाग:**
• मरीज़ की जानकारी - नाम (जैसे: राज कुमार, प्रिया शर्मा), उम्र, लिंग
• मुख्य शिकायत - क्या समस्या है
• रोग का इतिहास - समस्या कब और कैसे शुरू हुई
• शारीरिक जांच - डॉक्टर ने क्या देखा
• निदान - क्या बीमारी है
• उपचार - क्या इलाज दिया गया

**केवल हिंदी (देवनागरी) में लिखें। एक पूर्ण, मौलिक चिकित्सा नोट बनाएं:**"""

# ========================================
# ERROR HANDLING & RETRY LOGIC
# ========================================

def generate_with_retry(reference_text, attempt=1):
    """
    Generate with comprehensive error handling and retry logic
    
    Returns: (hindi_text, status, error_details)
    """
    
    if not reference_text or str(reference_text).strip() == '':
        return '', 'empty_input', None
    
    # Prepare prompt with truncated reference
    short_reference = str(reference_text)[:1000]
    prompt = GENERATION_PROMPT.format(reference_text=short_reference)
    
    try:
        log_message(f"  Attempt {attempt}/{MAX_RETRIES}...", "DEBUG")
        
        # Configure generation with safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.9,  # More creative
            top_p=0.95,
            top_k=40,
            max_output_tokens=2000,  # Longer output
            candidate_count=1
        )
        
        # Generate
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Extract text with multiple fallback methods
        hindi_text = None
        
        # Method 1: Direct text access
        try:
            hindi_text = response.text
        except:
            pass
        
        # Method 2: Check candidates
        if not hindi_text and hasattr(response, 'candidates'):
            try:
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if candidate.content.parts:
                            hindi_text = candidate.content.parts[0].text
            except:
                pass
        
        # Method 3: Check prompt_feedback
        if not hindi_text:
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason'):
                    return '', 'content_filtered', str(feedback.block_reason)
        
        if not hindi_text:
            # Check finish reason
            finish_reason = None
            try:
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
            except:
                pass
            
            if finish_reason == 2:  # SAFETY
                log_message("   Content filtered by safety settings", "WARNING")
                # Try with simpler prompt
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    # Use simplified reference
                    simpler_ref = str(reference_text)[:500]
                    return generate_with_retry(simpler_ref, attempt + 1)
                return '', 'content_filtered_max_retries', 'finish_reason=2'
            
            return '', 'no_text_generated', f'finish_reason={finish_reason}'
        
        hindi_text = hindi_text.strip()
        
        # Validation
        if len(hindi_text) < 200:
            log_message(f"   Output too short ({len(hindi_text)} chars)", "WARNING")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return generate_with_retry(reference_text, attempt + 1)
            return hindi_text, 'output_too_short', None
        
        # Check for Hindi
        has_hindi = any('\u0900' <= char <= '\u097F' for char in hindi_text)
        if not has_hindi:
            log_message("   No Hindi detected", "WARNING")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return generate_with_retry(reference_text, attempt + 1)
            return hindi_text, 'no_hindi_detected', None
        
        # Check for medical terms
        medical_terms = ['मरीज़', 'रोगी', 'जांच', 'उपचार', 'निदान', 'शिकायत', 'दर्द', 'बीमारी']
        has_medical = any(term in hindi_text for term in medical_terms)
        
        if not has_medical:
            log_message("   No medical terms detected", "WARNING")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return generate_with_retry(reference_text, attempt + 1)
            return hindi_text, 'no_medical_terms', None
        
        log_message(f"  Success! {len(hindi_text)} chars", "SUCCESS")
        return hindi_text, 'success', None
        
    except Exception as e:
        error_msg = str(e)
        log_message(f"  Error: {error_msg[:100]}", "ERROR")
        
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
            return generate_with_retry(reference_text, attempt + 1)
        
        return '', f'error_max_retries', error_msg

# ========================================
# BATCH PROCESSING WITH INCREMENTAL SAVES
# ========================================

def generate_batch(input_file, output_file, num_samples=100, start_index=0):
    """
    Generate batch with incremental saves and progress tracking
    """
    
    log_message("="*70)
    log_message(f"Processing: {input_file.name}")
    log_message("="*70)
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        log_message(f" Loaded {len(df)} total records")
    except Exception as e:
        log_message(f" Error loading file: {e}", "ERROR")
        return None
    
    # Check if we can resume from existing output
    existing_results = []
    if output_file.exists() and start_index == 0:
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            start_index = len(existing_results)
            log_message(f" Resuming from existing file: {start_index} samples done")
        except:
            pass
    
    # Calculate range
    end_index = min(start_index + num_samples, len(df))
    df_subset = df.iloc[start_index:end_index]
    
    log_message(f" Generating {len(df_subset)} samples (index {start_index} to {end_index})")
    log_message(f"  Estimated time: ~{len(df_subset) * 3 / 60:.1f} minutes\n")
    
    results = existing_results.copy()
    success_count = len([r for r in results if r.get('status') == 'success'])
    
    # Process each sample
    for idx, (_, row) in enumerate(df_subset.iterrows(), start=start_index):
        log_message(f"\n[{idx-start_index+1}/{len(df_subset)}] Processing ID: {row.get('note_id', idx)}")
        
        # Get reference text
        reference_text = row.get('cleaned_text', row.get('text', ''))
        
        if not reference_text:
            log_message("   Skipping: No text found", "WARNING")
            results.append({
                'synthetic_id': f'SYN_{idx:04d}',
                'reference_note_id': row.get('note_id', f'NOTE_{idx}'),
                'note_type': row.get('note_type', 'unknown'),
                'original_preview': '',
                'synthetic_hindi_text': '',
                'status': 'no_input_text',
                'error_details': 'No text in cleaned_text or text column',
                'original_length': 0,
                'generated_length': 0,
                'generation_time_sec': 0,
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Generate
        start_time = time.time()
        hindi_text, status, error_details = generate_with_retry(reference_text)
        duration = time.time() - start_time
        
        # Store result
        result = {
            'synthetic_id': f'SYN_{idx:04d}',
            'reference_note_id': row.get('note_id', f'NOTE_{idx}'),
            'note_type': row.get('note_type', 'unknown'),
            'original_preview': str(reference_text)[:300] + '...',
            'synthetic_hindi_text': hindi_text,
            'status': status,
            'error_details': error_details if error_details else '',
            'original_length': len(str(reference_text)),
            'generated_length': len(hindi_text),
            'generation_time_sec': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        if status == 'success':
            success_count += 1
        
        # Incremental save
        if (idx - start_index + 1) % SAVE_INTERVAL == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            log_message(f" Progress saved: {len(results)} samples", "INFO")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Statistics
    log_message("\n" + "="*70)
    log_message(" BATCH STATISTICS")
    log_message("="*70)
    log_message(f" Total Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    log_message(f" Avg length: {results_df[results_df['status']=='success']['generated_length'].mean():.0f} chars")
    log_message(f"  Total time: {results_df['generation_time_sec'].sum()/60:.1f} minutes")
    log_message(f" Saved to: {output_file}")
    
    # Error summary
    error_counts = results_df[results_df['status'] != 'success']['status'].value_counts()
    if len(error_counts) > 0:
        log_message("\n  Error Summary:")
        for error_type, count in error_counts.items():
            log_message(f"   • {error_type}: {count}")
    
    return results_df

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution with full pipeline"""
    
    log_message("\n" + "="*70)
    log_message(" PRODUCTION Hindi Medical Synthetic Generator")
    log_message("="*70)
    log_message(f" Model: {MODEL_NAME}")
    log_message(f"Log file: {log_file}")
    log_message("="*70)
    
    # Process discharge summaries
    log_message("\n" + "="*70)
    log_message("1️  DISCHARGE SUMMARIES")
    log_message("="*70)
    
    discharge_input = RAW_DIR / "discharge.csv"
    discharge_output = PROCESSED_DIR / "synthetic_discharge_hindi_100.csv"
    
    if discharge_input.exists():
        discharge_results = generate_batch(
            input_file=discharge_input,
            output_file=discharge_output,
            num_samples=100  # Generate 100 samples
        )
    else:
        log_message(f" File not found: {discharge_input}", "WARNING")
        discharge_results = None
    
    # Process radiology reports
    log_message("\n" + "="*70)
    log_message("2RADIOLOGY REPORTS")
    log_message("="*70)
    
    radiology_input = RAW_DIR / "radiology_sample.csv"
    radiology_output = PROCESSED_DIR / "synthetic_radiology_hindi_100.csv"
    
    if radiology_input.exists():
        radiology_results = generate_batch(
            input_file=radiology_input,
            output_file=radiology_output,
            num_samples=100  # Generate 100 samples
        )
    else:
        log_message(f"  File not found: {radiology_input}", "WARNING")
        radiology_results = None
    
    # Final summary
    log_message("\n" + "="*70)
    log_message(" GENERATION COMPLETE!")
    log_message("="*70)
    
    if discharge_results is not None or radiology_results is not None:
        log_message("\n Synthetic Hindi medical data generated successfully!")
        log_message(f"\n Output files:")
        if discharge_results is not None:
            log_message(f"   • {discharge_output}")
        if radiology_results is not None:
            log_message(f"   • {radiology_output}")
        
        log_message(f"\n Next steps:")
        log_message("   1. Review the CSV files")
        log_message("   2. Check quality of Hindi text")
        log_message("   3. Run validation analysis")
        log_message("   4. Use for model training!")
        log_message(f"\n Detailed logs: {log_file}")
    else:
        log_message("\n  No data generated", "WARNING")
    
    log_message("\n" + "="*70)

if __name__ == "__main__":
    main()