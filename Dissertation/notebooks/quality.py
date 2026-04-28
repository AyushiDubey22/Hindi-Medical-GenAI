"""
Quality Analysis for Synthetic Hindi Medical Data
Analyzes: completeness, diversity, medical accuracy indicators
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_hindi_content(text):
    """Analyze Hindi text characteristics"""
    if not text or pd.isna(text):
        return {
            'has_hindi': False,
            'devanagari_percent': 0,
            'word_count': 0,
            'sentence_count': 0,
            'has_sections': False
        }
    
    text = str(text)
    
    # Count Devanagari characters
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    devanagari_percent = (devanagari_chars / total_chars * 100) if total_chars > 0 else 0
    
    # Word count (split by spaces)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (approximate by punctuation)
    sentences = re.split('[।\\.\\?\\!]', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Check for required sections
    required_sections = ['मरीज़ की जानकारी', 'मुख्य शिकायत', 'इतिहास', 'जांच', 'निदान', 'उपचार']
    has_sections = sum(1 for section in required_sections if section in text)
    
    return {
        'has_hindi': devanagari_chars > 0,
        'devanagari_percent': round(devanagari_percent, 2),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'has_sections': has_sections >= 3,  # At least 3 sections
        'section_count': has_sections
    }

def extract_medical_terms(text):
    """Extract common Hindi medical terms"""
    if not text or pd.isna(text):
        return []
    
    medical_terms = [
        'मरीज़', 'रोगी', 'बीमारी', 'दर्द', 'जांच', 'उपचार', 'निदान',
        'शिकायत', 'लक्षण', 'दवा', 'इलाज', 'अस्पताल', 'डॉक्टर',
        'रक्तचाप', 'बुखार', 'सर्जरी', 'परीक्षण', 'रिपोर्ट'
    ]
    
    found_terms = [term for term in medical_terms if term in str(text)]
    return found_terms

def analyze_dataset(file_path, dataset_name):
    """Comprehensive analysis of one dataset"""
    
    print(f"\n{'='*70}")
    print(f"📊 Analyzing: {dataset_name}")
    print(f"{'='*70}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None
    
    # Filter successful generations
    success_df = df[df['status'] == 'success'].copy()
    print(f"✅ Successful generations: {len(success_df)}/{len(df)} ({len(success_df)/len(df)*100:.1f}%)")
    
    if len(success_df) == 0:
        print("⚠️  No successful generations to analyze")
        return None
    
    # Analyze each text
    print("\n🔍 Analyzing Hindi content quality...")
    analyses = []
    for idx, row in success_df.iterrows():
        analysis = analyze_hindi_content(row['synthetic_hindi_text'])
        analysis['medical_terms'] = extract_medical_terms(row['synthetic_hindi_text'])
        analysis['medical_term_count'] = len(analysis['medical_terms'])
        analysis['length'] = row['generated_length']
        analyses.append(analysis)
    
    analysis_df = pd.DataFrame(analyses)
    
    # Statistics
    print(f"\n📈 Quality Metrics:")
    print(f"   • Avg Devanagari %: {analysis_df['devanagari_percent'].mean():.1f}%")
    print(f"   • Avg Word Count: {analysis_df['word_count'].mean():.0f}")
    print(f"   • Avg Sentence Count: {analysis_df['sentence_count'].mean():.0f}")
    print(f"   • Has Proper Sections: {analysis_df['has_sections'].sum()}/{len(analysis_df)} ({analysis_df['has_sections'].sum()/len(analysis_df)*100:.1f}%)")
    print(f"   • Avg Medical Terms: {analysis_df['medical_term_count'].mean():.1f}")
    
    # Length distribution
    print(f"\n📏 Length Statistics:")
    print(f"   • Min: {analysis_df['length'].min()} chars")
    print(f"   • Max: {analysis_df['length'].max()} chars")
    print(f"   • Mean: {analysis_df['length'].mean():.0f} chars")
    print(f"   • Median: {analysis_df['length'].median():.0f} chars")
    
    # Most common medical terms
    all_terms = []
    for terms in analyses:
        all_terms.extend(terms['medical_terms'])
    term_counts = Counter(all_terms)
    
    print(f"\n🏥 Top 10 Medical Terms:")
    for term, count in term_counts.most_common(10):
        print(f"   • {term}: {count} times")
    
    # Error analysis
    error_df = df[df['status'] != 'success']
    if len(error_df) > 0:
        print(f"\n⚠️  Error Breakdown:")
        for status, count in error_df['status'].value_counts().items():
            print(f"   • {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Sample outputs
    print(f"\n📝 Sample Output (first 300 chars):")
    sample = success_df.iloc[0]['synthetic_hindi_text']
    print(f"   {sample[:300]}...")
    
    # Create visualizations
    create_visualizations(analysis_df, error_df, dataset_name)
    
    return {
        'total': len(df),
        'success': len(success_df),
        'success_rate': len(success_df)/len(df)*100,
        'avg_length': analysis_df['length'].mean(),
        'avg_medical_terms': analysis_df['medical_term_count'].mean(),
        'has_sections_pct': analysis_df['has_sections'].sum()/len(analysis_df)*100,
        'analysis_df': analysis_df,
        'error_df': error_df
    }

def create_visualizations(analysis_df, error_df, dataset_name):
    """Create quality visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Quality Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Length Distribution
    axes[0, 0].hist(analysis_df['length'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(analysis_df['length'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Medical Terms Distribution
    axes[0, 1].hist(analysis_df['medical_term_count'], bins=15, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Medical Terms')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Medical Terms per Text')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Section Completeness
    section_data = analysis_df['section_count'].value_counts().sort_index()
    axes[1, 0].bar(section_data.index, section_data.values, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Sections')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Section Completeness')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Quality Scores
    quality_metrics = {
        'Avg Length\n(chars)': analysis_df['length'].mean(),
        'Medical\nTerms': analysis_df['medical_term_count'].mean(),
        'Devanagari\n(%)': analysis_df['devanagari_percent'].mean(),
        'Sections': analysis_df['section_count'].mean()
    }
    axes[1, 1].bar(quality_metrics.keys(), quality_metrics.values(), color='purple', edgecolor='black')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Overall Quality Metrics')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = REPORTS_DIR / f'quality_analysis_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_file}")
    plt.close()

def generate_report():
    """Generate comprehensive quality report"""
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE QUALITY ANALYSIS")
    print("="*70)
    
    # Analyze both datasets
    discharge_file = PROCESSED_DIR / "synthetic_discharge_hindi_100.csv"
    radiology_file = PROCESSED_DIR / "synthetic_radiology_hindi_100.csv"
    
    discharge_stats = analyze_dataset(discharge_file, "Discharge Summaries")
    radiology_stats = analyze_dataset(radiology_file, "Radiology Reports")
    
    # Combined statistics
    print("\n" + "="*70)
    print("📈 COMBINED STATISTICS")
    print("="*70)
    
    if discharge_stats and radiology_stats:
        total_records = discharge_stats['total'] + radiology_stats['total']
        total_success = discharge_stats['success'] + radiology_stats['success']
        
        print(f"\n✅ Overall Success Rate: {total_success}/{total_records} ({total_success/total_records*100:.1f}%)")
        print(f"📏 Average Length: {(discharge_stats['avg_length'] + radiology_stats['avg_length'])/2:.0f} chars")
        print(f"🏥 Avg Medical Terms: {(discharge_stats['avg_medical_terms'] + radiology_stats['avg_medical_terms'])/2:.1f}")
        print(f"📋 Section Completeness: {(discharge_stats['has_sections_pct'] + radiology_stats['has_sections_pct'])/2:.1f}%")
        
        # Save summary report
        summary = {
            'Dataset': ['Discharge', 'Radiology', 'Combined'],
            'Total': [discharge_stats['total'], radiology_stats['total'], total_records],
            'Success': [discharge_stats['success'], radiology_stats['success'], total_success],
            'Success Rate (%)': [
                discharge_stats['success_rate'],
                radiology_stats['success_rate'],
                total_success/total_records*100
            ],
            'Avg Length': [
                discharge_stats['avg_length'],
                radiology_stats['avg_length'],
                (discharge_stats['avg_length'] + radiology_stats['avg_length'])/2
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_file = REPORTS_DIR / 'quality_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved: {summary_file}")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"\n📁 Reports saved in: {REPORTS_DIR}")
    print("\n📋 Next Steps:")
    print("   1. Review the visualizations")
    print("   2. Check sample outputs manually")
    print("   3. Run bias detection analysis")
    print("   4. Proceed to validation phase")

if __name__ == "__main__":
    # Install matplotlib if needed
    try:
        import matplotlib
    except ImportError:
        print("📦 Installing matplotlib...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'matplotlib', 'seaborn'])
    
    generate_report()