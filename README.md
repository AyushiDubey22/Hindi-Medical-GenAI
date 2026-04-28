# Synthetic Hindi Medical Text Generation Pipeline

> A comprehensive pipeline for generating high-quality synthetic Hindi medical text for low-resource healthcare applications, with built-in bias mitigation and error detection.

## 🎯 Project Objectives

This project designs a synthetic data generation pipeline that:

1. **Creates high-quality training data** for low-resource settings (Hindi medical text)
2. **Identifies and mitigates bias** in both real and synthetic data
3. **Detects and reduces error propagation** in the pipeline
4. **Ensures efficiency** by optimizing data/resource usage
5. **Evaluates improvements** in model performance, fairness, and efficiency

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Datasets](#datasets)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## 🔍 Problem Statement

Hindi is a low-resource language in the medical domain, with limited training data available for NLP applications. This creates challenges for:
- Medical chatbots and virtual assistants
- Clinical documentation systems
- Patient-doctor communication tools
- Medical education platforms

This project addresses these challenges by generating synthetic Hindi medical text that maintains medical accuracy while being culturally and linguistically appropriate.

## 📊 Datasets

### Source Data

1. **MIMIC-IV Clinical Database**
   - **Type:** English clinical notes
   - **Content:** Discharge summaries, radiology reports, clinical notes
   - **Size:** 1,000 sample records (cleaned)
   - **Location:** `data/raw/`
   - **Usage:** Source material for translation and synthesis

2. **IndicCorp V2 - Hindi Corpus**
   - **Type:** Hindi text corpus
   - **Size:** ~8-9 billion tokens
   - **Usage:** Reference for natural Hindi language patterns
   - **Access:** Streaming mode via Hugging Face
   - **Location:** `data/raw/hindi_corpus_sample/`

### Generated Data

- **Hindi Synthetic Medical Text:** `data/processed/hindi_synthetic_output.csv`
- **Validation Results:** `data/validation/quality_checks.csv`

## 🏗️ Pipeline Architecture

```
┌─────────────────┐
│  MIMIC-IV Data  │
│ (English Notes) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Preprocessing     │
│  - Cleaning             │
│  - Sampling             │
│  - Deidentification     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐      ┌──────────────────┐
│  LLM Translation        │◄─────┤ IndicCorp V2     │
│  - Medical terminology  │      │ (Hindi patterns) │
│  - Context preservation │      └──────────────────┘
│  - Natural phrasing     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Bias Detection         │
│  - Gender bias check    │
│  - Terminology bias     │
│  - Cultural sensitivity │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Error Propagation      │
│  Detection              │
│  - Medical accuracy     │
│  - Translation quality  │
│  - Consistency checks   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Quality Validation     │
│  - Human review         │
│  - Automated metrics    │
│  - Iterative refinement │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Synthetic Hindi        │
│  Medical Dataset        │
└─────────────────────────┘
```

## 📁 Project Structure

```
DISSERTATION/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── discharge.csv             # MIMIC-IV discharge summaries
│   │   ├── radiology_sample.csv      # MIMIC-IV radiology reports
│   │   └── hindi_corpus_sample/      # IndicCorp samples
│   ├── processed/                    # Generated synthetic data
│   │   └── hindi_synthetic_output.csv
│   └── validation/                   # Quality check results
│       ├── quality_checks.csv
│       └── bias_analysis.csv
│
├── src/
│   ├── __init__.py
│   ├── translate.py                  # Main translation pipeline
│   ├── validate.py                   # Quality validation module
│   ├── bias_detection.py             # Bias identification and mitigation
│   ├── error_propagation.py          # Error detection module
│   ├── batch_process.py              # Batch processing script
│   └── utils.py                      # Helper functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial data analysis
│   ├── 02_translation_test.ipynb     # Translation testing
│   ├── 03_bias_analysis.ipynb        # Bias detection analysis
│   ├── 04_error_analysis.ipynb       # Error propagation study
│   └── 05_evaluation.ipynb           # Final evaluation metrics
│
├── outputs/
│   ├── logs/                         # Processing logs
│   ├── samples/                      # Sample outputs for review
│   └── reports/                      # Evaluation reports
│
└── tests/
    ├── test_translation.py           # Unit tests for translation
    ├── test_validation.py            # Unit tests for validation
    └── test_bias_detection.py        # Unit tests for bias detection
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- API keys for chosen LLM (OpenAI/Google/Anthropic)

### Setup Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd DISSERTATION
```

2. **Create virtual environment**
```bash
# Using conda
conda create -n hindi_medical python=3.10
conda activate hindi_medical

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Choose ONE based on your LLM choice:
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AI...
# ANTHROPIC_API_KEY=sk-ant-...
```

5. **Verify installation**
```bash
python -c "import torch; import transformers; print('Setup successful!')"
```

## 💻 Usage

### Quick Start

1. **Explore the data**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. **Test translation on small sample**
```python
python src/translate.py --input data/raw/discharge.csv --output data/processed/test_output.csv --limit 10
```

3. **Run full pipeline**
```python
python src/batch_process.py --input data/raw/discharge.csv --output data/processed/hindi_synthetic_output.csv --batch-size 100
```

### Pipeline Components

#### 1. Translation
```python
from src.translate import MedicalTranslator

translator = MedicalTranslator(model="gpt-4o-mini")
hindi_text = translator.translate(english_text)
```

#### 2. Bias Detection
```python
from src.bias_detection import BiasDetector

detector = BiasDetector()
bias_report = detector.analyze(synthetic_data)
```

#### 3. Error Propagation Detection
```python
from src.error_propagation import ErrorDetector

error_detector = ErrorDetector()
errors = error_detector.check_medical_accuracy(original, translated)
```

#### 4. Quality Validation
```python
from src.validate import QualityValidator

validator = QualityValidator()
quality_score = validator.evaluate(hindi_text)
```

## 📊 Evaluation Metrics

### 1. Data Quality Metrics
- **Translation Accuracy:** BLEU, METEOR scores
- **Medical Term Preservation:** % of medical terms correctly translated
- **Fluency Score:** Native speaker ratings (1-5 scale)
- **Semantic Similarity:** Cosine similarity between embeddings

### 2. Bias Metrics
- **Gender Bias Score:** Difference in treatment suggestions by gender
- **Terminology Bias:** Formal vs colloquial term distribution
- **Cultural Sensitivity Score:** Cultural appropriateness rating

### 3. Error Propagation Metrics
- **Medical Error Rate:** % of medically inaccurate translations
- **Consistency Score:** Same term translated consistently
- **Hallucination Rate:** % of generated content not in source

### 4. Efficiency Metrics
- **Cost per Sample:** API costs / number of samples
- **Processing Time:** Average time per translation
- **Token Usage:** Input/output tokens per sample

### 5. Model Performance
- **Downstream Task Accuracy:** Performance on medical NER, classification
- **Fairness Metrics:** Equal performance across demographics
- **Robustness:** Performance on edge cases

## 🗺️ Roadmap

### Phase 1: Foundation (Weeks 1-2) ✅
- [x] Environment setup
- [x] Data collection and cleaning
- [ ] Initial translation pipeline
- [ ] Basic validation

### Phase 2: Core Pipeline (Weeks 3-4)
- [ ] Implement bias detection
- [ ] Build error propagation detection
- [ ] Optimize translation prompts
- [ ] Scale to 1,000 samples

### Phase 3: Evaluation (Weeks 5-6)
- [ ] Comprehensive quality assessment
- [ ] Bias analysis and mitigation
- [ ] Error analysis and fixes
- [ ] Performance benchmarking

### Phase 4: Optimization (Weeks 7-8)
- [ ] Cost optimization
- [ ] Speed improvements
- [ ] Quality refinement
- [ ] Final evaluation

### Phase 5: Documentation (Week 9)
- [ ] Final report
- [ ] Code documentation
- [ ] Research paper draft
- [ ] Presentation preparation

## 🤝 Contributing

This is a research project. For questions or suggestions:
- Open an issue for bugs or feature requests
- Submit pull requests for improvements
- Contact: [ayushidubey72@gmail.com]


## 🙏 Acknowledgments

- **MIMIC-IV Dataset:** PhysioNet, MIT Lab for Computational Physiology
- **IndicCorp V2:** AI4Bharat, IIT Madras
- **LLM Providers:** Google
- **Advisor:** Prof. Rushi Kumar

## 📚 References

1. Johnson, A., et al. (2023). MIMIC-IV Clinical Database
2. Kakwani, D., et al. (2020). IndicNLPSuite
3. [Add your other references]

## 📞 Contact

**Project Lead:** [Ayushi Dubey] 
**Email:** [ayushidubey72@gmail.com]
**Institution:** [Vellore Institute of Technoology]  
**Department:** [Department of Mathematics, SAS]

---

**Last Updated:** November 2025  
**Version:** 0.1.0 (Development)