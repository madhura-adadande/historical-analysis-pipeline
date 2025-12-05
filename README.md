# Historiographical Divergence Analyzer

An NLP-powered tool for detecting inconsistencies between primary and secondary historical sources using large language models.

---

## What This Does

This project builds an automated system to:

1. Collect historical documents from public archives
2. Extract structured claims about specific events
3. Compare how different authors describe the same events
4. Measure consistency and identify contradictions
5. Validate results with statistical rigor

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Add your OpenAI key to .env file
echo "OPENAI_API_KEY=sk-xxx" > .env

# Run everything
python run_pipeline.py --all
```

---

## How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SCRAPE    │───▶│   EXTRACT   │───▶│    JUDGE    │───▶│  VALIDATE   │
│  (sources)  │    │  (events)   │    │ (compare)   │    │  (stats)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Stage 1: Data Collection
- Downloads texts from Project Gutenberg (biographies)
- Fetches documents from Library of Congress (primary sources)
- Normalizes everything into consistent JSON format

### Stage 2: Event Extraction
- Chunks large documents for LLM processing
- Pre-filters with keywords to reduce API costs
- Extracts claims, dates, locations, and tone for each event

### Stage 3: Consistency Evaluation
- LLM Judge compares primary vs secondary accounts
- Outputs consistency score (0-100)
- Classifies contradictions: Factual | Interpretive | Omission

### Stage 4: Statistical Validation
- **Ablation Study**: Tests 3 prompting strategies
- **Self-Consistency**: 5 runs at temp=0.7
- **Cohen's Kappa**: Human vs LLM agreement

---

## Files

```
src/
  scrapers/           → gutenberg_scraper.py, loc_scraper.py
  extraction/         → event_extractor.py
  judge/              → llm_judge.py
  validation/         → statistical_tests.py

data/
  raw/                → original downloaded files
  processed/          → normalized JSON datasets
  extracted/          → LLM outputs (claims, judgments)
  validation/         → statistical reports

notebooks/
  analysis.ipynb      → final report with charts
```

---

## Configuration

Create `.env` in project root:
```
OPENAI_API_KEY=your-key-here
```

---

## CLI Options

| Command | Action |
|---------|--------|
| `--all` | Run full pipeline |
| `--scrape` | Data collection only |
| `--extract` | Event extraction only |
| `--judge` | Consistency evaluation only |
| `--validate` | Statistical validation only |

---

## Tech Stack

- Python 3.8+
- OpenAI GPT API
- BeautifulSoup (scraping)
- Pandas, NumPy (analysis)
- Scikit-learn (Cohen's Kappa)
- Jupyter (reporting)
