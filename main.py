#!/usr/bin/env python3
"""
Historical Text Analysis Pipeline - Entry Point

Execute different stages of the analysis:
    python main.py --full         # Complete workflow
    python main.py --collect      # Fetch source documents
    python main.py --parse        # Extract claims from texts
    python main.py --evaluate     # Run consistency checks
    python main.py --analyze      # Statistical analysis
"""

import os
import sys
import argparse
from pathlib import Path

# Configuration loading
from dotenv import load_dotenv
CONFIG_FILE = Path(__file__).parent / ".env"

if CONFIG_FILE.exists():
    load_dotenv(CONFIG_FILE)
    print("[OK] Configuration loaded")

# Module path setup
sys.path.insert(0, str(Path(__file__).parent / "modules"))


def verify_api_credentials():
    """Ensure API credentials are available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "-"*50)
        print("ERROR: Missing API credentials")
        print("-"*50)
        print("\nProvide your key via environment variable:")
        print("  PowerShell: $env:OPENAI_API_KEY='sk-...'")
        print("  CMD: set OPENAI_API_KEY=sk-...")
        print("  Bash: export OPENAI_API_KEY='sk-...'")
        print("\nOr add to .env file in project root")
        return False
    return True


def stage_collect():
    """Stage 1: Fetch documents from online archives."""
    print("\n" + "-"*50)
    print("STAGE 1: DOCUMENT COLLECTION")
    print("-"*50)
    
    os.chdir(Path(__file__).parent)
    
    # Fetch from Gutenberg
    print("\n[Gutenberg Archive]")
    from collectors.book_fetcher import BookCollector
    
    collector = BookCollector(save_path="data/raw/gutenberg")
    texts = collector.fetch_all_books(wait_time=1.0)
    if texts:
        collector.export_dataset(texts, "data/processed/gutenberg.json")
    
    # Fetch from Library of Congress
    print("\n[Congress Library Archive]")
    from collectors.archive_fetcher import ArchiveCollector
    
    archive = ArchiveCollector(save_path="data/raw/loc")
    docs = archive.fetch_all_documents(wait_time=2.0)
    if docs:
        archive.export_dataset(docs, "data/processed/loc.json")
    
    print(f"\n[OK] Collection finished: {len(texts)} books, {len(docs)} documents")


def stage_parse():
    """Stage 2: Parse and extract claims from documents."""
    print("\n" + "-"*50)
    print("STAGE 2: CLAIM EXTRACTION")
    print("-"*50)
    
    if not verify_api_credentials():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from parsers.claim_parser import ClaimExtractor
    
    parser = ClaimExtractor(llm_model="gpt-4o-mini")
    
    books_file = Path("data/processed/gutenberg.json")
    docs_file = Path("data/processed/loc.json")
    
    if not books_file.exists() or not docs_file.exists():
        print("ERROR: Source data not found. Run --collect first.")
        return False
    
    parser.extract_from_file(str(books_file), "gutenberg")
    parser.extract_from_file(str(docs_file), "loc")
    
    print("\n[OK] Extraction finished")
    return True


def stage_evaluate():
    """Stage 3: Evaluate consistency between sources."""
    print("\n" + "-"*50)
    print("STAGE 3: CONSISTENCY EVALUATION")
    print("-"*50)
    
    if not verify_api_credentials():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from evaluator.consistency_checker import ConsistencyEvaluator, load_claims
    
    primary_file = Path("data/extracted/loc_events.json")
    secondary_file = Path("data/extracted/gutenberg_events.json")
    
    if not primary_file.exists() or not secondary_file.exists():
        print("ERROR: Extracted data not found. Run --parse first.")
        return False
    
    checker = ConsistencyEvaluator(llm_model="gpt-4o")
    
    primary_claims = load_claims(str(primary_file))
    secondary_claims = load_claims(str(secondary_file))
    
    output = checker.evaluate_pairs(
        primary_claims,
        secondary_claims,
        method="chain_of_thought"
    )
    
    checker.export_results(output)
    
    print("\n[OK] Evaluation finished")
    return True


def stage_analyze():
    """Stage 4: Statistical analysis and validation."""
    print("\n" + "-"*50)
    print("STAGE 4: STATISTICAL ANALYSIS")
    print("-"*50)
    
    if not verify_api_credentials():
        return False
    
    os.chdir(Path(__file__).parent)
    
    from analysis.metrics import MetricsCalculator
    from evaluator.consistency_checker import load_claims
    
    primary_file = Path("data/extracted/loc_events.json")
    secondary_file = Path("data/extracted/gutenberg_events.json")
    
    if not primary_file.exists() or not secondary_file.exists():
        print("ERROR: Extracted data not found. Run --parse first.")
        return False
    
    calculator = MetricsCalculator(llm_model="gpt-4o")
    
    primary = load_claims(str(primary_file))
    secondary = load_claims(str(secondary_file))
    
    # Prompt strategy comparison
    print("\n[Ablation Study]")
    ablation = calculator.compare_strategies(primary, secondary)
    
    # Reliability testing
    print("\n[Reliability Test]")
    reliability = calculator.test_reliability(
        primary, secondary,
        sample_count=3,
        repetitions=5
    )
    
    # Human comparison template
    print("\n[Creating Human Labeling Template]")
    calculator.generate_labeling_template(primary, secondary, count=10)
    
    # Export results
    calculator.export_report(
        ablation_data=ablation,
        reliability_data=reliability
    )
    
    print("\n[OK] Analysis finished")
    print("\nManual step required:")
    print("  Fill in data/validation/manual_labels_template.json")
    return True


def execute():
    """Main execution handler."""
    cli = argparse.ArgumentParser(
        description="Historical Text Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
    python main.py --full         Run complete analysis
    python main.py --collect      Fetch source documents only
    python main.py --parse        Extract claims only
    python main.py --evaluate     Run consistency checks only
    python main.py --analyze      Statistical analysis only
        """
    )
    
    cli.add_argument("--full", action="store_true", help="Execute complete pipeline")
    cli.add_argument("--collect", action="store_true", help="Fetch source documents")
    cli.add_argument("--parse", action="store_true", help="Extract claims from texts")
    cli.add_argument("--evaluate", action="store_true", help="Run consistency evaluation")
    cli.add_argument("--analyze", action="store_true", help="Statistical analysis")
    
    opts = cli.parse_args()
    
    if not any([opts.full, opts.collect, opts.parse, opts.evaluate, opts.analyze]):
        cli.print_help()
        return
    
    print("="*50)
    print("HISTORICAL TEXT ANALYSIS PIPELINE")
    print("="*50)
    
    if opts.full or opts.collect:
        stage_collect()
    
    if opts.full or opts.parse:
        stage_parse()
    
    if opts.full or opts.evaluate:
        stage_evaluate()
    
    if opts.full or opts.analyze:
        stage_analyze()
    
    print("\n" + "="*50)
    print("FINISHED")
    print("="*50)


if __name__ == "__main__":
    execute()
