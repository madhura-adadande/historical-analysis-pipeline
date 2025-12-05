"""
Metrics Calculator - Statistical validation for consistency evaluations

Implements:
- Prompt strategy comparison (ablation)
- Reliability testing (self-consistency)
- Human agreement (Cohen's Kappa)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import cohen_kappa_score
import sys
from pathlib import Path

# Use relative import for sibling module
from ..evaluator.consistency_checker import ConsistencyEvaluator, load_claims


@dataclass
class StrategyMetrics:
    """Metrics for one prompt strategy."""
    strategy_name: str
    scores: List[float]
    avg: float
    std_dev: float
    min_val: float
    max_val: float


@dataclass
class ReliabilityMetrics:
    """Metrics from repeated runs."""
    event_key: str
    event_label: str
    pair_desc: str
    run_scores: List[int]
    avg: float
    std_dev: float
    score_range: int
    is_stable: bool


@dataclass
class AgreementMetrics:
    """Human-LLM agreement metrics."""
    kappa_value: float
    meaning: str
    llm_labels: List[str]
    human_labels: List[str]
    simple_agreement: float


class MetricsCalculator:
    """Calculates statistical validation metrics."""
    
    KAPPA_LEVELS = [
        (0.81, 1.00, "Almost perfect"),
        (0.61, 0.80, "Substantial"),
        (0.41, 0.60, "Moderate"),
        (0.21, 0.40, "Fair"),
        (0.01, 0.20, "Slight"),
        (-1.0, 0.00, "Poor/None"),
    ]
    
    def __init__(self, llm_model: str = "gpt-4o"):
        self.checker = ConsistencyEvaluator(llm_model=llm_model)
        self.output_path = Path("data/validation")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Ablation Study ---
    
    def compare_strategies(self, primary: List[Dict], secondary: List[Dict],
                           strategies: List[str] = None) -> Dict[str, StrategyMetrics]:
        """Run same comparisons with different prompt strategies."""
        if strategies is None:
            strategies = ["zero_shot", "chain_of_thought", "few_shot"]
        
        results = {}
        
        for strat in strategies:
            print(f"\n  Testing strategy: {strat}")
            comparisons = self.checker.evaluate_pairs(primary, secondary, method=strat)
            scores = [c.alignment_score for c in comparisons]
            
            if scores:
                results[strat] = StrategyMetrics(
                    strategy_name=strat,
                    scores=scores,
                    avg=float(np.mean(scores)),
                    std_dev=float(np.std(scores)),
                    min_val=float(min(scores)),
                    max_val=float(max(scores))
                )
                print(f"    Mean: {results[strat].avg:.1f}, Std: {results[strat].std_dev:.1f}")
        
        return results
    
    # --- Reliability Testing ---
    
    def test_reliability(self, primary: List[Dict], secondary: List[Dict],
                         sample_count: int = 3, repetitions: int = 5,
                         stability_threshold: float = 10.0) -> List[ReliabilityMetrics]:
        """Run same comparison multiple times to test consistency."""
        results = []
        
        # Get sample pairs
        pairs = self._get_sample_pairs(primary, secondary, sample_count)
        
        for pair in pairs:
            p_claims = pair["primary_claims"]
            s_claims = pair["secondary_claims"]
            event_key = pair["event_key"]
            event_label = pair["event_label"]
            writer = pair["writer"]
            
            print(f"\n  Testing: {event_label} vs {writer}")
            run_scores = []
            
            for run in range(repetitions):
                result = self.checker.evaluate_single(
                    p_claims, s_claims,
                    event_key, event_label,
                    pair["primary_source"], pair["secondary_source"], writer,
                    method="chain_of_thought",
                    temp=0.7  # Higher temp for variability
                )
                if result:
                    run_scores.append(result.alignment_score)
            
            if run_scores:
                std = float(np.std(run_scores))
                metrics = ReliabilityMetrics(
                    event_key=event_key,
                    event_label=event_label,
                    pair_desc=f"{event_label} ({writer})",
                    run_scores=run_scores,
                    avg=float(np.mean(run_scores)),
                    std_dev=std,
                    score_range=max(run_scores) - min(run_scores),
                    is_stable=std < stability_threshold
                )
                results.append(metrics)
                print(f"    Scores: {run_scores}, Std: {std:.1f}")
        
        return results
    
    def _get_sample_pairs(self, primary: List[Dict], secondary: List[Dict], count: int) -> List[Dict]:
        """Extract sample pairs for testing."""
        pairs = []
        
        primary_map = {}
        for p in primary:
            ek = p.get("event_key") or p.get("event_id")
            if ek:
                primary_map[ek] = p
        
        for s in secondary:
            ek = s.get("event_key") or s.get("event_id")
            if ek in primary_map and len(pairs) < count:
                p = primary_map[ek]
                pairs.append({
                    "event_key": ek,
                    "event_label": p.get("event_label") or p.get("event_name", ek),
                    "primary_claims": p.get("statements") or p.get("claims", []),
                    "secondary_claims": s.get("statements") or s.get("claims", []),
                    "primary_source": p.get("source_key") or p.get("source_id", ""),
                    "secondary_source": s.get("source_key") or s.get("source_id", ""),
                    "writer": s.get("writer") or s.get("author", "Unknown")
                })
        
        return pairs
    
    # --- Cohen's Kappa ---
    
    def generate_labeling_template(self, primary: List[Dict], secondary: List[Dict],
                                   count: int = 10) -> str:
        """Create template for human labeling."""
        pairs = self._get_sample_pairs(primary, secondary, count)
        
        template = {
            "instructions": "Label each pair as 'consistent' or 'contradictory' based on your judgment",
            "pairs": []
        }
        
        for i, pair in enumerate(pairs):
            # Get LLM judgment
            result = self.checker.evaluate_single(
                pair["primary_claims"], pair["secondary_claims"],
                pair["event_key"], pair["event_label"],
                pair["primary_source"], pair["secondary_source"], pair["writer"]
            )
            
            llm_label = "consistent" if result and result.alignment_score >= 50 else "contradictory"
            
            template["pairs"].append({
                "pair_id": i + 1,
                "event": pair["event_label"],
                "writer": pair["writer"],
                "primary_claims": pair["primary_claims"][:3],
                "secondary_claims": pair["secondary_claims"][:3],
                "llm_label": llm_label,
                "llm_score": result.alignment_score if result else None,
                "human_label": ""  # To be filled
            })
        
        output_file = self.output_path / "manual_labels_template.json"
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(template, fp, indent=2, ensure_ascii=False)
        
        print(f"Created labeling template: {output_file}")
        return str(output_file)
    
    def compute_kappa(self, template_file: str = None) -> Optional[AgreementMetrics]:
        """Calculate Cohen's Kappa from completed template."""
        if template_file is None:
            template_file = self.output_path / "manual_labels_template.json"
        
        with open(template_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        llm_labels = []
        human_labels = []
        
        for pair in data.get("pairs", []):
            llm = pair.get("llm_label", "")
            human = pair.get("human_label", "")
            
            if llm and human:
                llm_labels.append(llm)
                human_labels.append(human)
        
        if len(llm_labels) < 2:
            print("Not enough labeled pairs")
            return None
        
        # Calculate kappa
        kappa = cohen_kappa_score(llm_labels, human_labels)
        
        # Get interpretation
        meaning = "Unknown"
        for low, high, desc in self.KAPPA_LEVELS:
            if low <= kappa <= high:
                meaning = desc
                break
        
        # Simple agreement
        matches = sum(1 for l, h in zip(llm_labels, human_labels) if l == h)
        simple = matches / len(llm_labels)
        
        return AgreementMetrics(
            kappa_value=kappa,
            meaning=meaning,
            llm_labels=llm_labels,
            human_labels=human_labels,
            simple_agreement=simple
        )
    
    # --- Export ---
    
    def export_report(self, ablation_data: Dict = None, reliability_data: List = None,
                      agreement_data: AgreementMetrics = None):
        """Export validation report."""
        report = {"timestamp": str(Path.cwd())}
        
        if ablation_data:
            report["ablation_study"] = {
                name: asdict(metrics) for name, metrics in ablation_data.items()
            }
        
        if reliability_data:
            report["reliability_test"] = [asdict(m) for m in reliability_data]
        
        if agreement_data:
            report["agreement_metrics"] = asdict(agreement_data)
        
        output_file = self.output_path / "validation_report.json"
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        
        print(f"Exported validation report: {output_file}")


if __name__ == "__main__":
    calc = MetricsCalculator()
    primary = load_claims("data/extracted/loc_events.json")
    secondary = load_claims("data/extracted/gutenberg_events.json")
    
    ablation = calc.compare_strategies(primary, secondary)
    reliability = calc.test_reliability(primary, secondary)
    calc.export_report(ablation_data=ablation, reliability_data=reliability)
