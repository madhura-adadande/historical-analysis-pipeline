"""
Consistency Checker - Compares claims between primary and secondary sources

Uses LLM to evaluate historiographical consistency and detect contradictions.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Discrepancy:
    """Represents a detected discrepancy."""
    category: str  # Factual, Interpretive, Omission
    primary_statement: str
    secondary_statement: str
    description: str
    importance: str  # High, Medium, Low


@dataclass
class ComparisonResult:
    """Output of comparing two claim sets."""
    event_key: str
    event_label: str
    primary_source: str
    secondary_source: str
    secondary_writer: str
    alignment_score: int  # 0-100
    discrepancies: List[Discrepancy]
    matches: List[str]
    analysis: str
    method: str
    temp: float


def load_claims(filepath: str) -> List[Dict]:
    """Load extracted claims from JSON file."""
    with open(filepath, "r", encoding="utf-8") as fp:
        return json.load(fp)


class ConsistencyEvaluator:
    """Evaluates consistency between historical accounts."""
    
    METHODS = ["zero_shot", "chain_of_thought", "few_shot"]
    
    def __init__(self, llm_model: str = "gpt-4o"):
        self.llm = OpenAI()
        self.model_name = llm_model
        self.output_path = Path("data/extracted")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _format_zero_shot(self, primary: List[str], secondary: List[str],
                          event: str, writer: str) -> str:
        """Zero-shot prompt - direct instructions."""
        p_text = "\n".join(f"- {s}" for s in primary) if primary else "- No claims"
        s_text = "\n".join(f"- {s}" for s in secondary) if secondary else "- No claims"
        
        return f'''Compare claims about "{event}":

PRIMARY SOURCE:
{p_text}

SECONDARY SOURCE ({writer}):
{s_text}

Return JSON:
{{
  "alignment_score": 0-100,
  "discrepancies": [{{"category": "Factual|Interpretive|Omission", "primary_statement": "...", "secondary_statement": "...", "description": "...", "importance": "High|Medium|Low"}}],
  "matches": ["matching claim 1", ...],
  "analysis": "brief explanation"
}}'''
    
    def _format_cot(self, primary: List[str], secondary: List[str],
                    event: str, writer: str) -> str:
        """Chain-of-thought prompt - step by step reasoning."""
        p_text = "\n".join(f"- {s}" for s in primary) if primary else "- No claims"
        s_text = "\n".join(f"- {s}" for s in secondary) if secondary else "- No claims"
        
        return f'''You are analyzing historical accounts of "{event}".

PRIMARY SOURCE (first-hand account):
{p_text}

SECONDARY SOURCE by {writer}:
{s_text}

Think step by step:
1. List claims that align between sources
2. Identify factual contradictions (different facts)
3. Note interpretive differences (same facts, different meaning)
4. Check for omissions (important details missing)
5. Calculate overall alignment score

Return JSON:
{{
  "alignment_score": 0-100,
  "discrepancies": [{{"category": "Factual|Interpretive|Omission", "primary_statement": "...", "secondary_statement": "...", "description": "...", "importance": "High|Medium|Low"}}],
  "matches": ["..."],
  "analysis": "step-by-step reasoning"
}}'''
    
    def _format_few_shot(self, primary: List[str], secondary: List[str],
                         event: str, writer: str) -> str:
        """Few-shot prompt - includes examples."""
        p_text = "\n".join(f"- {s}" for s in primary) if primary else "- No claims"
        s_text = "\n".join(f"- {s}" for s in secondary) if secondary else "- No claims"
        
        return f'''Compare historical accounts about "{event}".

EXAMPLE:
Primary: "The meeting occurred at noon"
Secondary: "The meeting was in the afternoon around 2pm"
Result: {{"category": "Factual", "primary_statement": "meeting at noon", "secondary_statement": "meeting at 2pm", "description": "Time discrepancy", "importance": "Medium"}}

EXAMPLE:
Primary: "He felt confident about the decision"
Secondary: "He agonized over the choice for days"
Result: {{"category": "Interpretive", "primary_statement": "felt confident", "secondary_statement": "agonized over choice", "description": "Different characterization of emotional state", "importance": "High"}}

NOW ANALYZE:

PRIMARY:
{p_text}

SECONDARY ({writer}):
{s_text}

Return JSON:
{{
  "alignment_score": 0-100,
  "discrepancies": [...],
  "matches": [...],
  "analysis": "..."
}}'''
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _query_llm(self, prompt: str, temp: float = 0.0) -> Optional[Dict]:
        """Execute LLM query."""
        try:
            resp = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a historical consistency analyst. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as err:
            print(f"  LLM error: {err}")
            return None
    
    def evaluate_single(self, primary_claims: List[str], secondary_claims: List[str],
                        event_key: str, event_label: str, primary_source: str,
                        secondary_source: str, secondary_writer: str,
                        method: str = "chain_of_thought", temp: float = 0.0) -> Optional[ComparisonResult]:
        """Compare two claim sets for one event."""
        
        if method == "zero_shot":
            prompt = self._format_zero_shot(primary_claims, secondary_claims, event_label, secondary_writer)
        elif method == "few_shot":
            prompt = self._format_few_shot(primary_claims, secondary_claims, event_label, secondary_writer)
        else:
            prompt = self._format_cot(primary_claims, secondary_claims, event_label, secondary_writer)
        
        result = self._query_llm(prompt, temp)
        if not result:
            return None
        
        discrepancies = []
        for d in result.get("discrepancies", []):
            discrepancies.append(Discrepancy(
                category=d.get("category", "Unknown"),
                primary_statement=d.get("primary_statement", ""),
                secondary_statement=d.get("secondary_statement", ""),
                description=d.get("description", ""),
                importance=d.get("importance", "Medium")
            ))
        
        return ComparisonResult(
            event_key=event_key,
            event_label=event_label,
            primary_source=primary_source,
            secondary_source=secondary_source,
            secondary_writer=secondary_writer,
            alignment_score=result.get("alignment_score", 50),
            discrepancies=discrepancies,
            matches=result.get("matches", []),
            analysis=result.get("analysis", ""),
            method=method,
            temp=temp
        )
    
    def evaluate_pairs(self, primary_data: List[Dict], secondary_data: List[Dict],
                       method: str = "chain_of_thought") -> List[ComparisonResult]:
        """Compare all matching event pairs."""
        results = []
        
        # Group by event
        primary_by_event = {}
        for item in primary_data:
            ek = item.get("event_key") or item.get("event_id")
            if ek:
                primary_by_event[ek] = item
        
        secondary_by_event = {}
        for item in secondary_data:
            ek = item.get("event_key") or item.get("event_id")
            if ek not in secondary_by_event:
                secondary_by_event[ek] = []
            secondary_by_event[ek].append(item)
        
        # Compare pairs
        for event_key, primary in primary_by_event.items():
            if event_key not in secondary_by_event:
                continue
            
            primary_claims = primary.get("statements") or primary.get("claims", [])
            event_label = primary.get("event_label") or primary.get("event_name", event_key)
            primary_source = primary.get("source_key") or primary.get("source_id", "")
            
            for secondary in secondary_by_event[event_key]:
                secondary_claims = secondary.get("statements") or secondary.get("claims", [])
                secondary_source = secondary.get("source_key") or secondary.get("source_id", "")
                secondary_writer = secondary.get("writer") or secondary.get("author", "Unknown")
                
                print(f"  Comparing: {event_label} ({secondary_writer})")
                
                result = self.evaluate_single(
                    primary_claims, secondary_claims,
                    event_key, event_label,
                    primary_source, secondary_source, secondary_writer,
                    method=method
                )
                
                if result:
                    results.append(result)
                    print(f"    Score: {result.alignment_score}")
        
        return results
    
    def export_results(self, results: List[ComparisonResult], filename: str = "judgment_results.json"):
        """Save results to JSON."""
        output_file = self.output_path / filename
        
        data = []
        for r in results:
            entry = asdict(r)
            entry["discrepancies"] = [asdict(d) for d in r.discrepancies]
            data.append(entry)
        
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(results)} results to {output_file}")


if __name__ == "__main__":
    checker = ConsistencyEvaluator()
    primary = load_claims("data/extracted/loc_events.json")
    secondary = load_claims("data/extracted/gutenberg_events.json")
    results = checker.evaluate_pairs(primary, secondary)
    checker.export_results(results)
