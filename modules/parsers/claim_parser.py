"""
Claim Parser - Extracts structured claims from historical texts using LLM

Identifies key historical events and extracts factual claims about them.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Historical events to identify
HISTORICAL_EVENTS = {
    "election_night_1860": {
        "label": "Election Night 1860",
        "desc": "Presidential election of November 6, 1860",
        "terms": ["election", "november 1860", "voted", "elected", "president", "republican"],
        "period": "November 1860"
    },
    "fort_sumter_decision": {
        "label": "Fort Sumter Decision",
        "desc": "Decision to resupply Fort Sumter, triggering Civil War",
        "terms": ["sumter", "fort", "charleston", "resupply", "anderson", "april 1861"],
        "period": "March-April 1861"
    },
    "gettysburg_address": {
        "label": "Gettysburg Address",
        "desc": "Famous cemetery dedication speech",
        "terms": ["gettysburg", "cemetery", "fourscore", "four score", "november 1863"],
        "period": "November 19, 1863"
    },
    "second_inaugural_address": {
        "label": "Second Inaugural Address",
        "desc": "Second presidential inaugural with 'malice toward none'",
        "terms": ["inaugural", "second", "malice", "charity", "march 1865"],
        "period": "March 4, 1865"
    },
    "fords_theatre_assassination": {
        "label": "Ford's Theatre Assassination",
        "desc": "Assassination at Ford's Theatre",
        "terms": ["ford", "theatre", "theater", "booth", "assassin", "shot", "april 1865"],
        "period": "April 14-15, 1865"
    }
}


@dataclass
class ParsedClaim:
    """Container for extracted event information."""
    event_key: str
    event_label: str
    source_key: str
    source_name: str
    writer: str
    statements: List[str]
    timing: Dict[str, str]
    sentiment: str
    quotes: List[str]
    certainty: str


class ClaimExtractor:
    """Parses documents to extract claims about historical events."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = OpenAI()
        self.model_name = llm_model
        self.output_path = Path("data/extracted")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _split_document(self, content: str, max_len: int = 12000, overlap: int = 1000) -> List[str]:
        """Divide long documents into processable segments."""
        if len(content) <= max_len:
            return [content]
        
        segments = []
        pos = 0
        while pos < len(content):
            end = min(pos + max_len, len(content))
            if end < len(content):
                # Find sentence boundary
                for delim in [". ", ".\n", "! ", "? "]:
                    last = content.rfind(delim, pos, end)
                    if last != -1:
                        end = last + 1
                        break
            segments.append(content[pos:end])
            pos = end - overlap if end < len(content) else end
        return segments
    
    def _check_relevance(self, segment: str, event_key: str) -> bool:
        """Quick check if segment might contain event information."""
        event_data = HISTORICAL_EVENTS.get(event_key, {})
        terms = event_data.get("terms", [])
        segment_lower = segment.lower()
        return any(t in segment_lower for t in terms)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _query_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send prompt to LLM and parse JSON response."""
        try:
            resp = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a historical text analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as err:
            print(f"  LLM error: {err}")
            return None
    
    def _build_prompt(self, segment: str, event_key: str) -> str:
        """Create extraction prompt for segment."""
        event_data = HISTORICAL_EVENTS[event_key]
        return f'''Analyze this historical text for information about: {event_data["label"]}
Event description: {event_data["desc"]}
Time period: {event_data["period"]}

TEXT:
{segment[:10000]}

Extract information as JSON:
{{
  "found": true/false,
  "statements": ["factual claim 1", "claim 2", ...],
  "timing": {{"date": "...", "time": "...", "sequence": "..."}},
  "sentiment": "reverential/neutral/critical/dramatic",
  "quotes": ["direct quote 1", ...],
  "certainty": "high/medium/low"
}}

If no relevant info, return {{"found": false}}'''
    
    def parse_document(self, doc: Dict, source_type: str) -> List[ParsedClaim]:
        """Process single document for all events."""
        results = []
        doc_id = doc.get("id", "unknown")
        doc_title = doc.get("title", "Unknown")
        writer = doc.get("from", "Unknown")
        content = doc.get("content", "")
        
        if not content or len(content) < 100:
            return results
        
        segments = self._split_document(content)
        print(f"  Processing: {doc_title} ({len(segments)} segments)")
        
        for event_key, event_data in HISTORICAL_EVENTS.items():
            event_statements = []
            event_timing = {}
            event_sentiment = "neutral"
            event_quotes = []
            event_certainty = "low"
            
            for seg in segments:
                if not self._check_relevance(seg, event_key):
                    continue
                
                prompt = self._build_prompt(seg, event_key)
                result = self._query_llm(prompt)
                
                if result and result.get("found"):
                    event_statements.extend(result.get("statements", []))
                    if result.get("timing"):
                        event_timing.update(result["timing"])
                    if result.get("sentiment"):
                        event_sentiment = result["sentiment"]
                    event_quotes.extend(result.get("quotes", []))
                    if result.get("certainty") == "high":
                        event_certainty = "high"
                    elif result.get("certainty") == "medium" and event_certainty != "high":
                        event_certainty = "medium"
            
            if event_statements:
                claim = ParsedClaim(
                    event_key=event_key,
                    event_label=event_data["label"],
                    source_key=doc_id,
                    source_name=doc_title,
                    writer=writer,
                    statements=list(set(event_statements)),
                    timing=event_timing,
                    sentiment=event_sentiment,
                    quotes=list(set(event_quotes)),
                    certainty=event_certainty
                )
                results.append(claim)
                print(f"    Found: {event_data['label']} ({len(event_statements)} claims)")
        
        return results
    
    def extract_from_file(self, input_file: str, dataset_name: str):
        """Process entire dataset file."""
        with open(input_file, "r", encoding="utf-8") as fp:
            documents = json.load(fp)
        
        print(f"\nProcessing {dataset_name}: {len(documents)} documents")
        
        all_claims = []
        for doc in documents:
            claims = self.parse_document(doc, dataset_name)
            all_claims.extend(claims)
        
        # Export results
        output_file = self.output_path / f"{dataset_name}_events.json"
        output_data = [asdict(c) for c in all_claims]
        
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(all_claims)} claims to {output_file}")


if __name__ == "__main__":
    extractor = ClaimExtractor()
    extractor.extract_from_file("data/processed/gutenberg.json", "gutenberg")
    extractor.extract_from_file("data/processed/loc.json", "loc")
