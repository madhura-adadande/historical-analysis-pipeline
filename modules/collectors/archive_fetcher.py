"""
Archive Fetcher - Downloads documents from Library of Congress

Retrieves primary source documents for analysis.
"""

import re
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from bs4 import BeautifulSoup

# Target documents
SOURCE_DOCUMENTS = [
    {
        "doc_id": "election_night_1860",
        "name": "Letter about Election Night 1860",
        "link": "https://www.loc.gov/item/mal0440500/",
        "resource": "https://www.loc.gov/resource/mal.0440500/",
        "doc_type": "Letter",
        "written": "November 1860"
    },
    {
        "doc_id": "fort_sumter_decision",
        "name": "Fort Sumter Decision",
        "link": "https://www.loc.gov/resource/mal.0882800",
        "resource": "https://www.loc.gov/resource/mal.0882800/",
        "doc_type": "Note",
        "written": "April 1861"
    },
    {
        "doc_id": "gettysburg_address",
        "name": "Gettysburg Address (Nicolay Copy)",
        "link": "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
        "doc_type": "Speech",
        "written": "November 19, 1863",
        "loc": "Gettysburg, Pennsylvania"
    },
    {
        "doc_id": "second_inaugural",
        "name": "Second Inaugural Address",
        "link": "https://www.loc.gov/resource/mal.4361300",
        "resource": "https://www.loc.gov/resource/mal.4361300/",
        "doc_type": "Speech",
        "written": "March 4, 1865",
        "loc": "Washington, D.C."
    },
    {
        "doc_id": "last_public_address",
        "name": "Last Public Address",
        "link": "https://www.loc.gov/resource/mal.4361800/",
        "resource": "https://www.loc.gov/resource/mal.4361800/",
        "doc_type": "Speech",
        "written": "April 11, 1865",
        "loc": "Washington, D.C."
    },
]


@dataclass
class ArchiveRecord:
    """Stores archive document data."""
    doc_id: str
    name: str
    link: str
    doc_type: str
    written: str = ""
    loc: str = ""
    author: str = "Abraham Lincoln"
    recipient: str = ""
    body: str = ""


class ArchiveCollector:
    """Handles document retrieval from Congress Library."""
    
    def __init__(self, save_path: str = "data/raw/loc"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.http = requests.Session()
        self.http.headers.update({
            "User-Agent": "Mozilla/5.0 Research-Bot/1.0",
            "Accept": "application/json, text/html, */*"
        })
    
    def _get_json(self, url: str) -> Optional[Dict]:
        """Request JSON from API endpoint."""
        api_url = url.rstrip("/") + "/?fo=json"
        try:
            resp = self.http.get(api_url, timeout=30)
            if resp.ok and "application/json" in resp.headers.get("content-type", ""):
                return resp.json()
        except Exception:
            pass
        return None
    
    def _get_html(self, url: str) -> Optional[str]:
        """Request HTML page content."""
        try:
            resp = self.http.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except:
            return None
    
    def _parse_json_content(self, data: Dict) -> str:
        """Extract text from JSON response structure."""
        parts = []
        
        if "text" in data:
            return data["text"]
        
        if "resource" in data and isinstance(data["resource"], dict):
            if "text" in data["resource"]:
                return data["resource"]["text"]
        
        if "item" in data and isinstance(data["item"], dict):
            item = data["item"]
            if "notes" in item:
                parts.extend(item["notes"])
            if "contents" in item:
                parts.append(item["contents"])
        
        if "content" in data:
            c = data["content"]
            if isinstance(c, str):
                return c
            elif isinstance(c, list):
                parts.extend([str(x) for x in c])
        
        return "\n".join(parts)
    
    def _parse_html_content(self, html: str) -> str:
        """Extract text from HTML page."""
        soup = BeautifulSoup(html, "lxml")
        
        for selector in ["div.content", "div.main-content", "article", "main"]:
            elem = soup.select_one(selector)
            if elem:
                txt = elem.get_text(separator="\n", strip=True)
                if len(txt) > 100:
                    return txt
        
        body = soup.find("body")
        if body:
            for tag in body.find_all(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            return body.get_text(separator="\n", strip=True)
        
        return ""
    
    def retrieve_content(self, doc_meta: Dict) -> Optional[str]:
        """Get document content using best available method."""
        url = doc_meta.get("link", "")
        resource = doc_meta.get("resource", url)
        
        # Handle exhibit pages specially
        if "/exhibits/" in url:
            html = self._get_html(url)
            if html:
                cache = self.save_path / f"{doc_meta['doc_id']}_raw.html"
                cache.write_text(html, encoding="utf-8")
                return self._parse_html_content(html)
        
        # Try JSON first
        json_data = self._get_json(resource)
        if json_data:
            cache = self.save_path / f"{doc_meta['doc_id']}_raw.json"
            cache.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            text = self._parse_json_content(json_data)
            if text:
                return text
        
        # Fallback to HTML
        html = self._get_html(url)
        if html:
            return self._parse_html_content(html)
        
        return None
    
    def fetch_single(self, doc_meta: Dict) -> Optional[ArchiveRecord]:
        """Fetch and process one document."""
        content = self.retrieve_content(doc_meta)
        
        if not content:
            print(f"  [-] Failed: {doc_meta['name']}")
            return None
        
        print(f"  [+] {doc_meta['name']}: {len(content):,} chars")
        
        return ArchiveRecord(
            doc_id=f"loc_{doc_meta['doc_id']}",
            name=doc_meta["name"],
            link=doc_meta["link"],
            doc_type=doc_meta["doc_type"],
            written=doc_meta.get("written", ""),
            loc=doc_meta.get("loc", ""),
            author="Abraham Lincoln",
            body=content
        )
    
    def fetch_all_documents(self, wait_time: float = 2.0) -> List[ArchiveRecord]:
        """Fetch all configured documents with rate limiting."""
        results = []
        
        print(f"\nFetching {len(SOURCE_DOCUMENTS)} documents from LoC...")
        
        for meta in SOURCE_DOCUMENTS:
            record = self.fetch_single(meta)
            if record:
                results.append(record)
            time.sleep(wait_time)
        
        print(f"\nCompleted: {len(results)}/{len(SOURCE_DOCUMENTS)} documents")
        return results
    
    def export_dataset(self, records: List[ArchiveRecord], output_file: str):
        """Export to normalized JSON."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for rec in records:
            entry = {
                "id": rec.doc_id,
                "title": rec.name,
                "reference": rec.link,
                "document_type": rec.doc_type,
                "date": rec.written,
                "place": rec.loc,
                "from": rec.author,
                "to": rec.recipient,
                "content": rec.body
            }
            data.append(entry)
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(data)} records to {out_path}")


if __name__ == "__main__":
    collector = ArchiveCollector()
    docs = collector.fetch_all_documents()
    if docs:
        collector.export_dataset(docs, "data/processed/loc.json")
