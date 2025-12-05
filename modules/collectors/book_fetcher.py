"""
Book Fetcher - Downloads texts from Project Gutenberg

Handles fetching, parsing, and normalizing book content from the public domain archive.
"""

import re
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

# Target books for analysis
TARGET_BOOKS = [
    {"book_id": "6812", "source_url": "https://www.gutenberg.org/ebooks/6812"},
    {"book_id": "6811", "source_url": "https://www.gutenberg.org/ebooks/6811"},
    {"book_id": "12801", "source_url": "https://www.gutenberg.org/ebooks/12801"},
    {"book_id": "14004", "source_url": "https://www.gutenberg.org/ebooks/14004"},
    {"book_id": "18379", "source_url": "https://www.gutenberg.org/ebooks/18379"},
]


@dataclass
class BookRecord:
    """Container for book data."""
    book_id: str
    name: str
    writer: str
    source_link: str
    category: str = "Biography"
    pub_date: str = ""
    location: str = ""
    text: str = ""


class BookCollector:
    """Fetches and processes books from Gutenberg archive."""
    
    TEXT_URL_PATTERN = "https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
    
    def __init__(self, save_path: str = "data/raw/gutenberg"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.http = requests.Session()
        self.http.headers.update({
            "User-Agent": "Mozilla/5.0 Research-Bot/1.0"
        })
    
    def _build_download_url(self, bid: str) -> str:
        """Generate download URL for book ID."""
        return self.TEXT_URL_PATTERN.format(bid=bid)
    
    def fetch_single_book(self, bid: str) -> Optional[str]:
        """Download raw text for one book."""
        url = self._build_download_url(bid)
        
        try:
            resp = self.http.get(url, timeout=30)
            resp.raise_for_status()
            
            # Cache raw file
            cache_file = self.save_path / f"{bid}_raw.txt"
            cache_file.write_text(resp.text, encoding="utf-8")
            print(f"  [+] Book {bid}: {len(resp.text):,} characters")
            
            return resp.text
            
        except requests.RequestException as err:
            print(f"  [-] Book {bid} failed: {err}")
            return None
    
    def _parse_header(self, raw: str) -> Dict[str, str]:
        """Extract metadata from Gutenberg header section."""
        info = {"name": "Unknown", "writer": "Unknown", "pub_date": "", "lang": "English"}
        
        # Title extraction
        m = re.search(r"Title:\s*(.+?)(?:\r?\n|$)", raw)
        if m:
            info["name"] = m.group(1).strip()
        
        # Author extraction
        m = re.search(r"Author:\s*(.+?)(?:\r?\n|$)", raw)
        if m:
            info["writer"] = m.group(1).strip()
        
        # Date extraction
        m = re.search(r"Release [Dd]ate:\s*(.+?)(?:\[|$|\r?\n)", raw)
        if m:
            info["pub_date"] = m.group(1).strip()
        
        return info
    
    def _remove_wrapper(self, raw: str) -> str:
        """Strip Gutenberg boilerplate from content."""
        begin_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG",
        ]
        
        finish_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]
        
        start_pos = 0
        end_pos = len(raw)
        
        for marker in begin_markers:
            if marker in raw:
                idx = raw.find(marker)
                nl = raw.find("\n", idx)
                if nl != -1:
                    start_pos = nl + 1
                break
        
        for marker in finish_markers:
            if marker in raw:
                end_pos = raw.find(marker)
                break
        
        return raw[start_pos:end_pos].strip()
    
    def process_single(self, bid: str) -> Optional[BookRecord]:
        """Fetch and process one book completely."""
        raw = self.fetch_single_book(bid)
        if not raw:
            return None
        
        info = self._parse_header(raw)
        clean_text = self._remove_wrapper(raw)
        
        record = BookRecord(
            book_id=f"gutenberg_{bid}",
            name=info["name"],
            writer=info["writer"],
            source_link=f"https://www.gutenberg.org/ebooks/{bid}",
            category="Biography",
            pub_date=info["pub_date"],
            text=clean_text
        )
        
        return record
    
    def fetch_all_books(self, wait_time: float = 1.0) -> List[BookRecord]:
        """Fetch all target books with rate limiting."""
        results = []
        
        print(f"\nFetching {len(TARGET_BOOKS)} books from Gutenberg...")
        
        for item in TARGET_BOOKS:
            bid = item["book_id"]
            print(f"\nProcessing: {bid}")
            
            record = self.process_single(bid)
            if record:
                results.append(record)
                print(f"  Title: {record.name}")
                print(f"  Writer: {record.writer}")
            
            time.sleep(wait_time)
        
        print(f"\nCompleted: {len(results)}/{len(TARGET_BOOKS)} books")
        return results
    
    def export_dataset(self, records: List[BookRecord], output_file: str):
        """Export records to normalized JSON format."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for rec in records:
            entry = {
                "id": rec.book_id,
                "title": rec.name,
                "reference": rec.source_link,
                "document_type": rec.category,
                "date": rec.pub_date,
                "place": rec.location,
                "from": rec.writer,
                "to": "",
                "content": rec.text
            }
            data.append(entry)
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(data)} records to {out_path}")


if __name__ == "__main__":
    collector = BookCollector()
    books = collector.fetch_all_books()
    if books:
        collector.export_dataset(books, "data/processed/gutenberg.json")
