#!/usr/bin/env python3
"""
Check for broken links in documentation files.
"""

import re
import sys
import requests
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse


class LinkChecker:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.broken_links = []
        self.checked_urls = {}  # Cache for already checked URLs

    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in documentation."""
        return list(self.docs_dir.rglob("*.md"))

    def extract_links(self, content: str) -> List[str]:
        """Extract all links from markdown content."""
        # Markdown link pattern: [text](url)
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        # HTML link pattern: <a href="url">
        html_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>', content)

        links = [link[1] for link in md_links] + html_links
        return links

    def is_external_link(self, url: str) -> bool:
        """Check if URL is external."""
        parsed = urlparse(url)
        return bool(parsed.netloc)

    def check_link(self, url: str) -> Tuple[bool, str]:
        """Check if a link is accessible."""
        if url in self.checked_urls:
            return self.checked_urls[url]

        try:
            if self.is_external_link(url):
                response = requests.head(url, timeout=10, allow_redirects=True)
                success = response.status_code < 400
                message = f"HTTP {response.status_code}" if not success else "OK"
            else:
                # Internal link - check if file exists
                file_path = self.docs_dir / url.lstrip('/')
                success = file_path.exists()
                message = "File not found" if not success else "OK"

            self.checked_urls[url] = (success, message)
            return success, message

        except Exception as e:
            result = (False, str(e))
            self.checked_urls[url] = result
            return result

    def check_file(self, filepath: Path):
        """Check all links in a markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            links = self.extract_links(content)

            for link in links:
                # Skip certain types of links
                if link.startswith(('#', 'mailto:', 'javascript:')):
                    continue

                success, message = self.check_link(link)
                if not success:
                    self.broken_links.append((str(filepath), link, message))

        except Exception as e:
            print(f"Error checking {filepath}: {e}")

    def run(self) -> int:
        """Run link checking on all documentation files."""
        markdown_files = self.find_markdown_files()

        if not markdown_files:
            print("No markdown files found")
            return 0

        print(f"Checking links in {len(markdown_files)} files...")

        for md_file in markdown_files:
            print(f"Checking {md_file}...")
            self.check_file(md_file)

        # Report results
        if self.broken_links:
            print(f"\n❌ Found {len(self.broken_links)} broken links:")
            for filepath, link, message in self.broken_links:
                print(f"  {filepath}: {link} ({message})")
            return 1
        else:
            print(f"\n✅ All links are working!")
            return 0


if __name__ == "__main__":
    checker = LinkChecker()
    sys.exit(checker.run())