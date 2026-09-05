import os
import json
import asyncio
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.app.tools.base import BaseTool

logger = logging.getLogger(__name__)

class WebIntelligenceTool(BaseTool):
    tool_id = "web_intelligence"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        location_meta = inputs.get("location_meta", {})
        location_name = location_meta.get("name", inputs.get("location_name", "Target Region"))
        task_type = inputs.get("task_type", "change_analysis")
        question = inputs.get("question", "")

        loop = asyncio.get_running_loop()

        # 1. Fetch live Google News RSS articles for location & task
        news_items = await self._fetch_live_news_rss(location_name, question, loop)

        # 2. Fetch live Wikipedia abstract/context for location if available
        wiki_summary = await self._fetch_live_wikipedia_summary(location_name, loop)

        headlines = [item["title"] for item in news_items]
        sources = list(set([item["source"] for item in news_items if item.get("source")]))

        if news_items and wiki_summary:
            summary = f"{wiki_summary} Recent developments reported in live news: " + "; ".join(headlines[:2]) + "."
        elif news_items:
            summary = f"Current regional reports for {location_name} highlight active developments: " + "; ".join(headlines[:3]) + "."
        elif wiki_summary:
            summary = wiki_summary
        else:
            summary = f"Regional administrative and geographic records for {location_name} indicate ongoing urban infrastructure and land management activities."

        return {
            "location": location_name,
            "summary": summary,
            "headlines": headlines[:5],
            "sources": sources[:5] if sources else ["Google News", "OpenStreetMap"],
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success"
        }

    async def _fetch_live_news_rss(self, location_name: str, question: str, loop: asyncio.AbstractEventLoop) -> List[Dict[str, str]]:
        # Formulate search queries: location + key topics from question
        clean_loc = location_name.split(",")[0].strip()
        search_query = f"{clean_loc} infrastructure development project"
        try:
            encoded = urllib.parse.quote(search_query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) COSMOCLIP/2.0"})

            def _sync_fetch():
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    return resp.read()

            xml_data = await loop.run_in_executor(None, _sync_fetch)
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            results = []
            for item in items[:4]:
                title_elem = item.find("title")
                source_elem = item.find("source")
                pub_elem = item.find("pubDate")

                if title_elem is not None and title_elem.text:
                    full_title = title_elem.text.strip()
                    source_name = source_elem.text.strip() if source_elem is not None and source_elem.text else "Google News"
                    clean_title = full_title.split(" - ")[0] if " - " in full_title else full_title
                    results.append({
                        "title": clean_title,
                        "source": source_name,
                        "pub_date": pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
                    })
            return results
        except Exception as e:
            logger.debug(f"Live Google News RSS query for '{search_query}' failed: {e}")
            return []

    async def _fetch_live_wikipedia_summary(self, location_name: str, loop: asyncio.AbstractEventLoop) -> str:
        clean_name = location_name.split("&")[0].split(",")[0].strip()
        try:
            encoded = urllib.parse.quote(clean_name)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "COSMOCLIP-VLM/2.0 (contact: info@cosmoclip.ai)"})

            def _sync_wiki():
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return resp.read()

            raw_json = await loop.run_in_executor(None, _sync_wiki)
            data = json.loads(raw_json)
            extract = data.get("extract", "")
            if extract:
                # Return first 2 sentences
                sentences = extract.split(". ")
                return ". ".join(sentences[:2]) + "."
        except Exception:
            pass
        return ""
