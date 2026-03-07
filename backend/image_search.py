import requests
import re

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Extensions that are actual images (not audio/video/SVG icons)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.svg')

# Patterns to exclude: logos, icons, small decorative images
EXCLUDE_PATTERNS = [
    'icon', 'logo', 'flag', 'button', 'arrow', 'star', 'bullet',
    'edit', 'question', 'commons', 'wikiquote', 'wikipedia', 'portail',
    'red_question', 'merge', 'move', 'disambig', 'globe', 'nuvola',
    'folder', 'sound', 'audio', 'video', 'wikidata', 'wiktionary',
    'map', 'locator', 'location', 'coa', 'coat', 'seal', 'stamp',
    'symbol', 'insignia', 'emblem', 'portrait', 'photo', 'head',
]

# General diagram-style keywords that boost any image's score
GENERAL_DIAGRAM_KEYWORDS = [
    'diagram', 'structure', 'cycle', 'process', 'model', 'system',
    'cell', 'circuit', 'anatomy', 'cross', 'section', 'schema',
    'chart', 'graph', 'illustration', 'figure', 'scheme', 'overview',
    'labeled', 'label', 'cross_section', 'schematic',
]


def _is_valid_diagram(filename: str) -> bool:
    """Check if a filename looks like an actual scientific diagram."""
    name_lower = filename.lower()
    # Must have a valid image extension
    if not any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return False
    # Must not be a known icon/logo/decoration
    if any(pat in name_lower for pat in EXCLUDE_PATTERNS):
        return False
    return True


def _get_image_info(filename: str) -> str | None:
    """
    Given a Wikipedia image filename (e.g. 'File:Photosynthesis.svg'),
    fetch the actual direct URL via the MediaWiki API.
    Only returns images larger than 200x200 pixels.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|size",
        "titles": filename,
    }
    headers = {"User-Agent": "NCERT-AI-Chatbot/1.0"}
    try:
        res = requests.get(WIKIPEDIA_API, headers=headers, params=params, timeout=8)
        data = res.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo", [])
            if infos:
                url = infos[0].get("url", "")
                width = infos[0].get("width", 0)
                height = infos[0].get("height", 0)
                # Skip tiny images (likely icons) — increased threshold
                if width > 200 and height > 200:
                    return url
    except Exception as e:
        print(f"Image info fetch error: {e}")
    return None


def _score_image(filename: str, concept_keywords: list[str]) -> int:
    """
    Score an image filename based on:
    - How many concept-specific keywords appear in it (high weight)
    - How many general diagram keywords appear in it (low weight)
    SVG/PNG gets a small bonus.
    """
    name_lower = filename.lower()
    concept_score = sum(3 for kw in concept_keywords if kw in name_lower)
    diagram_score = sum(1 for kw in GENERAL_DIAGRAM_KEYWORDS if kw in name_lower)
    format_bonus = 1 if name_lower.endswith('.svg') or name_lower.endswith('.png') else 0
    return concept_score + diagram_score + format_bonus


def _extract_concept_keywords(concept: str) -> list[str]:
    """
    Break a concept phrase into individual keywords for filename matching.
    e.g. "plant cell" -> ["plant", "cell", "plant_cell"]
    """
    if not concept:
        return []
    words = [w.lower() for w in concept.split() if len(w) > 2]
    # Also add the joined form (e.g. "plant_cell")
    if len(words) > 1:
        words.append("_".join(words))
    return words


def _search_wikipedia_article_images(article_title: str, concept_keywords: list[str]) -> str | None:
    """
    Fetch all images from a Wikipedia article and return the best diagram URL.
    Uses concept_keywords for targeted scoring to avoid wrong images.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "images",
        "titles": article_title,
        "imlimit": 30,
    }
    headers = {"User-Agent": "NCERT-AI-Chatbot/1.0"}
    try:
        res = requests.get(WIKIPEDIA_API, headers=headers, params=params, timeout=8)
        data = res.json()
        pages = data.get("query", {}).get("pages", {})

        candidates = []
        for page in pages.values():
            images = page.get("images", [])
            for img in images:
                fname = img.get("title", "")
                if _is_valid_diagram(fname):
                    candidates.append(fname)

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda f: _score_image(f, concept_keywords), reverse=True)

        # Only try top 5 to avoid slow API calls
        for fname in candidates[:5]:
            url = _get_image_info(fname)
            if url:
                print(f"[image_search] ✅ Best match '{fname}' (score={_score_image(fname, concept_keywords)})")
                return url

    except Exception as e:
        print(f"Article image search error: {e}")
    return None


def get_image_url(query: str, concept: str = "") -> str | None:
    """
    Searches Wikipedia for the best educational diagram matching `query`.
    Uses `concept` for targeted keyword scoring to reduce wrong images.

    Strategy:
    1. Use Wikipedia opensearch to find up to 3 candidate article titles.
    2. For each article, fetch images and pick the best match using concept keywords.
    3. Fallback: pageimages thumbnail from top search result.
    4. Final Fallback: Wikimedia Commons direct image search.
    """
    headers = {"User-Agent": "NCERT-AI-Chatbot/1.0"}
    concept_keywords = _extract_concept_keywords(concept if concept else query)

    # --- Step 1: Find best Wikipedia article(s) ---
    search_params = {
        "action": "opensearch",
        "format": "json",
        "search": query,
        "limit": 3,
        "namespace": 0,
    }
    article_titles = []
    try:
        res = requests.get(WIKIPEDIA_API, headers=headers, params=search_params, timeout=8)
        data = res.json()
        titles = data[1] if len(data) > 1 else []
        article_titles = titles[:3]
    except Exception as e:
        print(f"Wikipedia opensearch error: {e}")

    # --- Step 2: Try each article, pick best image via concept scoring ---
    for article_title in article_titles:
        best_url = _search_wikipedia_article_images(article_title, concept_keywords)
        if best_url:
            print(f"[image_search] Found diagram in article '{article_title}': {best_url}")
            return best_url

    # --- Step 3: Fallback – pageimages thumbnail from search ---
    fallback_params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "pithumbsize": 800,
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 5,
    }
    try:
        res = requests.get(WIKIPEDIA_API, headers=headers, params=fallback_params, timeout=8)
        data = res.json()
        pages = data.get("query", {}).get("pages", {})
        for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            thumb = page.get("thumbnail", {}).get("source", "")
            if thumb and not any(pat in thumb.lower() for pat in EXCLUDE_PATTERNS):
                # Scale up the thumbnail
                thumb = re.sub(r'/\d+px-', '/800px-', thumb)
                print(f"[image_search] Fallback thumbnail: {thumb}")
                return thumb
    except Exception as e:
        print(f"Wikipedia pageimages fallback error: {e}")

    # --- Step 4: Final fallback – Wikimedia Commons ---
    # Use exact concept for more targeted search
    commons_query = f"{concept} diagram" if concept else f"{query} diagram"
    commons_params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|size",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap|drawing {commons_query}",
        "gsrlimit": 8,
        "gsrnamespace": 6,
    }
    try:
        res = requests.get(WIKIMEDIA_API, headers=headers, params=commons_params, timeout=8)
        data = res.json()
        pages = data.get("query", {}).get("pages", {})
        
        # Sort by concept score
        scored = []
        for page in pages.values():
            title = page.get("title", "")
            if _is_valid_diagram(title):
                score = _score_image(title, concept_keywords)
                infos = page.get("imageinfo", [])
                if infos:
                    url = infos[0].get("url", "")
                    w = infos[0].get("width", 0)
                    h = infos[0].get("height", 0)
                    if url and w > 200 and h > 200:
                        scored.append((score, url))
        
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            print(f"[image_search] Wikimedia Commons best match: {best}")
            return best
            
    except Exception as e:
        print(f"Wikimedia Commons search error: {e}")

    print("[image_search] No suitable image found.")
    return None


if __name__ == "__main__":
    test_cases = [
        ("plant cell Biology diagram", "plant cell"),
        ("photosynthesis process diagram", "photosynthesis"),
        ("human heart anatomy", "human heart"),
        ("electric circuit diagram", "electric circuit"),
        ("water cycle diagram", "water cycle"),
        ("mitosis cell division diagram", "mitosis"),
    ]
    for q, concept in test_cases:
        url = get_image_url(q, concept=concept)
        print(f"Query: {q}\n  -> {url}\n")
