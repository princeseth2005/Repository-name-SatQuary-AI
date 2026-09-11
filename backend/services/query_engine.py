import re
from typing import Tuple, Dict, Any

INTENT_KEYWORDS = {
    "VEGETATION": [
        "vegetation", "greenery", "forest", "crop", "crops", "agriculture", "plants",
        "trees", "canopy", "agricultural", "farm", "farming", "vari", "ndvi", "foliage",
        "vegetative", "green"
    ],
    "WATER": [
        "water", "river", "lake", "ocean", "sea", "pond", "reservoir", "waterbody",
        "waterbodies", "canal", "stream", "estuary", "lagoon", "bay", "coastal", "ndwi",
        "wetland", "marsh"
    ],
    "BUILT_UP_AREA": [
        "built-up", "built up", "urban", "city", "building", "buildings", "houses",
        "residential", "commercial", "industrial", "concrete", "infrastructure",
        "roof", "rooftops", "settlement", "construction", "town", "metropolis"
    ],
    "ROAD": [
        "road", "roads", "highway", "highways", "street", "streets", "path", "pathway",
        "corridor", "arterial", "expressway", "track", "linear"
    ],
    "LAND_COVER": [
        "land cover", "landcover", "classification", "classes", "types", "major types",
        "surface types", "terrain", "breakdown", "distribution", "barren", "soil"
    ],
    "STATISTICS": [
        "percentage", "percent", "how much", "statistics", "stats", "proportion",
        "metric", "metrics", "ratio", "breakdown", "quantity", "quantify"
    ],
    "CHANGE_DETECTION": [
        "change", "changes", "difference", "differ", "before and after", "temporal",
        "evolution", "expanded", "decreased", "increased", "deforestation", "growth",
        "altered", "variation"
    ],
    "IMAGE_COMPARISON": [
        "compare", "comparison", "versus", "vs", "two images", "both images",
        "side by side", "pair"
    ],
    "OBJECT_DETECTION": [
        "detect", "objects", "where are", "locate", "boundaries", "bounding box",
        "features", "clusters", "shapes", "find"
    ],
    "METADATA": [
        "metadata", "dimensions", "resolution", "pixel", "size", "crs", "coordinates",
        "latitude", "longitude", "geotiff", "sensor", "satellite", "file size", "format"
    ],
    "SUMMARY": [
        "summarize", "summary", "overview", "brief", "digest", "conclude", "recap"
    ],
    "GENERAL_DESCRIPTION": [
        "what is visible", "describe", "description", "explain", "what do you see",
        "scene", "look like", "tell me about", "what is this"
    ]
}


def classify_query_intent(query: str) -> Tuple[str, float]:
    """
    Classifies natural language remote sensing query into one of the designated intents
    using weighted keyword matching, phrase regex, and contextual rules.

    Returns:
        (intent_name, confidence_score)
    """
    if not query or not query.strip():
        return "GENERAL_DESCRIPTION", 0.5

    clean_query = query.lower().strip()

    # 1. Regex Exact Phrase Check
    if re.search(r"\b(compare|comparison|versus|diff between)\b", clean_query):
        if re.search(r"\b(change|changes|deforestation|growth|temporal)\b", clean_query):
            return "CHANGE_DETECTION", 0.95
        return "IMAGE_COMPARISON", 0.92

    if re.search(r"\b(change|changed|changes|temporal change|increased or decreased)\b", clean_query):
        return "CHANGE_DETECTION", 0.94

    if re.search(r"\b(how much|percentage|percent|proportion of)\s+(vegetation|green|forest|crops)\b", clean_query):
        return "VEGETATION", 0.98

    if re.search(r"\b(how much|percentage|percent|proportion of)\s+(water|river|lake)\b", clean_query):
        return "WATER", 0.98

    if re.search(r"\b(how much|percentage|percent|proportion of)\s+(urban|built-up|built up|buildings)\b", clean_query):
        return "BUILT_UP_AREA", 0.98

    if re.search(r"\b(metadata|dimensions|resolution|coordinates|lat|lon|crs)\b", clean_query):
        return "METADATA", 0.95

    if re.search(r"\b(summarize|summary|overview)\b", clean_query):
        return "SUMMARY", 0.92

    # 2. Weighted Keyword Scoring
    scores: Dict[str, float] = {}
    tokens = set(re.findall(r"\w+", clean_query))

    for intent, kws in INTENT_KEYWORDS.items():
        score = 0.0
        for kw in kws:
            if " " in kw:
                if kw in clean_query:
                    score += 2.5
            elif kw in tokens:
                score += 1.2
        if score > 0:
            scores[intent] = score

    if not scores:
        return "GENERAL_DESCRIPTION", 0.65

    # Pick top scoring intent
    best_intent = max(scores.items(), key=lambda x: x[1])

    # If general intent or object detection scored, but a specific target exists, prefer specific
    if best_intent[0] in ["GENERAL_DESCRIPTION", "SUMMARY", "STATISTICS", "OBJECT_DETECTION"]:
        for specific in ["WATER", "VEGETATION", "BUILT_UP_AREA", "ROAD", "LAND_COVER"]:
            if specific in scores and scores[specific] >= (best_intent[1] * 0.4):
                return specific, 0.92

    confidence = min(0.98, 0.65 + (best_intent[1] * 0.1))
    return best_intent[0], round(confidence, 2)
