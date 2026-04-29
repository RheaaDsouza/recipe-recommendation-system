# Recipe Recommendation System

A knowledge-graph-driven recipe assistant that recommends meals from pantry ingredients while prioritizing food waste reduction.

The system combines:
- a Neo4j knowledge graph of recipes and ingredients,
- shelf-life from FoodKeeper data,
- rule-based filtering for dietary preferences and allergies,
- and an LLM response layer to produce user-friendly recommendations.

## Project Overview

The main user-facing app (`app.py`) lets users:
- add pantry ingredients,
- specify dietary preferences and allergies,
- and ask for recipe recommendations in a chat interface.

Recommendations are ranked to:
1. maximize pantry usage,
2. prioritize ingredients expiring soon,
