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
2. prioritize ingredients expiring soon


## Setup

### Prerequisites

- Python 3.10+ recommended
- Neo4j (AuraDB)
- Replicate API token

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env`. Use the `env.example` file as template.


## Build the Neo4j Knowledge Graph

Run:

```bash
python3 neo4j_kg.py
```

## Run the Main Application

```bash
streamlit run app.py
```

Then open the URL shown by Streamlit in your browser.

## Evaluation / Testing

### Manual evaluation scripts

```bash
python3 tests/manual_tests/constraint.py
python3 tests/manual_tests/shelf_life.py
```

These scripts evaluate scenarios and export results to Excel files.

### Deepeval-based evaluation scripts

```bash
cd tests/deepeval_tests
python test_constraints.py
python test_shelf_life.py
```
