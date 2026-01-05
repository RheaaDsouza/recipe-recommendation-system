import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
import ast
import re
from fuzzywuzzy import fuzz, process

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Load datasets
recipes_df = pd.read_csv('./data/recipe_nlp_1000.csv')
recipes_df['id'] = recipes_df['id'].astype(str)

shelf_life_df = pd.read_csv('./data_processed/foodkeeper_shelf_life_processed.csv')
shelf_life_df['Ingredient'] = shelf_life_df['Ingredient'].astype(str).str.lower().str.strip()

def find_best_match(ingredient, choices, threshold):
    """Find best fuzzy match between recipe ingredient and ingredients from shelf life dataset"""
    if not ingredient:
        return None

    matches = process.extract(ingredient, choices, limit=5, scorer=fuzz.token_sort_ratio)
    
    for match, score in matches:
        if score >= threshold:
            return match
    
    return None

def find_shelf_life_data(ner_ingredient, shelf_life_df, foodkeeper_choices):
    clean_ingredient = ner_ingredient.lower().strip()
    
    # Get the base form of ingredients
    if clean_ingredient.endswith('s'):
        base_form = clean_ingredient[:-1]
    else:
        base_form = clean_ingredient
    
    # Try multiple variations
    variations = [
        clean_ingredient,
        base_form,
        f"{clean_ingredient}es",
        f"{base_form}es"
    ]
    
    for var in variations:
        exact_match = shelf_life_df[shelf_life_df['Ingredient'] == var]
        if not exact_match.empty:
            return {
                'matched_ingredient': var,
                'category': exact_match.iloc[0]['Category'],
                'shelf_life': exact_match.iloc[0]['Shelf_Life'],
                'match_found': True
            }
    
    # Fuzzy matching
    best_match = find_best_match(clean_ingredient, foodkeeper_choices, threshold=70)  # Higher threshold
    
    if best_match:
        shelf_data = shelf_life_df[shelf_life_df['Ingredient'] == best_match]
        if not shelf_data.empty:
            return {
                'matched_ingredient': best_match,
                'category': shelf_data.iloc[0]['Category'],
                'shelf_life': shelf_data.iloc[0]['Shelf_Life'],
                'match_found': True
            }
    
    return {
        'matched_ingredient': None,
        'category': 'Unknown',
        'shelf_life': 'Unknown',
        'match_found': False
    }

# Preprocess FoodKeeper dataset
print("Preprocessing FoodKeeper ingredients...")
shelf_life_df['cleaned_ingredient'] = shelf_life_df['Ingredient'].str.lower().str.strip()
foodkeeper_choices = shelf_life_df['cleaned_ingredient'].unique()

def create_knowledge_graph_correct(tx, recipe):
    # Create Recipe node
    tx.run("""
    MERGE (r:Recipe {id: $id})
    SET r.title = $title,
        r.source = $source,
        r.link = $link,
        r.directions = $directions
    """, 
    id=recipe['id'],
    title=recipe.get('title', ''),
    source=recipe.get('source', ''),
    link=recipe.get('link', ''),
    directions=ast.literal_eval(recipe['directions']) if isinstance(recipe['directions'], str) else recipe['directions']
    )

    ner_ingredients = ast.literal_eval(recipe['NER']) if isinstance(recipe['NER'], str) else recipe['NER']
    
    for ner_ing in ner_ingredients:
        ner_ing_clean = ner_ing.strip().lower()
        if not ner_ing_clean:
            continue
            
        shelf_data = find_shelf_life_data(ner_ing_clean, shelf_life_df, foodkeeper_choices)
        
        if shelf_data['match_found'] and shelf_data['shelf_life'] not in ['Unknown', 'None', None]:
            tx.run("""
            MATCH (r:Recipe {id: $recipe_id})
            MERGE (i:Ingredient {name: $name})
            SET i.category = $category,
                i.shelf_life = $shelf_life,
                i.source = 'FoodKeeper'
            MERGE (r)-[u:USES]->(i)
            """,
            recipe_id=recipe['id'],
            name=shelf_data['matched_ingredient'],
            category=shelf_data['category'],
            shelf_life=shelf_data['shelf_life']
            )
        else:
            print(f"Skip '{ner_ing_clean}': No shelf life data found")

# Function to create Knowledge graph 
def create_knowledge_graph(tx, recipe):
    
    # Create Recipe node with directions
    directions = ast.literal_eval(recipe['directions']) if isinstance(recipe['directions'], str) else recipe['directions']

    # Create Recipe node
    tx.run("""
    MERGE (r:Recipe {id: $id})
    SET r.title = $title,
        r.source = $source,
        r.link = $link,
        r.directions = $directions
    """, 
    id=recipe['id'],
    title=recipe.get('title', ''),
    source=recipe.get('source', ''),
    link=recipe.get('link', ''),
    directions=directions
    )

    # Use NER ingredients instead of raw ingredients as they are cleaner 
    ner_ingredients = ast.literal_eval(recipe['NER']) if isinstance(recipe['NER'], str) else recipe['NER']
    
    for ner_ing in ner_ingredients:
        ner_ing = ner_ing.strip()
        if not ner_ing:
            continue
            
        # Use shelf life data
        shelf_data = find_shelf_life_data(ner_ing, shelf_life_df, foodkeeper_choices)
        
        # Use matched name if found, otherwise use NER name directly
        ingredient_name = shelf_data['matched_ingredient'] if shelf_data['match_found'] else ner_ing.lower()
    
        # Create Ingredient node with shelf life
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        MERGE (i:Ingredient {name: $name})
        SET i.category = $category,
            i.shelf_life = $shelf_life,
            i.source = $source,
            i.ner_original = $ner_original
        MERGE (r)-[u:USES]->(i)
        """,
        recipe_id=recipe['id'],
        name=ingredient_name,
        category=shelf_data['category'],
        shelf_life=shelf_data['shelf_life'],
        source='FoodKeeper' if shelf_data['match_found'] else 'RecipeNLG',
        ner_original=ner_ing
        )

def main():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception as e:
        print(f"Failed to create Neo4j driver: {e}")
        return

    response = input("\nProceed with creating knowledge graph? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return

    print("\nCreating knowledge graph...")
    
    with driver.session() as session:
        for i, recipe in recipes_df.iterrows():
            try:
                session.execute_write(create_knowledge_graph, recipe)
                # if (i + 1) % 100 == 0:
                #     print(f"Processed {i + 1} recipes...")
            except Exception as e:
                print(f"Error with recipe {recipe['id']}: {str(e)}")
                continue

    # Create indexes
    with driver.session() as session:
        session.run("CREATE INDEX recipe_id_index IF NOT EXISTS FOR (r:Recipe) ON (r.id)")
        session.run("CREATE INDEX ingredient_name_index IF NOT EXISTS FOR (i:Ingredient) ON (i.name)")

    try:
        if driver is not None:
            driver.close()
    except Exception:
        pass
    
    print("Knowledge graph created successfully!")

if __name__ == "__main__":
    main()