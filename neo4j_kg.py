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

# This function helps find best fuzzy match between recipe ingredient and 
# ingredients from shelf life dataset
def find_best_match(ingredient, choices, threshold):
    if not ingredient:
        return None

    matches = process.extract(ingredient, choices, limit=5, scorer=fuzz.token_sort_ratio)
    
    for match, score in matches:
        if score >= threshold:
            return match
    
    return None

# Function to turn the string shelf life data ("3-4 Days") into something like 3.5
def parse_to_days(shelf_life_str):
    if not shelf_life_str or shelf_life_str == 'Unknown':
        return 999 # Treat unknown as long-lasting
    try:
        # Simple regex to find numbers
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", shelf_life_str)
        if not nums: return 999
        avg_val = sum(float(n) for n in nums) / len(nums)
        
        if 'week' in shelf_life_str.lower(): return avg_val * 7
        if 'month' in shelf_life_str.lower(): return avg_val * 30
        if 'year' in shelf_life_str.lower(): return avg_val * 365
        return avg_val # assume days
    except:
        return 999
    
# This function helps find shelf data from foodkeeper 
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
                'days': parse_to_days(exact_match.iloc[0]['Shelf_Life']),
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
                'days': parse_to_days(shelf_data.iloc[0]['Shelf_Life']),
                'match_found': True
            }
    
    return {
        'matched_ingredient': None,
        'category': 'Unknown',
        'shelf_life': 'Unknown',
        'days': 999,
        'match_found': False
    }

# Preprocess FoodKeeper dataset
print("Preprocessing FoodKeeper ingredients...")
shelf_life_df['cleaned_ingredient'] = shelf_life_df['Ingredient'].str.lower().str.strip()
foodkeeper_choices = shelf_life_df['cleaned_ingredient'].unique()


def setup(tx):
    """Creates constraints to ensure speed and prevent duplicates"""
    tx.run("CREATE CONSTRAINT recipe_id_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")

def create_knowledge_graph(tx, recipe):
    # Create Recipe node with directions
    directions = ast.literal_eval(recipe['directions']) if isinstance(recipe['directions'], str) else recipe['directions']
    
    tx.run("""
    MERGE (r:Recipe {id: $id})
    SET r.title = $title,
        r.directions = $directions
    """, id=recipe['id'], title=recipe.get('title', ''), directions=directions)

    ner_ingredients = ast.literal_eval(recipe['NER']) if isinstance(recipe['NER'], str) else recipe['NER']
    
    for ner_ing in ner_ingredients:
        ner_ing = ner_ing.strip().lower()
        shelf_data = find_shelf_life_data(ner_ing, shelf_life_df, foodkeeper_choices)
        
        # Parse shelf life to a number 
        days = parse_to_days(shelf_data['shelf_life'])

        # Create Ingredient and link to Recipe and 
        # Create Category Node and link to Ingredient
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        
        // Merge the Ingredient
        MERGE (i:Ingredient {name: $name})
        SET i.shelf_life = $shelf_life,
            i.shelf_life_days = $days
            
        // Merge the Category as a separate Node
        MERGE (c:Category {name: $category})
        
        // Create Relationships
        MERGE (r)-[:USES]->(i)
        MERGE (i)-[:BELONGS_TO]->(c)
        """,
        recipe_id=recipe['id'],
        name=shelf_data['matched_ingredient'] if shelf_data['match_found'] else ner_ing,
        category=shelf_data['category'] if shelf_data['match_found'] else "Uncategorized",
        shelf_life=shelf_data['shelf_life'],
        days=days
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
        session.run("CREATE CONSTRAINT recipe_id_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.id IS UNIQUE")
        session.run("CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
    try:
        if driver is not None:
            driver.close()
    except Exception:
        pass
    
    print("Knowledge graph created successfully!")

if __name__ == "__main__":
    main()