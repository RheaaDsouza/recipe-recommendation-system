import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
import ast
from fuzzywuzzy import fuzz
from langchain.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load RecipeNLG dataset
recipes_df = pd.read_csv('./data/recipe_nlp_1000.csv')
recipes_df['id'] = recipes_df['id'].astype(str)  # Ensure ID is string

# Load Shelf Life dataset
shelf_life_df = pd.read_csv('./data_processed/foodkeeper_shelf_life_processed.csv')
shelf_life_df['Ingredient'] = shelf_life_df['Ingredient'].str.lower().str.strip()


def find_shelf_life(ingredient):
    """Finds the best shelf life match using exact + fuzzy matching"""
    ingredient = ingredient.lower().strip()
    
    # Proper empty check
    exact_match = shelf_life_df[shelf_life_df['Ingredient'] == ingredient]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Work on a copy to avoid modifying original DataFrame
    shelf_life_copy = shelf_life_df.copy()
    shelf_life_copy['similarity'] = shelf_life_copy['Ingredient'].apply(
        lambda x: fuzz.ratio(ingredient, str(x))
    )
    
    best_match = shelf_life_copy.nlargest(1, 'similarity')
    
    if not best_match.empty and best_match.iloc[0]['similarity'] >= 75:
        return best_match.iloc[0]
    
    return None

# Function to create Knowledge graph 
def create_knowledge_graph(tx, recipe):
    # Create Recipe node
    tx.run("""
    MERGE (r:Recipe {id: $id})
    SET r.title = $title,
        r.source = $source,
        r.link = $link
    """, 
    id=recipe['id'],
    title=recipe.get('title', ''),
    source=recipe.get('source', ''),
    link=recipe.get('link', ''))
    
    # Process Ingredients
    ingredients = ast.literal_eval(recipe['ingredients']) if isinstance(recipe['ingredients'], str) else recipe['ingredients']
    
    for ing in ingredients:
        ing = ing.strip()
        if not ing:
            continue
            
        # Get shelf life data
        shelf_data = find_shelf_life(ing)
        
        # Create Ingredient node with shelf life
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        MERGE (i:Ingredient {name: $name})
        SET i.category = $category,
            i.shelf_life = $shelf_life,
            i.source = $source
        MERGE (r)-[:USES]->(i)
        """,
        recipe_id=recipe['id'],
        name=ing,
        category=shelf_data['Category'] if shelf_data else 'Unknown',
        shelf_life=shelf_data['Shelf_Life'] if shelf_data else 'Not specified',
        source='FoodKeeper' if shelf_data else 'RecipeNLG')
    
    # Process Steps
    directions = ast.literal_eval(recipe['directions']) if isinstance(recipe['directions'], str) else recipe['directions']
    for step_num, step in enumerate(directions, 1):
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        MERGE (s:Step {description: $description})
        SET s.order = $order
        MERGE (r)-[:HAS_STEP {order: $order}]->(s)
        """,
        recipe_id=recipe['id'],
        description=step.strip(),
        order=step_num)

def process_embedding_batch(session, batch):
    """Process a batch of recipes for embeddings"""
    try:
        # Extract just the texts for batch embedding
        texts = [text for _, text in batch]
        
        # Get batch embeddings - MUCH faster than individual calls
        batch_embeddings = embeddings.embed_documents(texts)
        
        # Update each recipe in the batch
        for (recipe_id, _), embedding in zip(batch, batch_embeddings):
            session.run("""
                MATCH (r:Recipe {id: $id})
                SET r.embedding = $embedding
            """, id=recipe_id, embedding=embedding)
        
        return True
        
    except Exception as e:
        print(f"Batch failed: {e}")
        return False

def add_embeddings_in_batch():
    """Add embeddings to all recipes in batches of 50"""
    batch_size = 50 
    print(f"Starting batch embedding process with batch size {batch_size}...")
    
    with driver.session() as session:
        recipes = session.run("""
            MATCH (r:Recipe) 
            WHERE r.embedding IS NULL 
            RETURN r.id AS id, r.title AS title,
                   [(r)-[:USES]->(i) | i.name] AS ingredients
        """)
        
        recipe_batch = []
        processed_count = 0
        
        for record in recipes:
            recipe_text = f"Recipe: {record['title']}. Ingredients: {', '.join(record['ingredients'])}"
            recipe_batch.append((record['id'], recipe_text))
            
            if len(recipe_batch) >= batch_size:
                process_embedding_batch(session, recipe_batch)
                processed_count += len(recipe_batch)
                print(f"Processed {processed_count} recipes so far...")
                recipe_batch = []
        
        if recipe_batch:
            process_embedding_batch(session, recipe_batch)
            processed_count += len(recipe_batch)
        
        print(f"Completed! Processed {processed_count} recipes with embeddings")

add_embeddings_in_batch()  
        
# First create vector index
with driver.session() as session:
    session.run("""
        CREATE VECTOR INDEX IF NOT EXISTS FOR (r:Recipe) ON r.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}
    """)

# Create knowledge graph (without embeddings for speed)
with driver.session() as session:
    for _, recipe in recipes_df.iterrows():
        try:
            session.execute_write(create_knowledge_graph, recipe)
        except Exception as e:
            print(f"Error with recipe {recipe['id']}: {str(e)}")

add_embeddings_in_batch()

driver.close()
print("Knowledge graph creation complete!")