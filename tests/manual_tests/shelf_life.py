import pandas as pd
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app import get_recipe_recommendations, parse_shelf_life, driver

def load_test_cases(file_path):
    """Load test scenarios from Excel file"""
    df = pd.read_excel(file_path)
    test_cases = []
    
    for index, row in df.iterrows():
            
        # Parse ingredients
        pantry_ingredients = [ing.strip() for ing in str(row['pantry_ingredients']).split(',')]
        
        test_case = {
            "name": row['scenario_id'],
            "pantry": pantry_ingredients,
            "constraints": row['dietary_constraints'] if pd.notna(row['dietary_constraints']) else "",
            "allergies": row['allergies'] if pd.notna(row['allergies']) else ""
        }
        
        test_cases.append(test_case)
    return test_cases

def get_ingredient_shelf_life_data(ingredient_names):
    """Get shelf life data for ingredients"""
    if not driver:
        return {"shelf_life_data": {}, "urgent_ingredients": []}
    
    with driver.session() as session:
        result = session.run("""
            UNWIND $ingredients AS search_ing
            MATCH (i:Ingredient)
            WHERE toLower(trim(i.name)) CONTAINS toLower(trim(search_ing))
            RETURN i.name AS name, i.shelf_life AS shelf_life
            """, ingredients=[str(x).lower().strip() for x in ingredient_names])
        
        shelf_life_data = {}
        urgent_ingredients = []
        
        for record in result:
            name = record["name"]
            shelf_life = record["shelf_life"]
            days_remaining = parse_shelf_life(shelf_life)
            
            shelf_life_data[name] = {
                "shelf_life": shelf_life,
                "days_remaining": days_remaining
            }
            
            if days_remaining <= 7:
                urgent_ingredients.append(f"{name} ({int(days_remaining)} days)")
        
        return {
            "shelf_life_data": shelf_life_data,
            "urgent_ingredients": urgent_ingredients
        }

def evaluate_objective_shelf_life_prioritization(pantry_ingredients, dietary_constraints="", allergies=""):
    """Evaluate based on the TOP recipe recommendation"""
    
    # Get recipe recommendations
    recipes = get_recipe_recommendations(pantry_ingredients, dietary_constraints, allergies)
    
    if not recipes:
        return {"score": 0, "reason": "No recipes found", "recipe_details": []}
    
    top_recipe = recipes[0]
    
    # Get shelf life data
    shelf_life_data = get_ingredient_shelf_life_data(pantry_ingredients)
    
    # Find urgent ingredients in user's pantry
    user_urgent_ingredients = []
    for pantry_ing in pantry_ingredients:
        for ing_name, ing_data in shelf_life_data["shelf_life_data"].items():
            if pantry_ing.lower() in ing_name.lower():
                if ing_data["days_remaining"] <= 7:
                    user_urgent_ingredients.append(pantry_ing)
                break
    
    if not user_urgent_ingredients:
        return {"score": 1, "reason": "No urgent ingredients", "recipe_details": [{"recipe_title": top_recipe['title']}]}
    
    # Calculate score
    recipe_ingredients = [ing.get("name", "").lower() if isinstance(ing, dict) else str(ing).lower() 
                         for ing in top_recipe.get('ingredients', [])]
    
    urgent_used = [ing for ing in user_urgent_ingredients 
                  if any(ing.lower() in recipe_ing for recipe_ing in recipe_ingredients)]
    
    coverage = len(urgent_used) / len(user_urgent_ingredients)
    
    if coverage >= 0.8:
        score = 2
        reason = f"GOOD: Uses {len(urgent_used)}/{len(user_urgent_ingredients)} urgent ingredients"
    elif coverage >= 0.5:
        score = 1
        reason = f"FAIR: Uses {len(urgent_used)}/{len(user_urgent_ingredients)} urgent ingredients"
    else:
        score = 0
        reason = f"POOR: Uses {len(urgent_used)}/{len(user_urgent_ingredients)} urgent ingredients"
    
    return {
        "score": score,
        "reason": reason,
        "recipe_details": [{"recipe_title": top_recipe['title']}]
    }

def run_test(excel_file_path):
    """Main function to run tests from Excel file"""
    test_scenarios = load_test_cases(excel_file_path)
    
    results = []
    
    for scenario in test_scenarios:
        result = evaluate_objective_shelf_life_prioritization(
            scenario["pantry"], 
            scenario["constraints"],
            scenario["allergies"]
        )
        
        # Save results
        detailed_result = {
            "Test Name": scenario["name"],
            "Pantry Ingredients": ", ".join(scenario["pantry"]),
            "Objective Score": result["score"],
            "Score Explanation": result["reason"],
            "Top Recipe": result["recipe_details"][0]["recipe_title"] if result["recipe_details"] else "No recipe"
        }
        
        results.append(detailed_result)
    
    # Save results to Excel
    output_file = "shelf_life_output.xlsx"
    
    pd.DataFrame(results).to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return results, output_file

if __name__ == "__main__":
    FILE_PATH = "test_scenario.xlsx"
    
    if os.path.exists(FILE_PATH):
        results, output_file = run_test(FILE_PATH)
        print(f"\nEvaluation complete")
    else:
        print(f"Excel file not found: {FILE_PATH}")