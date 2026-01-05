import json
import re
import pandas as pd
import os
import sys

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app import generate_recommendation, get_recipe_recommendations, parse_shelf_life, driver, query_llama_api

# judge model
class JudgeModel():
    def call_model(self, prompt):
        return query_llama_api(prompt)

# Initialize judge
judge_model = JudgeModel()

# LLM-as-Judge Prompt with focus on shelf life prioritization
SHELF_LIFE_PRIORITIZATION_PROMPT = """
As a food waste reduction expert, evaluate this recipe recommendation:

USER PANTRY: {pantry_ingredients}
URGENT INGREDIENTS (expire soon): {urgent_ingredients}
RECIPES RECOMMENDED: {recipes_used}
ASSISTANT EXPLANATION: {actual_output}

Evaluate based on ACTUAL ingredient usage:
1. Do the recommended recipes contain urgent ingredients? 
2. Does the explanation acknowledge ingredient urgency?
3. Is proper reasoning provided for using urgent ingredients?

Rating:
0 = Recipes don't use urgent ingredients, no urgency mentioned
1 = Some urgent ingredients used, weak explanation  
2 = Recipes prioritize urgent ingredients with clear waste reduction reasoning

Rating: [0/1/2]
Explanation: [Your reasoning]
"""

def save_to_excel(row):
    """Save evaluation results to Excel file"""
    row = {key: str(value) if value is not None else "" for key, value in row.items()}
    
    os.makedirs('./output', exist_ok=True)
    output_file = './output/shelf_life_output_new.xlsx'
    
    try:
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
            new_df = pd.DataFrame([row])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(output_file, index=False)
        else:
            new_df = pd.DataFrame([row])
            new_df.to_excel(output_file, index=False)
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        new_df = pd.DataFrame([row])
        new_df.to_excel(output_file, index=False)


class ShelfLifePrioritizationMetric(BaseMetric):
    def __init__(self):
        self.score = 0
        self.reason = ""
        
    def measure(self, test_case):
        context_text = test_case.context[0] if test_case.context else ""
        
        pantry_ingredients = ""
        urgent_ingredients = ""
        recipes_used = ""
        actual_output = test_case.actual_output
        
        # Split by lines and extract the components
        for line in context_text.split('\n'):
            if line.startswith("PANTRY:"):
                pantry_ingredients = line.replace("PANTRY:", "").strip()
            elif line.startswith("URGENT INGREDIENTS:"):
                urgent_ingredients = line.replace("URGENT INGREDIENTS:", "").strip()
            elif line.startswith("RECIPES RECOMMENDED:"):
                recipes_used = line.replace("RECIPES RECOMMENDED:", "").strip()
        
        prompt = SHELF_LIFE_PRIORITIZATION_PROMPT.format(
            pantry_ingredients=pantry_ingredients,
            urgent_ingredients=urgent_ingredients,
            recipes_used=recipes_used,
            actual_output=actual_output
        )
        
        try:
            response = judge_model.call_model(prompt)
            score_text = response if isinstance(response, str) else response[0] if isinstance(response, tuple) else str(response)
            
            self.score = self.extract_score(score_text)
            self.reason = self.extract_reason(score_text)
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            self.score = 0
            self.reason = f"Evaluation error: {str(e)}"
        
        return self.score
    
    def extract_score(self, text):
        pattern = r'Rating:\s*(\d+)'
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except:
                return 0
        return 0
    
    def extract_reason(self, text):
        pattern = r'Explanation:\s*([\s\S]*)'
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
        return text

def get_ingredient_shelf_life_data(ingredient_names):
    if not driver:
        print("No Neo4j driver available")
        return {"shelf_life_data": {}, "urgent_ingredients": []}
    
    try:
        with driver.session() as session:
            result = session.run("""
            UNWIND $ingredients AS search_ing
            MATCH (i:Ingredient)
            WHERE toLower(trim(i.name)) = toLower(trim(search_ing))
            OR toLower(trim(i.name)) = toLower(trim(search_ing + 's'))
            OR toLower(trim(search_ing)) = toLower(trim(i.name))
            OR toLower(trim(search_ing + 's')) = toLower(trim(i.name))
            RETURN i.name AS name, i.shelf_life AS shelf_life
            """, ingredients=[str(x).lower() for x in ingredient_names])
    
            shelf_life_data = {}
            urgent_ingredients = []
            
            records_found = 0
            for record in result:
                records_found += 1
                name = record["name"]
                shelf_life = record["shelf_life"]
                days_remaining = parse_shelf_life(shelf_life)
                
                shelf_life_data[name] = {
                    "shelf_life": shelf_life,
                    "days_remaining": days_remaining
                }
                
                if days_remaining <= 7: 
                    urgent_ingredients.append(f"{name} ({int(days_remaining)} days)")
                    # print(f"{name} is URGENT ({int(days_remaining)} days)")
            
            # print(f"    Found {records_found} ingredients with shelf life data")
            # print(f"    Found {len(urgent_ingredients)} urgent ingredients")
            
            return {
                "shelf_life_data": shelf_life_data,
                "urgent_ingredients": urgent_ingredients
            }
    except Exception as e:
        print(f"Error getting shelf life data: {e}")
        import traceback
        traceback.print_exc()
        return {"shelf_life_data": {}, "urgent_ingredients": []}


def evaluate_urgent_usage(recipes, pantry_ingredients, urgent_ingredients):
    """measure if recipes use urgent ingredients"""
    if not urgent_ingredients or not recipes:
        return 0.0
    
    pantry_urgent = []
    for urgent in urgent_ingredients:
        # Extract base name: "tomatoes (0 days)" -> "tomatoes"
        urgent_base = urgent.split(' (')[0].strip().lower()
        
        # Check if this matches any pantry ingredient
        for pantry_item in pantry_ingredients:
            pantry_lower = str(pantry_item).lower()
            if (urgent_base in pantry_lower) or (pantry_lower in urgent_base):
                pantry_urgent.append(pantry_lower)
                break
    
    # calculate usage against urgent ingredients
    total_urgent_used = 0
    for recipe in recipes[:3]:
        recipe_ings = []
        for ing in recipe.get('ingredients', []):
            if isinstance(ing, dict):
                recipe_ings.append(ing.get('name', '').lower())
            else:
                recipe_ings.append(str(ing).lower())
        
        # Count how many pantry urgent items are used
        urgent_used = sum(1 for urgent in pantry_urgent 
                         if any(urgent in recipe_ing for recipe_ing in recipe_ings))
        total_urgent_used += urgent_used
    
    return total_urgent_used / len(pantry_urgent) if pantry_urgent else 0.0

def evaluate_scenario_with_shelf_life(scenario):
    """Evaluate a single scenario with shelf life analysis"""
    
    scenario_id = scenario['scenario_id']
    ctx = scenario['context']

    pantry = ctx['pantry_ingredients']
    dietary_constraints = ctx['dietary_constraints'] 
    allergies = ctx['allergies']
    
    print(f" Evaluating: {scenario_id}")
    print(f"   Pantry: {pantry}")
    
    # Initialize variables with default values
    assistant_output = ""
    recipes_used = []
    urgent_ingredients_used = []
    all_urgent_ingredients = []
    urgent_usage_rate = 0.0
        
    try:
        # Get the assistant's recommendation
        assistant_output = generate_recommendation(pantry, dietary_constraints, allergies)
        
        # Get the actual recipes recommended
        recipes = get_recipe_recommendations(pantry, dietary_constraints, allergies)
        recipes_used = [r.get('title', 'Unknown') for r in recipes[:3]] if recipes else []
        
        print(f"   Recipes recommended: {recipes_used}")
        
        # DEBUG: Check what urgent ingredients each recipe has and why
        if recipes:
            for i, recipe in enumerate(recipes[:3]):
                urgent_ings = recipe.get('urgent_ingredients', [])
                all_ingredients = recipe.get('ingredients', [])
                print(f"   Recipe {i+1} '{recipe.get('title', 'Unknown')}':")
                print(f"     - All ingredients: {[ing['name'] if isinstance(ing, dict) else str(ing) for ing in all_ingredients]}")
                print(f"     - Urgent ingredients: {urgent_ings}")
        
        # Extract urgent ingredients from the recipes themselves
        urgent_ingredients_used = []
        if recipes:
            for recipe in recipes[:3]:
                urgent_ings = recipe.get('urgent_ingredients', [])
                for urgent_desc in urgent_ings:
                    ing_name = urgent_desc.split(' (')[0].strip()
                    urgent_ingredients_used.append(ing_name)
        
        urgent_ingredients_used = list(set(urgent_ingredients_used))
        print(f"   Urgent ingredients used in recommendations: {urgent_ingredients_used}")
        
        # Get shelf life context for the judge
        shelf_life_info = get_ingredient_shelf_life_data(pantry)
        all_urgent_ingredients = shelf_life_info["urgent_ingredients"]
        print(f" All urgent ingredients from database: {all_urgent_ingredients}")
        
        # Calculate objective urgent usage rate
        urgent_usage_rate = evaluate_urgent_usage(recipes, pantry, all_urgent_ingredients)
        print(f" Objective urgent usage: {urgent_usage_rate:.1%}")
        
    except Exception as e:
        print(f" Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Create test case for LLM judge - include recipes in context
    context_string = f"PANTRY: {', '.join(pantry)}\nURGENT INGREDIENTS: {'; '.join(all_urgent_ingredients) if all_urgent_ingredients else 'None found'}\nRECIPES RECOMMENDED: {', '.join(recipes_used) if recipes_used else 'None found'}\nASSISTANT RESPONSE: {assistant_output}"
    
    shelf_life_test_case = LLMTestCase(
        input=f"Recipes using {', '.join(pantry)}", 
        actual_output=assistant_output,
        expected_output="Recipe that prioritizes soon-to-expire ingredients",
        context=[context_string]
    )

    # Evaluate with LLM judge
    shelf_life_metric = ShelfLifePrioritizationMetric()
    shelf_life_score = shelf_life_metric.measure(shelf_life_test_case)

    print(f" LLM-as-Judge Score: {shelf_life_score:.2f}")
    if shelf_life_metric.reason:
        print(f"   Explanation: {shelf_life_metric.reason[:150]}...")

    # Save comprehensive results
    try:
        row = {
            'scenario_id': scenario_id,
            'pantry_ingredients': ', '.join(pantry),
            'dietary_constraints': dietary_constraints,
            'allergies': allergies,
            'urgent_ingredients_found': ', '.join(all_urgent_ingredients) if all_urgent_ingredients else 'None',
            'recipes_recommended': ', '.join(recipes_used) if recipes_used else 'None',
            'urgent_ingredients_used': ', '.join(urgent_ingredients_used) if urgent_ingredients_used else 'None',
            'urgent_usage_rate': f"{urgent_usage_rate:.1%}",
            'assistant_response': assistant_output[:1500] + '...' if len(assistant_output) > 1500 else assistant_output,
            'llm_shelf_life_score': shelf_life_score,
            'shelf_life_explanation': shelf_life_metric.reason,
            'eval_passed': shelf_life_score >= 1.0
        }
        
        save_to_excel(row)
        print(f" Saved results for {scenario_id}")
    except Exception as e:
        print(f" Error saving to excel: {e}")

    return {
        "scenario_id": scenario_id,
        "llm_shelf_life_score": shelf_life_score,
        "urgent_ingredients_found": all_urgent_ingredients,
        "urgent_ingredients_used": urgent_ingredients_used,
        "urgent_usage_rate": urgent_usage_rate,
        "recipes_recommended": recipes_used,
        "eval_passed": shelf_life_score >= 1.0
    }

def load_test_cases():
    """Load test scenarios from JSON file"""
    try:
        with open('./test_data/test_scenarios_new.json', 'r') as f:
            data = json.load(f)
            return data.get('test_scenarios', [])
    except Exception as e:
        print(f"Failed to load test scenarios: {e}")
        return []

def run_shelf_life_evaluation():
    """Run shelf life prioritization evaluation"""
    scenarios = load_test_cases()
    
    if not scenarios:
        print(" No test scenarios found!")
        return

    print(f" Loaded {len(scenarios)} test scenarios")
    
    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n Processing {i}/{len(scenarios)}: {scenario['scenario_id']}")
        
        results = evaluate_scenario_with_shelf_life(scenario)
        
        if results:
            all_results.append(results)
    
    # Generate Summary Report
    print(" SHELF LIFE PRIORITIZATION EVALUATION SUMMARY")
    
    if all_results:
        shelf_life_scores = [r.get("llm_shelf_life_score", 0) for r in all_results]
        shelf_life_passed = [r for r in all_results if r.get("llm_shelf_life_score", 0) >= 1.0]
        
        # Calculate urgent ingredient usage statistics
        total_urgent_found = sum(len(r.get("urgent_ingredients_found", [])) for r in all_results)
        total_urgent_used = sum(len(r.get("urgent_ingredients_used", [])) for r in all_results)
        avg_urgent_usage = sum(r.get("urgent_usage_rate", 0) for r in all_results) / len(all_results)
        
        urgent_usage_rate = (total_urgent_used / total_urgent_found * 100) if total_urgent_found > 0 else 0

        print(f"Shelf Life Prioritization:")
        print(f"Passed: {len(shelf_life_passed)}/{len(all_results)}")
        print(f"Average Score: {sum(shelf_life_scores)/len(shelf_life_scores):.2f}/2.0")
        print(f"Urgent Ingredients Usage: {total_urgent_used}/{total_urgent_found} ({urgent_usage_rate:.1f}%)")
        print(f"Average Urgent Usage Rate: {avg_urgent_usage:.1%}")
        
        # Show scenarios that failed
        failed_scenarios = [r for r in all_results if r.get("llm_shelf_life_score", 0) < 1.0]
        if failed_scenarios:
            print(f"\nFailed Scenarios:")
            for failed in failed_scenarios:
                print(f" - {failed['scenario_id']} (Score: {failed['llm_shelf_life_score']:.1f}, Usage: {failed.get('urgent_usage_rate', 0):.1%})")
        
    else:
        print("No results to summarize")

    return all_results

if __name__ == "__main__":
    # Run the evaluation
    results = run_shelf_life_evaluation()