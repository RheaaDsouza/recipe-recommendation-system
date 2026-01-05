import json
import os
import pandas as pd
import re

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app import generate_recommendation, get_recipe_recommendations, query_llama_api

# Load environment variables
load_dotenv()

# ---- LLM-as-Judge Evaluation Prompts ----
CONSTRAINT_ADHERENCE_PROMPT = """
As a nutrition expert, evaluate how well this recipe recommendation adheres to the user's dietary constraints and allergies.

Give a rating between 0-2, where:
0 means fail (the recommendation clearly violates dietary constraints or contains allergens)
1 means partial (the recommendation mostly adheres but has minor issues)
2 means pass (the recommendation fully respects all constraints and allergies)

Provide a brief explanation for your rating.

Your response MUST be in the following format:

Rating: [0 or 1 or 2]
Explanation: [Brief reasoning for the rating]

USER DIETARY CONSTRAINTS: {dietary_constraints}
USER ALLERGIES: {allergies}
ASSISTANT RESPONSE: {actual_output}

Your evaluation result:
"""

class JudgeModel():
    def call_model(self, prompt):
        return query_llama_api(prompt)

# Initialize judge
judge_model = JudgeModel()

def save_to_excel(row):
  # Convert row values to strings to avoid type issues
  row = {key: str(value) if value is not None else "" for key, value in row.items()}
  df_row = pd.DataFrame([row])
  try:
    res = pd.read_excel('./output/constraint_output_new.xlsx')
    combined = pd.concat([res, df_row], ignore_index=True)
    combined.to_excel('./output/constraint_output_new.xlsx', index=False)
  except FileNotFoundError:
    # Create new file if it doesn't exist
    df_row.to_excel('./output/constraint_output_new.xlsx', index=False)
    print(f"Created new Excel file with scenario")
  except Exception as e:
    print(f"Error saving to excel: {e}")
    df_row.to_excel('./output/constraint_output_new.xlsx', index=False)


def violates_dietary_preferences(ingredients_text, dietary_needs):
    """Check if recipe violates dietary preferences - copied from app"""
    if not dietary_needs:
        return False
    
    # Convert to lowercase for comparison
    dietary_needs_lower = dietary_needs.lower()
    ingredients_lower = ingredients_text.lower()
    
    if 'vegetarian' in dietary_needs_lower:
        meat_keywords = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'bacon', 'sausage', 'lamb', 'turkey', 'meat']
        if any(meat in ingredients_lower for meat in meat_keywords):
            return True
    
    if 'vegan' in dietary_needs_lower:
        animal_products = ['milk', 'cheese', 'butter', 'egg', 'honey', 'yogurt', 'cream', 'gelatin']
        if any(product in ingredients_lower for product in animal_products):
            return True
    
    if 'gluten-free' in dietary_needs_lower or 'gluten free' in dietary_needs_lower:
        gluten_keywords = ['wheat', 'flour', 'bread', 'pasta', 'barley', 'rye', 'couscous', 'farro']
        if any(gluten in ingredients_lower for gluten in gluten_keywords):
            return True
        
    if 'dairy-free' in dietary_needs_lower or 'dairy free' in dietary_needs_lower: 
        dairy_products = ["milk", "cheese", "butter", "cream", "yogurt", "ghee"]
        if any(dairy in ingredients_lower for dairy in dairy_products):
            return True
    
    if 'dairy-free' in dietary_needs_lower or 'dairy free' in dietary_needs_lower: 
        dairy_products = ["milk", "cheese", "butter", "cream", "yogurt", "ghee"]
        if any(dairy in ingredients_lower for dairy in dairy_products):
            return True
        
    if 'keto' in dietary_needs_lower: 
        keto_products = ["sugar", "grains", "wheat", "rice", "pasta", "bread", "potato", 
             "corn", "legumes", "beans", "lentils"]
        if any(keto in ingredients_lower for keto in keto_products):
            return True
        
    if 'low-carb' in dietary_needs_lower or 'carb free': 
        carb_keywords = ["sugar", "grains", "wheat", "rice", "pasta", "bread", "potato", "starchy"]
        if any(carb in ingredients_lower for carb in carb_keywords):
            return True

    return False

# Custom LLM-as-Judge Metrics
class ConstraintAdherenceMetric(BaseMetric):
    def __init__(self):
        self.score = 0
        self.reason = ""
        
    def measure(self, test_case):
        # Just use the actual output and rely on the context being in the prompt
        prompt = CONSTRAINT_ADHERENCE_PROMPT.format(
            dietary_constraints=", ".join([ctx for ctx in test_case.context if "Dietary Constraints" in ctx]),
            allergies=", ".join([ctx for ctx in test_case.context if "Allergies" in ctx]),
            actual_output=test_case.actual_output
        )
        
        try:
            gen = judge_model.call_model(prompt)
            if isinstance(gen, tuple) and len(gen) >= 1:
                score_text = gen[0]
            else:
                score_text = gen

            self.score = self.extract_score(score_text)
            
            if not self.reason:
                self.reason = score_text
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
        
        return self.score
    
    def extract_score(self, text):
        # Handle structured 'Rating: N\nExplanation: ...' first, then integer-only fallbacks.
        pattern = r'Rating:\s*(\d+)\s*\n?Explanation:\s*([\s\S]*)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                rating = int(m.group(1))
                explanation = m.group(2).strip()
                self.score = rating
                self.reason = explanation
                return rating
            except Exception:
                self.score = None
                return 0

        self.score = None
        self.reason = text
        return 0

def evaluate_scenario(scenario):
    """Evaluate a single scenario - testing the assistants's final response"""
    id = scenario['scenario_id']
    ctx = scenario['context']

    pantry = ctx['pantry_ingredients']
    dietary_constraints = ctx['dietary_constraints'] 
    allergies = ctx['allergies']
    
    print(f" Evaluating: {id}")
    print(f" Pantry: {pantry}")
    print(f" Constraints: {dietary_constraints}")
    print(f" Allergies: {allergies}")
        
    # Get the assistant's final response
    try:
        assistant_output = generate_recommendation(pantry, dietary_constraints, allergies)
    except Exception as e:
        print(f" Assistant generation failed: {e}")
        assistant_output = ""

    # Get the underlying recipes to provide context to the LLM judge
    recipe_context = ""
    try:
        recipes = get_recipe_recommendations(pantry, dietary_constraints, allergies)
        if recipes:
            # Build detailed recipe context for the LLM judge
            recipe_details = []
            for i, recipe in enumerate(recipes[:3]):  # Show top 3 recipes
                # Extract ingredient names
                raw_ings = recipe.get('ingredients', []) or []
                ingredient_names = []
                for ing in raw_ings:
                    if isinstance(ing, dict):
                        ingredient_names.append(ing.get('name', 'Unknown'))
                    else:
                        ingredient_names.append(str(ing))
                
                # Build recipe description
                recipe_desc = f"Recipe {i+1}: {recipe.get('title', 'Unknown')}\n"
                recipe_desc += f"Ingredients: {', '.join(ingredient_names)}\n"
                
                # Add steps if available
                steps = recipe.get('steps', [])
                if steps:
                    recipe_desc += f"Instructions: {' '.join(steps[:2])}\n"  # First 2 steps
                
                recipe_details.append(recipe_desc)
            
            recipe_context = "\n".join(recipe_details)
            
            # Also print for debugging
            print(f"Top recipe: {recipes[0].get('title', 'Unknown')}")
            if ingredient_names:
                print(f"Uses ingredients: {', '.join(ingredient_names)}")
            
        else:
            print("No recipes found underlying the response")
            recipe_context = "No recipes found that match the constraints."
            
    except Exception as e:
        print(f"Could not get recipes: {e}")
        recipe_context = f"Error retrieving recipes: {e}"

    # Provide the LLM judge with BOTH the assistant's response AND the recipe details
    test_case = LLMTestCase(
        input=f"Recipes using {', '.join(pantry)}",
        actual_output=assistant_output,
        expected_output="Appropriate recipe recommendation",
        context=[
            f"Dietary Constraints: {dietary_constraints}",
            f"Allergies: {allergies}", 
            f"Pantry Ingredients: {', '.join(pantry)}",
            f"Recommended Recipes Details:\n{recipe_context}"
        ]
    )

    constraint_metric = ConstraintAdherenceMetric()
    constraint_score = constraint_metric.measure(test_case)

    print(f"LLM-as-Judge Scores:")
    print(f"Constraint Adherence: {constraint_score:.2f}")
    if constraint_metric.reason:
        print(f"Explanation: {constraint_metric.reason}")

    results = {
        "scenario": id,
        "assistant_response": assistant_output[:200] + "..." if len(assistant_output) > 200 else assistant_output,
        "llm_constraint_score": constraint_score,
        "eval_passed": constraint_score >= 1.0
    }

    # Save to excel
    try:
        row = {
            'scenario': id,
            'pantry': ', '.join(pantry),
            'dietary_constraints': dietary_constraints,
            'allergies': allergies,
            'assistant_response': (str(assistant_output)[:1000] + '...') if assistant_output and len(str(assistant_output)) > 1000 else str(assistant_output),
            'judge_explanation': constraint_metric.reason if hasattr(constraint_metric, 'reason') else '',
            'llm_constraint_score': constraint_score,
            "eval_passed": constraint_score >= 1.0
        }
        save_to_excel(row)
    except Exception as e:
        print(f"Error saving to excel: {e}")

    return results

def load_test_cases():
    """Load test scenarios from JSON file"""
    try:
        with open('test_data/test_scenarios_new.json', 'r') as f:
            data = json.load(f)
            return data.get('test_scenarios', [])
    except Exception as e:
        print(f"Failed to load test scenarios: {e}")
        return []

# Main Evaluation Function 
def evaluation():
    print("🥑🥑 COMPREHENSIVE EVALUATION")
    scenarios = load_test_cases()

    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        results = evaluate_scenario(scenario)
        
        if results:
            all_results.append(results)
    
    #  Summary Report 
    # TODO: Just to get an idea of the performance on the terminal. Remove this later
    print("🌼 EVALUATION SUMMARY") 
    if all_results:
      passed_evals = [r for r in all_results if r.get("llm_constraint_score", 0) >= 1.0]
      failed_evals = [r for r in all_results if r.get("llm_constraint_score", 0) < 1.0]
        
      print(f"Passed Evaluations: {len(passed_evals)}/{len(all_results)}")
      print(f"Failed Evaluations: {len(failed_evals)}/{len(all_results)}")


if __name__ == "__main__":
    evaluation()