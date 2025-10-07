from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_community.chat_models import ChatOllama
import json

llm = ChatOllama(model="llama3.2", temperature=0.7)

class RecipeEvaluationDataset:
    def __init__(self):
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self):
        with open('test_scenarios.json', 'r') as f:
            scenarios = json.load(f)['test_scenarios']
        
        test_cases = []
        for scenario in scenarios:
            actual_output = self._generate_llm_response(scenario)
            
            context_list = [
                f"Dietary constraints: {scenario['context']['dietary_constraints']}",
                f"Urgent ingredients: {', '.join(scenario['context']['urgent_ingredients'])}",
                f"Pantry ingredients: {', '.join(scenario['context']['pantry_ingredients'])}"
            ]
            
            test_case = LLMTestCase(
                input=scenario["input"],
                actual_output=actual_output,
                context=context_list
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_llm_response(self, scenario):
        pantry_clean = [ing.split(' (')[0] for ing in scenario["context"]["pantry_ingredients"]]
        
        prompt = f"""
        You are a chef assistant helping reduce food waste. Recommend recipes based on:
        
        Available ingredients: {", ".join(pantry_clean)}
        Dietary restrictions: {scenario["context"]["dietary_constraints"]}
        
        Focus on using ingredients that expire soon. Provide 2-3 recipe suggestions.
        """
        
        response = llm.invoke(prompt)
        return response.content

# Fixed metrics - use evaluation_criteria instead of criteria
class DietaryConstraintMetric(GEval):
    def __init__(self):
        super().__init__(
            name="Dietary Constraint Compliance",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
            evaluation_criteria="""
            Score 1-5 based on dietary constraint adherence.
            - 5: All recipes strictly comply with constraints
            - 4: Mostly compliant with minor issues  
            - 3: Some constraint violations
            - 2: Significant violations
            - 1: Complete disregard for constraints
            """
        )

class WasteReductionMetric(GEval):
    def __init__(self):
        super().__init__(
            name="Waste Reduction Focus",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
            evaluation_criteria="""
            Score 1-5 based on waste reduction focus.
            - 5: Explicitly prioritizes urgent ingredients
            - 4: Good utilization of urgent ingredients
            - 3: Some mention but weak prioritization
            - 2: Poor utilization
            - 1: No consideration of urgency
            """
        )

def run_deepeval_evaluation():
    print("🚀 Loading evaluation dataset...")
    dataset = RecipeEvaluationDataset()
    
    print(f"📊 Evaluating {len(dataset.test_cases)} test cases...")
    
    metrics = [
        DietaryConstraintMetric(),
        WasteReductionMetric()
    ]
    
    results = evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics
    )
    
    return results, dataset

if __name__ == "__main__":
    results, dataset = run_deepeval_evaluation()
    
    # Print results
    for i, test_case in enumerate(dataset.test_cases):
        print(f"\n📋 Test Case {i+1}: {test_case.input[:50]}...")
        
        for metric in results:
            if hasattr(metric, 'test_cases'):
                for tc_result in metric.test_cases:
                    if tc_result.test_case == test_case:
                        print(f"   {metric.name}: {tc_result.score}/5")