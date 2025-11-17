import pandas as pd
from app import generate_recommendation

def process_excel_with_recommendations(input_file, output_file):
    """
    Read Excel file, get system recommendations for each scenario, and save with outputs
    """
    df = pd.read_excel(input_file)
    
    # Initialize list to store system outputs
    system_outputs = []
    
    print(f"Processing {len(df)} scenarios...")
    
    for index, row in df.iterrows():
        print(f"Processing scenario {index + 1}/{len(df)}: {row['scenario_id']}")
        
        try:
            # Extract parameters from the row
            user_input = row['input']
            pantry_ingredients = [x.strip() for x in str(row['pantry_ingredients']).split(',')]
            dietary_constraints = str(row['dietary_constraints'])
            allergies = str(row['allergies'])
        
            # Get recipe recommendations from your system
            recipes = generate_recommendation(
                pantry=pantry_ingredients,
                dietary_needs=dietary_constraints,
                allergies=allergies
            )

            if recipes:
              system_output = recipes

            else:
                system_output = "No suitable recipes found matching the constraints and ingredients."
            
            system_outputs.append(system_output)
            print(f"Successfully processed {row['scenario_id']}")
            
        except Exception as e:
            error_msg = f"Error processing scenario: {str(e)}"
            print(f"✗ Error in {row['scenario_id']}: {error_msg}")
            system_outputs.append(error_msg)
    
    # Add system outputs to dataframe
    df['system_recommendation'] = system_outputs
    
    # Save to new Excel file
    df.to_excel(output_file, index=False)
    print(f"\n Processing complete! Results saved to: {output_file}")
    print(f"Processed {len(df)} scenarios")
    
    return df


input_excel = "./tests/manual_tests/test_scenarios.xlsx"
output_excel = "./tests/manual_tests/constraints_output.xlsx"


results_df = process_excel_with_recommendations(input_excel, output_excel)

# Print summary
success_count = len([x for x in results_df['system_recommendation'] if "Error" not in x and "error" not in x])
print(f"\n Summary:")
print(f"   Total scenarios: {len(results_df)}")
print(f"   Successfully processed: {success_count}")
print(f"   Output saved to: {output_excel}")
