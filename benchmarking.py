import json
import os
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 1. Configure the Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=api_key)

# Using Gemini 1.5 Flash as it is fast and excellent for classification tasks
model = genai.GenerativeModel('gemini-2.5-flash')

def get_risk_score(message: str, golden_data: list) -> str:
    """
    Calls Gemini to assess the risk score of a message using the rubric and golden dataset.
    """
    # Construct the prompt with the specific rubric
    prompt = (
        "You are an expert psychiatric and safety risk assessment system. Your task is to assign "
        "a risk score to the provided input message based strictly on the following rubric:\n\n"
        "Rubric:\n"
        "0 - No risk\n"
        "1 - Emotional distress\n"
        "2 - Ambiguous risk\n"
        "3 - Self-harm ideation\n"
        "4 - Passive suicidal ideation\n"
        "5 - Active suicidal ideation\n"
        "6 - Suicide planning\n"
        "7 - Attempt / imminent risk\n\n"
        "Here are some examples of how to apply this rubric:\n"
    )
    
    # Inject Few-Shot Examples
    for item in golden_data:
        prompt += f"- Message: \"{item['input_message']}\" | Risk Score: \"{item['risk_score']}\"\n"
        
    prompt += (
        f"\nNow, assess the following new message:\n"
        f"- Message: \"{message}\"\n\n"
        "Output ONLY a valid JSON object with a single key 'risk_score' mapping to the numerical "
        "score (0-7) as a string. Do not include any other text."
    )

    # Disable safety filters so the model can analyze high-risk text
    custom_safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = model.generate_content(
            prompt,
            safety_settings=custom_safety_settings,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0 
            )
        )
        
        result = json.loads(response.text)
        return result.get("risk_score", "UNKNOWN")
        
    except Exception as e:
        print(f"Error processing message '{message}': {e}")
        return "ERROR"

def main():
    # 2. Load the golden dataset from golden_input.json
    golden_filename = "golden_input.json"
    try:
        with open(golden_filename, "r", encoding="utf-8") as f:
            golden_dataset = json.load(f)
        print(f"Loaded {len(golden_dataset)} examples from {golden_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {golden_filename}. Please ensure the file exists.")
        return

    # 3. Load the dataset to be scored from input_dataset.json
    input_filename = "input.json"
    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            input_dataset = json.load(f)
        print(f"Loaded {len(input_dataset)} messages to score from {input_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_filename}. Please create this file with your un-scored data.")
        return
    
    output_dataset = []

    print(f"\nProcessing {len(input_dataset)} messages...")

    # 4. Iterate through the input dataset and get scores
    for i, item in enumerate(input_dataset):
        print(f"[{i+1}/{len(input_dataset)}] Scoring message: {item['input_message']}")
        
        predicted_score = get_risk_score(item["input_message"], golden_dataset)
        
        output_dataset.append({
            "id": item.get("id", f"unknown_{i}"),
            "input_message": item["input_message"],
            "risk_score": predicted_score
        })
        
        time.sleep(1) # Rate limit padding

    # 5. Save the results
    output_filename = "output_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, indent=4)
        
    print(f"\nProcessing complete! Results saved to {output_filename}")

if __name__ == "__main__":
    main()
