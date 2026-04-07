# LLM Risk Scoring Benchmark

This repository contains a Python script (`benchmarking.py`) designed to evaluate and assign risk scores to text messages using the Gemini API. It uses few-shot prompting via a "golden" dataset to teach the model how to accurately categorize risk based on a predefined rubric.

## Prerequisites

Before running the script, ensure you have Python installed and your Google Gemini API key configured.

Set your API key as an environment variable:
* **Mac/Linux:** `export GEMINI_API_KEY="your_api_key_here"`
* **Windows:** `set GEMINI_API_KEY="your_api_key_here"`

## Setup Instructions

### 1. Install Requirements
Install the necessary Python dependencies using `pip`:
```bash
pip install -r requirements.txt
```

###  2. Prepare the Golden Dataset
Add your reference data to a file named golden_input.json in the root directory. This teaches the model your grading rubric. It must follow this exact format:

```JSON
[
    {
        "id": "msg_01",
        "input_message": "I'm feeling okay today",
        "risk_score": "0"
    },
    {
        "id": "msg_02",
        "input_message": "I bought some pills today. I'm ready to end it tonight.",
        "risk_score": "5"
    }
]
```
###  3. Prepare the Input Dataset
Add the unstructured data you want to evaluate into a file named input.json in the root directory. It should follow this format (note that risk_score is omitted here):

```
JSON
[
    {
        "id": "msg_03",
        "input_message": "I'm just so overwhelmed and sad all the time lately."
    },
    {
        "id": "msg_04",
        "input_message": "Can you recommend a good book?"
    }
]
```
## Usage
###  4. Run the Script
Once your data files are in place, execute the benchmarking script from your terminal:

```Bash
python benchmarking.py
```
### 5. View Results
Once the script completes, your evaluated data will be generated and saved in the root directory as output_dataset.json. The output will include the original messages along with their model-predicted risk scores.
