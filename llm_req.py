#request_dashboard_plan

import json
from openai import OpenAI
import time
from huggingface_hub import InferenceClient
import yaml

# ---------------- CONFIG ----------------
INPUT_FILE = "data/profile_summary.json"
OUTPUT_FILE = "data/dashboard_plan.json"
MAX_RETRIES = 1
# -----------------------------------------

def load_profile(file_path):
    """Load profile JSON from file."""
    with open(file_path, "r") as f:
        return json.load(f)

def build_prompt(profile):
    """Construct a detailed, example-rich prompt for best results."""
    return f"""
You are a senior data analytics expert who designs dashboards programmatically.

I will provide a dataset profile in JSON format. 
The JSON includes:
- dataset shape (rows, columns)
- inferred column types, cardinality, ranges, and null stats
- a few sample rows.

Your task:
1. Detect the most likely DOMAIN of the dataset 
   (examples: e-commerce, manufacturing, SaaS, finance, generic).
2. Suggest 1–3 TOP KPI cards relevant to this dataset:
   - Each KPI must include: 
     * "name" (short, clear), 
     * "description" (what it measures), 
     * "formula" (using existing columns only),
     * "importance" (0–10, 10 = must-have),
     * "reason" (30-40 words explaining why this KPI was selected)
3. Suggest 4–7 CHARTS for an interactive dashboard:
   - Each chart must include:
     * "name" (short),
     * "type" (line, bar, histogram, pie, etc.),
     * "x" column,
     * "y" metric or aggregation,
     * "description" (purpose),
     * "importance" (0–10, 10 = must-have),
     * "reason" (30-40 words explaining why this chart was selected)
4. Suggest GLOBAL FILTERS (date + 1–2 categorical columns)
5. Always output as valid JSON only. Example format:

{{
  "domain": "e-commerce",
  "kpis": [
    {{
      "name": "Total Revenue",
      "description": "Sum of sales across all transactions",
      "formula": "SUM(quantity * unitprice)",
      "importance": 4,
      "reason": "Selected because revenue is the most important measure of sales performance and directly reflects business growth across all transactions in the dataset."
    }}
  ],
  "charts": [
    {{
      "name": "Revenue Over Time",
      "type": "line",
      "x": "invoicedate",
      "y": "quantity * unitprice",
      "importance": 9,
      "description": "Shows total revenue trends over time",
      "reason": "This chart visualizes revenue changes over time, highlighting trends, seasonality, or anomalies, which helps in identifying peak sales periods and overall business performance."
    }}
  ],
  "filters": [
    {{
      "name": "Date Range",
      "column": "invoicedate",
      "type": "datetime"
    }}
  ]
}}

Rules:
- Use only the columns present in the dataset.
- Provide clear, concise, and actionable names.
- Make all KPIs and charts computable from the dataset.
- If uncertain, set domain = "generic".

Dataset profile JSON:
{json.dumps(profile)}
"""



def request_dashboard_plan_hugging_face(profile: dict, model_name, api_key) -> dict:
    """
    Send profile JSON to a Hugging Face LLM and get structured dashboard plan.
    Retries if JSON is invalid or API fails.
    """
    API_KEY = api_key
    MODEL = model_name  # Or any LLM model on Hugging Face

    client = InferenceClient(token=API_KEY)
    prompt = f"""
    You are a data analytics assistant that outputs **JSON only**.
    The input profile is: {json.dumps(profile)}
    """

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.text_generation(
                model=MODEL,
                prompt=prompt,                 # <-- corrected here
                max_new_tokens=500,
                temperature=0
            )
            content = response.generated_text.strip()
            # Validate JSON
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            print(f"⚠️ JSON invalid, retrying... ({attempt}/{MAX_RETRIES})")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ API call failed: {str(e)}, retrying... ({attempt}/{MAX_RETRIES})")
            time.sleep(1)

    raise ValueError("Failed to get valid JSON from Hugging Face after retries.")

def request_dashboard_plan_openai(profile, model_name, api_key):
    """Send profile JSON to LLM and get structured dashboard plan."""

    API_KEY = api_key
    MODEL = model_name

    client = OpenAI(api_key=API_KEY)
    prompt = build_prompt(profile)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a data analytics assistant that outputs JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()

            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

            # Validate JSON
            json.loads(content)
            return content
        except json.JSONDecodeError:
            print(f"⚠️ JSON invalid, retrying... ({attempt+1}/{MAX_RETRIES})")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ API call failed: {str(e)}")
            time.sleep(1)

    raise ValueError("Failed to get valid JSON from LLM after retries.")

def save_dashboard_plan(content, output_file):
    """Save LLM response as JSON file if valid."""
    try:
        plan = json.loads(content)
        with open(output_file, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"✅ Dashboard plan saved to {output_file}")
    except json.JSONDecodeError:
        print("❌ LLM output is not valid JSON. Raw output printed below:\n")
        print(content)

if __name__ == "__main__":
    profile = load_profile(INPUT_FILE)
    
    yaml_data = {}

    with open('llm_req.yaml', "r") as f:
        yaml_data = yaml.safe_load(f)

    model_to_use = yaml_data['model_to_use']

    model_name = yaml_data[model_to_use]['model_name']
    api_key = yaml_data[model_to_use]['api_key']

    print("using ", model_to_use , model_name)

    if model_to_use == 'huggingface' :
        raw_output = request_dashboard_plan_hugging_face(profile, model_name, api_key)

    elif model_to_use == 'openai' : 
        raw_output = request_dashboard_plan_openai(profile, model_name, api_key)


    save_dashboard_plan(raw_output, OUTPUT_FILE)