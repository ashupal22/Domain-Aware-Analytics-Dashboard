

### Clone and setup Pipeline:
```
git clone <repository-url>
cd ai-analytics-dashboard
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the pipeline:
```
python main.py your-file.csv
```

##### Launch dashboard Only:
````
streamlit run dashboard_app.py
````

### 🏗️ Architecture
#### The pipeline follows SOLID principles with a modular, extensible design:
```

📦 AI Analytics Dashboard Pipeline
├── 🔧 Data Processing Layer
│   ├── CSVDataLoader - Handles file reading, encoding detection, preprocessing
│   └── DataProfiler - Analyzes data types, patterns, and quality
├── 🤖 AI Analysis Layer
│   └── LLMClient - Generates dashboard plans using OpenAI/HuggingFace
├── 📊 Visualization Layer
│   └── ClassicDashboard - Renders interactive Streamlit dashboard
└── ⚙️ Configuration Layer
    └── ConfigManager - Manages settings and API keys#
```
##### Project Structure

```
.
├── config/
│   └── settings.yaml          # Configuration file
├── dashboard_app.py           # Main Streamlit dashboard
├── data/
│   ├── raw/                   # Original CSV files
│   ├── processed/             # Cleaned data
│   └── cache/                 # Temporary files
├── main.py                    # Pipeline orchestrator
├── outputs/
│   ├── dashboard_plan.json    # AI-generated dashboard config
│   ├── profile_summary.json   # Data analysis results
│   └── dashboard_cache.json   # Cached AI responses
├── src/
│   ├── config.py             # Configuration management
│   ├── data_loader.py        # CSV loading and preprocessing
│   ├── data_profiler.py      # Data analysis and profiling
│   └── llm_client.py         # AI client for dashboard generation
└── requirements.txt
```

##### Dashboard
 
 ![alt text](<images/Screenshot 2025-08-17 at 1.13.35 AM.png>) 
 ![alt text](<images/Screenshot 2025-08-17 at 1.13.26 AM.png>) 
 ![alt text](<images/Screenshot 2025-08-17 at 1.13.20 AM.png>) 
 ![alt text](<images/Screenshot 2025-08-17 at 1.13.11 AM.png>)

