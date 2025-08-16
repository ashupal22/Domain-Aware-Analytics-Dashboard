import json
import time
import hashlib
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from huggingface_hub import InferenceClient

class LLMCache:
    """Simple caching system for LLM responses"""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_cache_key(self, profile: Dict) -> str:
        """Generate cache key from profile data"""
        cache_data = {
            'columns': [col['name'] for col in profile.get('columns', [])],
            'shape': profile.get('shape', {})
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()[:16]
    
    def get(self, profile: Dict) -> Optional[Dict]:
        cache_key = self.get_cache_key(profile)
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get('dashboard_plan')
        return None
    
    def set(self, profile: Dict, dashboard_plan: Dict):
        cache_key = self.get_cache_key(profile)
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'dashboard_plan': dashboard_plan
        }
        self._save_cache()

class LLMClient:
    """Optimized LLM client with correct KPI and Chart format"""
    
    def __init__(self, config, cache_file: str = "outputs/dashboard_cache.json"):
        self.config = config
        self.cache = LLMCache(cache_file) if config.cache_enabled else None
    
    def generate_dashboard_plan(self, profile: Dict) -> Optional[Dict]:
        """Generate dashboard plan from profile - FASTEST approach"""
        # Check cache first
        if self.cache:
            cached_plan = self.cache.get(profile)
            if cached_plan:
                print("⚡ Using cached dashboard plan")
                return cached_plan
        
        try:
            # Step 1: Quick domain detection
            domain_info = self._detect_domain_fast(profile)
            print(f"🎯 Detected domain: {domain_info.get('domain_name', 'Generic')}")
            
            # Step 2: Generate optimized dashboard plan
            dashboard_plan = self._generate_optimized_plan(profile, domain_info)
            
            # Cache result
            if self.cache:
                self.cache.set(profile, dashboard_plan)
            
            return dashboard_plan
            
        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            return self._generate_fallback_plan(profile)
    
    def _detect_domain_fast(self, profile: Dict) -> Dict:
        """Fast domain detection - LLM or heuristic"""
        try:
            if not self.config.api_key or self.config.api_key == "your-api-key-here":
                print("⚠️ API key not configured, using heuristic domain detection")
                return {"domain_name": "Generic"}
            
            # Quick LLM domain detection
            domain_prompt = self._build_domain_prompt_fast(profile)
            
            if self.config.provider == 'openai':
                result = self._request_openai_domain(domain_prompt)
            else:
                result = self._request_huggingface_domain(domain_prompt)
            
            domain_info = json.loads(result.strip())
            if 'domain_name' not in domain_info:
                return {"domain_name": "Generic"}
            
            return domain_info
            
        except Exception as e:
            print(f"⚠️ Domain detection failed, using heuristic: {e}")
            return {"domain_name": "Generic"}
    
    def _build_domain_prompt_fast(self, profile: Dict) -> str:
        """Build fast domain detection prompt"""
        columns = profile.get('columns', [])
        shape = profile.get('shape', {})
        
        # Quick column analysis
        column_names = [col['name'] for col in columns[:10]]
        
        return f"""Analyze this dataset and determine the domain(e.g., e-commerce, manufacturing, SaaS, finance, or generic).

DATA:
- Columns: {', '.join(column_names)}
- Shape: {shape.get('rows', 0)} rows, {shape.get('columns', 0)} columns

Return ONLY JSON:
{{
  "domain_name": "Specific Domain Name (1-3 words)(e.g., e-commerce, manufacturing, SaaS, finance, or generic)"
}}"""
    
    def _generate_optimized_plan(self, profile: Dict, domain_info: Dict) -> Dict:
        """Generate optimized dashboard plan with correct KPI/Chart format"""
        if not self.config.api_key or self.config.api_key == "your-api-key-here":
            print("⚠️ API key not configured, generating fallback plan")
            return self._generate_fallback_plan(profile, domain_info)
        
        try:
            prompt = self._build_optimized_prompt(profile, domain_info)
            
            if self.config.provider == 'openai':
                result = self._request_openai_comprehensive(prompt)
            else:
                result = self._request_huggingface_comprehensive(prompt)
            
            return json.loads(result.strip())
        except Exception as e:
            print(f"⚠️ Plan generation failed: {e}")
            return self._generate_fallback_plan(profile, domain_info)
    
    def _build_optimized_prompt(self, profile: Dict, domain_info: Dict) -> str:
        """Build optimized prompt with correct KPI and Chart format"""
        columns = profile.get('columns', [])
        shape = profile.get('shape', {})
        domain_name = domain_info.get('domain_name', 'Data Analytics')
        
        # Analyze columns efficiently
        date_cols = [col['name'] for col in columns if col['type'] == 'datetime']
        numeric_cols = [col['name'] for col in columns if col['type'] in ['integer', 'float', 'numeric_string']]
        categorical_cols = [col['name'] for col in columns if col['type'] == 'categorical']
        
        return f"""Create a Streamlit dashboard plan for {domain_name}.

DATASET:
- Domain: {domain_name}
- Shape: {shape.get('rows', 0)} rows, {shape.get('columns', 0)} columns
- Date columns: {date_cols[:3]}
- Numeric columns: {numeric_cols[:5]}
- Categorical columns: {categorical_cols[:3]}

TASK 1 - Generate 2-3 TOP KPIs relevant to {domain_name}:
Each KPI must have:
- name: short, clear name
- description: what it measures  
- formula: must be a DataFrame-level operation that produces a single aggregated value (e.g., using sum, mean, count, ratio of column aggregates), not a row-wise expression
- importance: 0-10 (10 = must-have)
- reason: 30-40 words explaining why this KPI was selected

TASK 2 - Generate 6-8 CHARTS for interactive dashboard:
Each chart must have:
- name: short name
- importance: 0-10 (10 = must-have)  
- type: one of [time_series, bar, histogram, scatter, heatmap, pie, funnel]
- x_column: exact column name for x-axis
- y_column: exact column name for y-axis (if applicable)
- description: what insights this chart provides

STREAMLIT CHART TEMPLATES AVAILABLE:
- time_series: line chart over time (requires date column)
- bar: categorical breakdown 
- histogram: distribution of values
- scatter: correlation between two numeric variables
- heatmap: correlation matrix of numeric columns
- pie: percentage breakdown of categories  
- funnel: conversion steps (if applicable)

OUTPUT FORMAT:
{{
  "domain": {{
    "name": "{domain_name}"
  }},
  "kpis": [
    {{
      "name": "KPI Name",
      "description": "What this KPI measures",
      "formula": "df['column'].sum()",
      "importance": 10,
      "reason": "30-40 word explanation of why this KPI is important for {domain_name}"
    }}
  ],
  "charts": [
    {{
      "name": "Chart Name", 
      "importance": 10,
      "type": "time_series|bar|histogram|scatter|heatmap|pie|funnel",
      "x_column": "exact_column_name",
      "y_column": "exact_column_name",
      "description": "What insights this chart provides"
    }}
  ],
  "filters": [
    {{
      "name": "Filter Name",
      "column": "exact_column_name", 
      "type": "date|multiselect|slider",
      "importance": 10
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. Use ONLY exact column names from the dataset
2. KPIs must be relevant to {domain_name} domain
3. Chart types must match available Streamlit templates
4. All formulas must use existing columns only
5. Return ONLY valid JSON

Generate the dashboard plan:"""
    
   
    def _generate_fallback_plan(self, profile: Dict, domain_info: Dict = None) -> Dict:
        """Generate optimized fallback plan with correct format"""
        if domain_info is None:
            domain_info = self._heuristic_domain_detection(profile)
        
        columns = profile.get('columns', [])
        domain_name = domain_info.get('domain_name', 'Data Analytics')
        
        # Quick column analysis
        date_cols = [col for col in columns if col['type'] == 'datetime']
        numeric_cols = [col for col in columns if col['type'] in ['integer', 'float', 'numeric_string']]
        categorical_cols = [col for col in columns if col['type'] == 'categorical']
        
        # Generate KPIs with correct format
        kpis = []
        if numeric_cols:
            col = numeric_cols[0]
            kpis.append({
                "name": f"Total {col['name'].title()}",
                "description": f"Sum of all {col['name']} values in the dataset",
                "formula": f"df['{col['name']}'].sum()",
                "importance": 10,
                "reason": f"Key metric showing overall volume and scale of {col['name']} across all records, essential for understanding business magnitude."
            })
        
        if len(numeric_cols) > 1:
            col = numeric_cols[1]
            kpis.append({
                "name": f"Average {col['name'].title()}",
                "description": f"Mean value of {col['name']} per record",
                "formula": f"df['{col['name']}'].mean()",
                "importance": 8,
                "reason": f"Critical performance indicator showing typical {col['name']} values, helps identify trends and benchmark against targets."
            })
        
        # Generate charts with correct format
        charts = []
        
        # Time series if date + numeric available
        if date_cols and numeric_cols:
            charts.append({
                "name": f"{numeric_cols[0].title()} Trend",
                "importance": 10,
                "type": "time_series",
                "x_column": date_cols[0]['name'],
                "y_column": numeric_cols[0]['name'],
                "description": f"Shows how {numeric_cols[0]} changes over time, revealing trends and patterns"
            })
        
        # Bar chart if categorical + numeric available
        if categorical_cols and numeric_cols:
            charts.append({
                "name": f"{numeric_cols[0].title()} by {categorical_cols[0]['name'].title()}",
                "importance": 9,
                "type": "bar",
                "x_column": categorical_cols[0]['name'],
                "y_column": numeric_cols[0]['name'],
                "description": f"Compares {numeric_cols[0]} across different {categorical_cols[0]['name']} categories"
            })
        
        # Histogram for distribution
        if numeric_cols:
            charts.append({
                "name": f"{numeric_cols[0].title()} Distribution",
                "importance": 7,
                "type": "histogram", 
                "x_column": numeric_cols[0]['name'],
                "y_column": None,
                "description": f"Shows the distribution pattern of {numeric_cols[0]} values"
            })
        
        # Scatter plot if 2+ numeric columns
        if len(numeric_cols) >= 2:
            charts.append({
                "name": f"{numeric_cols[0].title()} vs {numeric_cols[1].title()}",
                "importance": 6,
                "type": "scatter",
                "x_column": numeric_cols[0]['name'],
                "y_column": numeric_cols[1]['name'],
                "description": f"Explores relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            })
        
        # Generate filters
        filters = []
        
        if date_cols:
            filters.append({
                "name": f"{date_cols[0]['name'].title()} Filter",
                "column": date_cols[0]['name'],
                "type": "date",
                "importance": 10
            })
        
        if categorical_cols:
            filters.append({
                "name": f"{categorical_cols[0]['name'].title()} Filter", 
                "column": categorical_cols[0]['name'],
                "type": "multiselect",
                "importance": 8
            })
        
        return {
            "domain": {
                "name": domain_name
            },
            "kpis": kpis,
            "charts": charts,
            "filters": filters
        }
    
    def _request_openai_domain(self, prompt: str) -> str:
        """Fast OpenAI domain detection"""
        client = OpenAI(api_key=self.config.api_key)
        
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a business domain expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200  # Reduced for speed
        )
        
        return self._clean_json_response(response.choices[0].message.content)
    
    def _request_openai_comprehensive(self, prompt: str) -> str:
        """Fast OpenAI comprehensive plan"""
        client = OpenAI(api_key=self.config.api_key)
        
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a dashboard expert. Create KPIs and charts with exact format. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2500  # Optimized for speed
        )
        
        return self._clean_json_response(response.choices[0].message.content)
    
    def _request_huggingface_domain(self, prompt: str) -> str:
        """Fast Hugging Face domain detection"""
        client = InferenceClient(token=self.config.api_key)
        
        response = client.text_generation(
            model=self.config.model_name,
            prompt=f"You are a business analyst. {prompt}",
            max_new_tokens=150,  # Reduced for speed
            temperature=0
        )
        
        return self._clean_json_response(response)
    
    def _request_huggingface_comprehensive(self, prompt: str) -> str:
        """Fast Hugging Face comprehensive plan"""
        client = InferenceClient(token=self.config.api_key)
        
        response = client.text_generation(
            model=self.config.model_name,
            prompt=f"You are a dashboard expert. {prompt}",
            max_new_tokens=1500,  # Optimized for speed
            temperature=0
        )
        
        return self._clean_json_response(response)
    
    def _clean_json_response(self, response: str) -> str:
        """Fast JSON cleaning and validation"""
        response = response.strip()
        response = response.replace("```json", "").replace("```", "")
        
        # Quick JSON extraction
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx + 1]
        
        # Fast validation
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")