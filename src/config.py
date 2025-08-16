"""
Configuration management following Single Responsibility Principle
"""
import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataConfig:
    input_file: str = "data/raw/data.csv"
    processed_file: str = "data/processed/data_cleaned.csv"
    profile_file: str = "outputs/profile_summary.json"
    dashboard_plan_file: str = "outputs/dashboard_plan.json"
    cache_file: str = "outputs/dashboard_cache.json"

@dataclass
class LLMConfig:
    provider: str = "openai"
    model_name: str = "gpt-4"
    api_key: str = ""
    max_retries: int = 3
    timeout: int = 30

@dataclass
class AppConfig:
    debug: bool = False
    cache_enabled: bool = True
    auto_refresh: bool = False

class ConfigManager:
    """Manages all configuration settings"""
    
    def __init__(self, config_file: str = "config/settings.yaml"):
        self.config_file = config_file
        self.data = DataConfig()
        self.llm = LLMConfig()
        self.app = AppConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update configurations
                if 'data' in config:
                    for key, value in config['data'].items():
                        setattr(self.data, key, value)
                
                if 'llm' in config:
                    for key, value in config['llm'].items():
                        setattr(self.llm, key, value)
                
                if 'app' in config:
                    for key, value in config['app'].items():
                        setattr(self.app, key, value)
                        
        except Exception as e:
            print(f"⚠️ Config loading error: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration to file"""
        config = {
            'data': self.data.__dict__,
            'llm': self.llm.__dict__,
            'app': self.app.__dict__
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)