"""
Main pipeline orchestrator following SOLID principles
Single entry point for the entire pipeline
"""

import sys
import os
import subprocess
from src.config import ConfigManager
from src.data_loader import CSVDataLoader
from src.data_profiler import DataProfiler
from src.llm_client import LLMClient
import time

def main():
    """Main pipeline execution"""
    print("🚀 Starting AI Analytics Dashboard Pipeline")
    for path in [
    "data/processed/data_cleaned.csv",
    "outputs/profile_summary.json",
    "outputs/dashboard_plan.json",
    "outputs/dashboard_cache.json"
    ]:
     
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️  Deleted {path}")
    print("=" * 60)
    
    # Load configuration
    config = ConfigManager()
    print(f"📋 Configuration loaded")
    
    # Get input file from command line or config
    if len(sys.argv) > 1:
        config.data.input_file = sys.argv[1]
    
    if not os.path.exists(config.data.input_file):
        print(f"❌ Input file not found: {config.data.input_file}")
        print("Usage: python main.py <csv_file_path>")
        return False
    
    start_time = time.time()
    
    # Step 1: Load and process data
    print(f"\n📂 Step 1: Loading data from {config.data.input_file}")
    data_loader = CSVDataLoader()
    
    success, message = data_loader.load_csv(config.data.input_file)
    if not success:
        print(f"❌ Data loading failed: {message}")
        return False
    
    print(f"✅ {message}")
    
    # Step 2: Profile data
    print(f"\n📊 Step 2: Profiling dataset")
    profiler = DataProfiler(data_loader)
    
    if not profiler.profile_dataset():
        print("❌ Data profiling failed")
        return False
    
    if not profiler.save_profile(config.data.profile_file):
        print("❌ Failed to save profile")
        return False
    
    print(f"✅ Profile saved to {config.data.profile_file}")
    
    # Step 3: Generate dashboard plan
    print(f"\n🤖 Step 3: Generating dashboard plan")
    llm_client = LLMClient(config.llm, config.data.cache_file)
    
    dashboard_plan = llm_client.generate_dashboard_plan(profiler.get_summary())
    if not dashboard_plan:
        print("❌ Dashboard plan generation failed")
        return False
    
    # Save dashboard plan
    try:
        os.makedirs(os.path.dirname(config.data.dashboard_plan_file), exist_ok=True)
        with open(config.data.dashboard_plan_file, 'w') as f:
            import json
            json.dump(dashboard_plan, f, indent=2)
        print(f"✅ Dashboard plan saved to {config.data.dashboard_plan_file}")
    except Exception as e:
        print(f"❌ Failed to save dashboard plan: {e}")
        return False
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n🎉 Pipeline completed successfully!")
    # print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Generated files:")
    print(f"   • {config.data.profile_file}")
    print(f"   • {config.data.dashboard_plan_file}")
    print(f"   • data/processed/data_cleaned.csv")
    print(f"\n💡 Next step: Run dashboard with 'streamlit run dashboard_app.py'")

    print("\n💡 Launching dashboard with 'streamlit run dashboard_app.py'...\n")
    subprocess.run(["streamlit", "run", "dashboard_app.py"])
 
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)