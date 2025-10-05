#!/usr/bin/env python3
"""
Startup script for the Streamlit frontend
"""

import subprocess
import sys
import os
import warnings

def main():
    """Run the Streamlit application"""
    
    # Suppress PyTorch warnings that can interfere with Streamlit
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    
    # Set environment variables to reduce PyTorch interference
    os.environ["TORCH_LOGS"] = "-dynamo"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Please install it with: pip install streamlit")
        return 1
    
    # Check if src directory exists
    if not os.path.exists('src'):
        print("❌ src directory not found. Please run this script from the project root.")
        return 1
    
    print("🚀 Starting VIT Campus Connect Streamlit App...")
    print("📱 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run streamlit with additional flags to avoid conflicts
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
            "--logger.level", "error"
        ]
        
        # Set environment for subprocess
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["TORCH_LOGS"] = "-dynamo"
        
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        print("💡 Try running: streamlit run streamlit_app.py --server.port 8501")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
