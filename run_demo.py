#!/usr/bin/env python3
"""
Demo runner script for VIT Campus Connect - Cyber Simulation Exercise
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from main_workflow import CyberSimulationWorkflow

def main():
    """Main demo function"""
    print("=" * 60)
    print("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE")
    print("Automating File Cleansing and Analysis leveraging AI")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize the workflow
        print("[INFO] Initializing system...")
        workflow = CyberSimulationWorkflow(output_dir="demo_output")
        print("[SUCCESS] System initialized successfully!")
        print()
        
        # Run the demo
        print("[INFO] Starting demo with sample files...")
        results = workflow.run_demo("demo_sample_files")
        
        if results['status'] == 'success':
            print("[SUCCESS] Demo completed successfully!")
            print()
            
            # Display results summary
            stats = results['processing_stats']
            print("DEMO RESULTS SUMMARY:")
            print("-" * 30)
            print(f"Total Files Processed: {stats['total_files']}")
            print(f"Successful Files: {stats['successful_files']}")
            print(f"Failed Files: {stats['failed_files']}")
            print(f"PII Detections: {stats['total_pii_detections']}")
            print(f"Security Findings: {stats['total_security_findings']}")
            print(f"LLM Analyses: {stats['total_llm_analyses']}")
            print()
            
            # Show output files
            if results['output_files']:
                print("Generated Output Files:")
                for file_type, file_path in results['output_files'].items():
                    print(f"   - {file_type}: {file_path}")
                print()
            
            # Show processing time
            if stats['start_time'] and stats['end_time']:
                duration = (stats['end_time'] - stats['start_time']).total_seconds()
                print(f"Total Processing Time: {duration:.2f} seconds")
                print()
            
            print("[SUCCESS] Demo completed successfully!")
            print("Check the 'demo_output' directory for generated files.")
            
        else:
            print(f"[ERROR] Demo failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Error running demo: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
