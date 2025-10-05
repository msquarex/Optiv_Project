#!/usr/bin/env python3
"""
Demo script showing image processing with OCR and Gemma3 description
"""

import sys
import os
sys.path.append('src')

from main_workflow import CyberSimulationWorkflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_image_processing():
    """Demo image processing with OCR and AI description"""
    print("VIT Campus Connect - Image Processing Demo")
    print("=" * 50)
    
    # Check if sample image exists
    sample_image = "demo_sample_files/File_003.png"
    if not os.path.exists(sample_image):
        print(f"ERROR: Sample image not found: {sample_image}")
        return
    
    print(f"Processing image: {sample_image}")
    print("This will extract text using OCR and generate an AI description using Gemma3...")
    print()
    
    # Initialize workflow
    workflow = CyberSimulationWorkflow()
    
    # Test connection first
    print("1. Testing Gemma3 connection...")
    test_result = workflow.file_processor.image_descriptor.test_connection()
    
    if test_result['status'] == 'success':
        print("   SUCCESS: Connected to Gemma3")
        print(f"   Model: {test_result['target_model']}")
    else:
        print("   WARNING: Gemma3 connection failed")
        print(f"   Error: {test_result.get('error', 'Unknown error')}")
        print("   OCR will still work, but AI description will fail")
    
    print()
    
    # Process the image
    print("2. Processing image...")
    result = workflow.process_single_file(sample_image)
    
    if result['status'] == 'success':
        print("   SUCCESS: Image processed successfully!")
        
        # Show results
        proc_result = result['processing_results'][0]
        
        print()
        print("3. Results:")
        print("-" * 30)
        
        # OCR Results
        if 'ocr_text' in proc_result:
            print("OCR EXTRACTED TEXT:")
            print(proc_result['ocr_text'])
            print()
        
        # AI Description Results
        if 'image_description' in proc_result:
            desc_status = proc_result.get('description_status', 'error')
            if desc_status == 'success':
                print("AI IMAGE DESCRIPTION (Gemma3):")
                print(proc_result['image_description'])
            else:
                print("AI DESCRIPTION ERROR:")
                print(proc_result.get('description_error', 'Unknown error'))
        
        print()
        print("4. Combined Content:")
        print("-" * 30)
        print(proc_result['content'])
        
    else:
        print("   ERROR: Image processing failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    demo_image_processing()
