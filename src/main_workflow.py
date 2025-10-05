"""
Main Workflow Module for VIT Campus Connect
Orchestrates the complete file cleansing and analysis pipeline
"""

import os
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time

from file_processor import FileProcessor
from pii_detector import PIIDetector
from security_analyzer import SecurityAnalyzer
from output_generator import OutputGenerator
from llm_analyzer import LLMAnalyzer
from progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyberSimulationWorkflow:
    """Main workflow class for the cyber simulation exercise"""
    
    def __init__(self, output_dir: str = "output", gemma_url: str = "http://127.0.0.1:11500"):
        self.file_processor = FileProcessor(gemma_url)
        self.pii_detector = PIIDetector()
        self.security_analyzer = SecurityAnalyzer()
        self.llm_analyzer = LLMAnalyzer(gemma_url)
        self.output_generator = OutputGenerator(output_dir)
        
        # Progress tracking
        self.progress_tracker: Optional[ProgressTracker] = None
        self.progress_callback: Optional[Callable] = None
        
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_pii_detections': 0,
            'total_security_findings': 0,
            'total_llm_analyses': 0
        }
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function to receive progress updates"""
        self.progress_callback = callback
    
    def process_files(self, file_paths: List[str], enable_image_descriptions: bool = True) -> Dict[str, Any]:
        """
        Main method to process files through the complete pipeline
        
        Args:
            file_paths (List[str]): List of file paths to process
            enable_image_descriptions (bool): Whether to enable AI image descriptions for image files
            
        Returns:
            Dict containing complete processing results
        """
        try:
            self.processing_stats['start_time'] = datetime.now()
            self.processing_stats['total_files'] = len(file_paths)
            
            # Initialize progress tracker
            self.progress_tracker = ProgressTracker(len(file_paths))
            if self.progress_callback:
                self.progress_tracker.set_progress_callback(self.progress_callback)
            
            logger.info(f"Starting processing of {len(file_paths)} files")
            self.progress_tracker.start_step("Processing Files")
            
            # Step 1: File Processing
            logger.info("Step 1: Processing files and extracting content...")
            processing_results = []
            for i, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)
                self.progress_tracker.start_file_processing(file_name, i)
                
                # Process individual file
                result = self.file_processor.process_file(file_path, enable_image_descriptions=enable_image_descriptions)
                processing_results.append(result)
                
                # Update progress
                success = result['status'] == 'success'
                self.progress_tracker.complete_file_processing(file_name, success, {
                    'file_type': result.get('file_type', 'Unknown'),
                    'content_length': len(result.get('content', '')),
                    'error': result.get('error') if not success else None
                })
            
            self.processing_stats['successful_files'] = sum(1 for r in processing_results if r['status'] == 'success')
            self.processing_stats['failed_files'] = self.processing_stats['total_files'] - self.processing_stats['successful_files']
            self.progress_tracker.complete_step("Processing Files")
            
            # Step 2: PII Detection and Masking
            logger.info("Step 2: Detecting and masking PII...")
            self.progress_tracker.start_step("Detecting PII")
            pii_results = []
            for i, result in enumerate(processing_results):
                file_name = os.path.basename(result.get('file_path', f'file_{i}'))
                self.progress_tracker.start_file_processing(f"PII Detection: {file_name}", i)
                
                if result['status'] == 'success':
                    pii_result = self.pii_detector.detect_and_mask_pii(result['content'])
                    pii_count = pii_result.get('detection_count', 0)
                    self.processing_stats['total_pii_detections'] += pii_count
                    self.progress_tracker.update_pii_detection(pii_count)
                    self.progress_tracker.complete_file_processing(f"PII Detection: {file_name}", True, {
                        'pii_items_found': pii_count,
                        'detected_types': [item.get('type', 'unknown') for item in pii_result.get('detected_items', [])]
                    })
                else:
                    pii_result = {
                        'original_text': '',
                        'masked_text': '',
                        'detected_items': [],
                        'detection_count': 0,
                        'status': 'error',
                        'error': 'File processing failed'
                    }
                    self.progress_tracker.complete_file_processing(f"PII Detection: {file_name}", False, {
                        'error': 'File processing failed'
                    })
                pii_results.append(pii_result)
            self.progress_tracker.complete_step("Detecting PII")
            
            # Step 3: Security Analysis
            logger.info("Step 3: Analyzing security content...")
            self.progress_tracker.start_step("Analyzing Security Content")
            analysis_results = []
            for i, pii_result in enumerate(pii_results):
                file_name = os.path.basename(processing_results[i].get('file_path', f'file_{i}'))
                self.progress_tracker.start_file_processing(f"Security Analysis: {file_name}", i)
                
                if pii_result['status'] == 'success':
                    analysis_result = self.security_analyzer.analyze_security_content(pii_result['masked_text'])
                    # Count security findings
                    if analysis_result['status'] == 'success':
                        summary = analysis_result['analysis']['summary']
                        findings_count = (
                            summary.get('total_iam_policies', 0) +
                            summary.get('total_firewall_rules', 0) +
                            summary.get('total_ids_logs', 0) +
                            summary.get('total_vulnerabilities', 0)
                        )
                        self.processing_stats['total_security_findings'] += findings_count
                        self.progress_tracker.update_security_findings(findings_count)
                        self.progress_tracker.complete_file_processing(f"Security Analysis: {file_name}", True, {
                            'security_findings': findings_count,
                            'iam_policies': summary.get('total_iam_policies', 0),
                            'firewall_rules': summary.get('total_firewall_rules', 0),
                            'ids_logs': summary.get('total_ids_logs', 0),
                            'vulnerabilities': summary.get('total_vulnerabilities', 0)
                        })
                    else:
                        self.progress_tracker.complete_file_processing(f"Security Analysis: {file_name}", False, {
                            'error': analysis_result.get('error', 'Security analysis failed')
                        })
                else:
                    analysis_result = {
                        'text': '',
                        'analysis': {},
                        'status': 'error',
                        'error': 'PII processing failed'
                    }
                    self.progress_tracker.complete_file_processing(f"Security Analysis: {file_name}", False, {
                        'error': 'PII processing failed'
                    })
                analysis_results.append(analysis_result)
            self.progress_tracker.complete_step("Analyzing Security Content")
            
            # Step 4: LLM Analysis for File Descriptions and Security Findings
            logger.info("Step 4: Performing LLM analysis for file descriptions and security findings...")
            self.progress_tracker.start_step("LLM Analysis")
            llm_analysis_results = []
            for i, (proc_result, pii_result) in enumerate(zip(processing_results, pii_results)):
                file_name = os.path.basename(proc_result.get('file_path', f'file_{i}'))
                self.progress_tracker.start_file_processing(f"LLM Analysis: {file_name}", i)
                
                if proc_result['status'] == 'success':
                    # Prepare file data for LLM analysis
                    file_data = {
                        'file_path': proc_result['file_path'],
                        'file_type': proc_result['file_type'],
                        'content': pii_result.get('masked_text', proc_result.get('content', ''))
                    }
                    
                    # Perform LLM analysis
                    llm_result = self.llm_analyzer.analyze_file_content(file_data)
                    if llm_result['status'] == 'success':
                        self.processing_stats['total_llm_analyses'] += 1
                        self.progress_tracker.update_llm_analyses(1)
                        self.progress_tracker.complete_file_processing(f"LLM Analysis: {file_name}", True, {
                            'file_description_length': len(llm_result.get('file_description', '')),
                            'key_findings_count': len(llm_result.get('key_findings', [])),
                            'analysis_timestamp': llm_result.get('analysis_timestamp', '')
                        })
                    else:
                        self.progress_tracker.complete_file_processing(f"LLM Analysis: {file_name}", False, {
                            'error': llm_result.get('error', 'LLM analysis failed')
                        })
                else:
                    llm_result = {
                        'status': 'error',
                        'file_name': proc_result.get('file_path', 'Unknown').split('/')[-1].split('\\')[-1].split('.')[0],
                        'file_type': proc_result.get('file_type', 'Unknown'),
                        'file_path': proc_result.get('file_path', 'Unknown'),
                        'file_description': 'Analysis failed - file processing error',
                        'key_findings': ['Error: File processing failed'],
                        'error': 'File processing failed'
                    }
                    self.progress_tracker.complete_file_processing(f"LLM Analysis: {file_name}", False, {
                        'error': 'File processing failed'
                    })
                llm_analysis_results.append(llm_result)
            self.progress_tracker.complete_step("LLM Analysis")
            
            # Step 5: Generate Outputs
            logger.info("Step 5: Generating output files...")
            self.progress_tracker.start_step("Generating Outputs")
            output_files = self.output_generator.generate_all_outputs(
                processing_results, pii_results, analysis_results, llm_analysis_results
            )
            self.progress_tracker.complete_step("Generating Outputs")
            
            self.processing_stats['end_time'] = datetime.now()
            processing_duration = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
            
            # Complete the entire process
            self.progress_tracker.start_step("Completed")
            
            # Generate final results
            final_results = {
                'processing_stats': self.processing_stats,
                'processing_duration_seconds': processing_duration,
                'processing_results': processing_results,
                'pii_results': pii_results,
                'analysis_results': analysis_results,
                'llm_analysis_results': llm_analysis_results,
                'output_files': output_files,
                'progress_summary': self.progress_tracker.get_progress_summary(),
                'status': 'success'
            }
            
            logger.info(f"Processing completed successfully in {processing_duration:.2f} seconds")
            logger.info(f"Generated {len(output_files)} output files")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in main workflow: {str(e)}")
            return {
                'processing_stats': self.processing_stats,
                'processing_results': [],
                'pii_results': [],
                'analysis_results': [],
                'llm_analysis_results': [],
                'output_files': {},
                'status': 'error',
                'error': str(e)
            }
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline
        
        Args:
            file_path (str): Path to the file to process
            
        Returns:
            Dict containing processing results for the single file
        """
        return self.process_files([file_path])
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics"""
        if self.processing_stats['start_time'] and self.processing_stats['end_time']:
            duration = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
        else:
            duration = 0
        
        return {
            'total_files': self.processing_stats['total_files'],
            'successful_files': self.processing_stats['successful_files'],
            'failed_files': self.processing_stats['failed_files'],
            'success_rate': (self.processing_stats['successful_files'] / self.processing_stats['total_files'] * 100) if self.processing_stats['total_files'] > 0 else 0,
            'total_pii_detections': self.processing_stats['total_pii_detections'],
            'total_security_findings': self.processing_stats['total_security_findings'],
            'total_llm_analyses': self.processing_stats['total_llm_analyses'],
            'processing_duration_seconds': duration,
            'start_time': self.processing_stats['start_time'].isoformat() if self.processing_stats['start_time'] else None,
            'end_time': self.processing_stats['end_time'].isoformat() if self.processing_stats['end_time'] else None
        }
    
    def validate_file_paths(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Validate file paths and categorize them
        
        Args:
            file_paths (List[str]): List of file paths to validate
            
        Returns:
            Dict with valid and invalid file paths
        """
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in self.file_processor.supported_formats:
                    valid_files.append(file_path)
                else:
                    invalid_files.append(f"{file_path} (unsupported format: {file_ext})")
            else:
                invalid_files.append(f"{file_path} (file not found)")
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files
        }
    
    def create_sample_files(self, sample_dir: str = "sample_files") -> List[str]:
        """
        Create sample files for testing the system
        
        Args:
            sample_dir (str): Directory to create sample files in
            
        Returns:
            List of created sample file paths
        """
        try:
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            
            sample_files = []
            
            # Create sample text file
            text_file = os.path.join(sample_dir, "sample_security_log.txt")
            with open(text_file, 'w') as f:
                f.write("""Security Log Sample
==================

2024-01-15 10:30:45 - CRITICAL - Firewall Rule Violation
Source IP: 192.168.1.100
Destination IP: 10.0.0.50
Port: 22
Protocol: TCP
Action: DENY
Rule: 1001

2024-01-15 10:31:12 - HIGH - IDS Alert
Signature ID: 12345
Attack Type: SQL Injection
Source: 203.0.113.5
Target: 198.51.100.10
Service: HTTP

IAM Policy Sample:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-bucket/*",
      "Principal": "user:john.doe@company.com"
    }
  ]
}

Contact Information:
John Smith - john.smith@company.com
Phone: (555) 123-4567
SSN: 123-45-6789
""")
            sample_files.append(text_file)
            
            # Create sample CSV file
            csv_file = os.path.join(sample_dir, "sample_firewall_rules.csv")
            with open(csv_file, 'w') as f:
                f.write("""Rule_Number,Action,Protocol,Source_IP,Dest_IP,Port,Description
1,ALLOW,TCP,192.168.1.0/24,10.0.0.0/8,80,Web Traffic
2,DENY,TCP,0.0.0.0/0,10.0.0.50,22,SSH Block
3,ALLOW,UDP,192.168.1.100,8.8.8.8,53,DNS Query
4,DENY,ICMP,203.0.113.0/24,10.0.0.0/8,any,Block ICMP
5,ALLOW,TCP,192.168.1.0/24,10.0.0.100,443,HTTPS Traffic
""")
            sample_files.append(csv_file)
            
            logger.info(f"Created {len(sample_files)} sample files in {sample_dir}")
            return sample_files
            
        except Exception as e:
            logger.error(f"Error creating sample files: {str(e)}")
            return []
    
    def run_demo(self, sample_dir: str = "sample_files") -> Dict[str, Any]:
        """
        Run a complete demo with sample files
        
        Args:
            sample_dir (str): Directory containing sample files
            
        Returns:
            Dict containing demo results
        """
        try:
            logger.info("Starting VIT Campus Connect Demo...")
            
            # Check if directory exists and has files
            if not os.path.exists(sample_dir):
                return {
                    'status': 'error',
                    'error': f'Directory "{sample_dir}" not found'
                }
            
            # Get existing files
            sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                          if os.path.isfile(os.path.join(sample_dir, f))]
            
            if not sample_files:
                return {
                    'status': 'error',
                    'error': f'No files found in directory "{sample_dir}"'
                }
            
            # Process the sample files
            results = self.process_files(sample_files)
            
            # Add demo-specific information
            results['demo_info'] = {
                'sample_files_found': len(sample_files),
                'sample_directory': sample_dir,
                'demo_completed_at': datetime.now().isoformat()
            }
            
            logger.info("Demo completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error running demo: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
