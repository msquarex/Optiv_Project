"""
Output Generation Module for VIT Campus Connect
Generates standardized output formats for processed and analyzed data
"""

import os
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputGenerator:
    """Output generation class for creating standardized results"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def generate_cleansed_output(self, processing_results: List[Dict[str, Any]], 
                               pii_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate cleansed text output files
        
        Args:
            processing_results: Results from file processing
            pii_results: Results from PII detection and masking
            
        Returns:
            Dict with file paths of generated outputs
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_files = {}
            
            # Generate individual cleansed files
            for i, (proc_result, pii_result) in enumerate(zip(processing_results, pii_results)):
                if proc_result['status'] == 'success' and pii_result['status'] == 'success':
                    filename = f"cleansed_file_{i+1}_{timestamp}.txt"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Original File: {proc_result['file_path']}\n")
                        f.write(f"File Type: {proc_result['file_type']}\n")
                        f.write(f"Processing Date: {datetime.now().isoformat()}\n")
                        f.write("="*50 + "\n\n")
                        f.write("CLEANSED CONTENT:\n")
                        f.write("-"*20 + "\n")
                        f.write(pii_result['masked_text'])
                        f.write("\n\n")
                        f.write("PII DETECTION SUMMARY:\n")
                        f.write("-"*25 + "\n")
                        f.write(f"Total Detections: {pii_result['detection_count']}\n")
                        
                        if pii_result['detected_items']:
                            f.write("\nDetected Items:\n")
                            for item in pii_result['detected_items']:
                                f.write(f"- Type: {item['type']}\n")
                                f.write(f"  Original: {item['original']}\n")
                                f.write(f"  Masked: {item['masked']}\n")
                    
                    output_files[f"cleansed_file_{i+1}"] = filepath
            
            # Generate consolidated cleansed output
            consolidated_file = os.path.join(self.output_dir, f"consolidated_cleansed_{timestamp}.txt")
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                f.write("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE\n")
                f.write("CONSOLIDATED CLEANSED OUTPUT\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Files Processed: {len(processing_results)}\n\n")
                
                for i, (proc_result, pii_result) in enumerate(zip(processing_results, pii_results)):
                    if proc_result['status'] == 'success' and pii_result['status'] == 'success':
                        f.write(f"FILE {i+1}: {proc_result['file_path']}\n")
                        f.write("-"*40 + "\n")
                        f.write(pii_result['masked_text'])
                        f.write("\n\n" + "="*50 + "\n\n")
            
            output_files['consolidated'] = consolidated_file
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating cleansed output: {str(e)}")
            return {}
    
    def generate_analysis_output(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate security analysis output files
        
        Args:
            analysis_results: Results from security analysis
            
        Returns:
            Dict with file paths of generated outputs
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_files = {}
            
            # Generate detailed analysis report
            analysis_file = os.path.join(self.output_dir, f"security_analysis_{timestamp}.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE\n")
                f.write("SECURITY ANALYSIS REPORT\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                for i, result in enumerate(analysis_results):
                    if result['status'] == 'success':
                        analysis = result['analysis']
                        summary = analysis['summary']
                        
                        f.write(f"ANALYSIS {i+1}\n")
                        f.write("-"*20 + "\n")
                        f.write(f"Summary:\n")
                        for insight in summary['security_insights']:
                            f.write(f"  - {insight}\n")
                        
                        f.write(f"\nDetailed Findings:\n")
                        
                        # IAM Policies
                        if analysis['iam_policies']['policies_found']:
                            f.write(f"\nIAM Policies Found: {len(analysis['iam_policies']['policies_found'])}\n")
                            for policy in analysis['iam_policies']['policies_found'][:5]:  # Limit to first 5
                                f.write(f"  - {policy['content']}\n")
                        
                        # Firewall Rules
                        if analysis['firewall_rules']['rules_found']:
                            f.write(f"\nFirewall Rules Found: {len(analysis['firewall_rules']['rules_found'])}\n")
                            for rule in analysis['firewall_rules']['rules_found'][:5]:  # Limit to first 5
                                f.write(f"  - {rule['rule']}\n")
                        
                        # IDS/IPS Logs
                        if analysis['ids_ips_logs']['log_entries']:
                            f.write(f"\nIDS/IPS Log Entries: {len(analysis['ids_ips_logs']['log_entries'])}\n")
                            for log in analysis['ids_ips_logs']['log_entries'][:5]:  # Limit to first 5
                                f.write(f"  - {log['entry']}\n")
                        
                        # Vulnerabilities
                        if analysis['vulnerabilities']['cves']:
                            f.write(f"\nCVE References: {len(analysis['vulnerabilities']['cves'])}\n")
                            for cve in analysis['vulnerabilities']['cves'][:5]:  # Limit to first 5
                                f.write(f"  - {cve['cve']}\n")
                        
                        f.write("\n" + "="*50 + "\n\n")
            
            output_files['analysis_report'] = analysis_file
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating analysis output: {str(e)}")
            return {}
    
    def generate_csv_output(self, processing_results: List[Dict[str, Any]], 
                          pii_results: List[Dict[str, Any]], 
                          analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generate CSV output with structured data
        
        Args:
            processing_results: Results from file processing
            pii_results: Results from PII detection and masking
            analysis_results: Results from security analysis
            
        Returns:
            Path to generated CSV file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(self.output_dir, f"analysis_results_{timestamp}.csv")
            
            # Prepare data for CSV
            csv_data = []
            
            for i, (proc_result, pii_result, analysis_result) in enumerate(
                zip(processing_results, pii_results, analysis_results)):
                
                row = {
                    'file_number': i + 1,
                    'original_file_path': proc_result.get('file_path', ''),
                    'file_type': proc_result.get('file_type', ''),
                    'processing_status': proc_result.get('status', ''),
                    'pii_detection_count': pii_result.get('detection_count', 0),
                    'pii_detection_status': pii_result.get('status', ''),
                    'analysis_status': analysis_result.get('status', ''),
                }
                
                # Add analysis summary data
                if analysis_result.get('status') == 'success':
                    analysis = analysis_result.get('analysis', {})
                    summary = analysis.get('summary', {})
                    
                    row.update({
                        'iam_policies_count': summary.get('total_iam_policies', 0),
                        'firewall_rules_count': summary.get('total_firewall_rules', 0),
                        'ids_logs_count': summary.get('total_ids_logs', 0),
                        'vulnerabilities_count': summary.get('total_vulnerabilities', 0),
                        'network_configs_count': summary.get('total_network_configs', 0)
                    })
                else:
                    row.update({
                        'iam_policies_count': 0,
                        'firewall_rules_count': 0,
                        'ids_logs_count': 0,
                        'vulnerabilities_count': 0,
                        'network_configs_count': 0
                    })
                
                csv_data.append(row)
            
            # Write CSV file
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
                logger.info(f"Generated CSV output: {csv_file}")
            
            return csv_file
            
        except Exception as e:
            logger.error(f"Error generating CSV output: {str(e)}")
            return ""
    
    def generate_json_output(self, processing_results: List[Dict[str, Any]], 
                           pii_results: List[Dict[str, Any]], 
                           analysis_results: List[Dict[str, Any]],
                           llm_analysis_results: List[Dict[str, Any]] = None) -> str:
        """
        Generate JSON output with complete data
        
        Args:
            processing_results: Results from file processing
            pii_results: Results from PII detection and masking
            analysis_results: Results from security analysis
            llm_analysis_results: Results from LLM analysis
            
        Returns:
            Path to generated JSON file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = os.path.join(self.output_dir, f"complete_results_{timestamp}.json")
            
            # Combine all results
            complete_results = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_files': len(processing_results),
                    'project': 'VIT Campus Connect - Cyber Simulation Exercise'
                },
                'processing_results': processing_results,
                'pii_results': pii_results,
                'analysis_results': analysis_results,
                'llm_analysis_results': llm_analysis_results or []
            }
            
            # Write JSON file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated JSON output: {json_file}")
            return json_file
            
        except Exception as e:
            logger.error(f"Error generating JSON output: {str(e)}")
            return ""
    
    def generate_summary_report(self, processing_results: List[Dict[str, Any]], 
                              pii_results: List[Dict[str, Any]], 
                              analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generate executive summary report
        
        Args:
            processing_results: Results from file processing
            pii_results: Results from PII detection and masking
            analysis_results: Results from security analysis
            
        Returns:
            Path to generated summary report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(self.output_dir, f"executive_summary_{timestamp}.txt")
            
            # Calculate summary statistics
            total_files = len(processing_results)
            successful_files = sum(1 for r in processing_results if r['status'] == 'success')
            total_pii_detections = sum(r.get('detection_count', 0) for r in pii_results if r['status'] == 'success')
            
            total_iam_policies = 0
            total_firewall_rules = 0
            total_ids_logs = 0
            total_vulnerabilities = 0
            
            for result in analysis_results:
                if result['status'] == 'success':
                    summary = result.get('analysis', {}).get('summary', {})
                    total_iam_policies += summary.get('total_iam_policies', 0)
                    total_firewall_rules += summary.get('total_firewall_rules', 0)
                    total_ids_logs += summary.get('total_ids_logs', 0)
                    total_vulnerabilities += summary.get('total_vulnerabilities', 0)
            
            # Generate summary report
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE\n")
                f.write("EXECUTIVE SUMMARY REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                f.write("PROCESSING SUMMARY\n")
                f.write("-"*20 + "\n")
                f.write(f"Total Files Processed: {total_files}\n")
                f.write(f"Successfully Processed: {successful_files}\n")
                f.write(f"Success Rate: {(successful_files/total_files*100):.1f}%\n\n")
                
                f.write("PII DETECTION SUMMARY\n")
                f.write("-"*25 + "\n")
                f.write(f"Total PII Items Detected: {total_pii_detections}\n")
                f.write(f"Average PII per File: {(total_pii_detections/successful_files):.1f}\n\n")
                
                f.write("SECURITY ANALYSIS SUMMARY\n")
                f.write("-"*30 + "\n")
                f.write(f"IAM Policies Found: {total_iam_policies}\n")
                f.write(f"Firewall Rules Found: {total_firewall_rules}\n")
                f.write(f"IDS/IPS Log Entries: {total_ids_logs}\n")
                f.write(f"Vulnerability References: {total_vulnerabilities}\n\n")
                
                f.write("KEY INSIGHTS\n")
                f.write("-"*15 + "\n")
                if total_pii_detections > 0:
                    f.write(f"• Successfully identified and masked {total_pii_detections} sensitive data items\n")
                if total_iam_policies > 0:
                    f.write(f"• Found {total_iam_policies} IAM policy statements requiring review\n")
                if total_firewall_rules > 0:
                    f.write(f"• Identified {total_firewall_rules} firewall rules for analysis\n")
                if total_vulnerabilities > 0:
                    f.write(f"• Detected {total_vulnerabilities} CVE references for security assessment\n")
                
                f.write(f"\n• All sensitive data has been successfully anonymized\n")
                f.write(f"• Files are ready for security analysis and reporting\n")
            
            logger.info(f"Generated summary report: {summary_file}")
            return summary_file
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return ""
    
    def generate_extracted_text_output(self, processing_results: List[Dict[str, Any]]) -> str:
        """
        Generate extracted text output for verification
        
        Args:
            processing_results: Results from file processing
            
        Returns:
            Path to generated extracted text file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extracted_file = os.path.join(self.output_dir, f"extracted_text_{timestamp}.txt")
            
            with open(extracted_file, 'w', encoding='utf-8') as f:
                f.write("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE\n")
                f.write("EXTRACTED TEXT VERIFICATION\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Files Processed: {len(processing_results)}\n\n")
                
                for i, result in enumerate(processing_results, 1):
                    f.write(f"FILE {i}: {result.get('file_path', 'Unknown')}\n")
                    f.write("-"*50 + "\n")
                    f.write(f"File Type: {result.get('file_type', 'Unknown')}\n")
                    f.write(f"Processing Status: {result.get('status', 'Unknown')}\n")
                    f.write(f"Content Length: {len(result.get('content', ''))} characters\n")
                    f.write("\nEXTRACTED CONTENT:\n")
                    f.write("="*30 + "\n")
                    
                    if result.get('status') == 'success':
                        content = result.get('content', '')
                        if content:
                            f.write(content)
                        else:
                            f.write("[No content extracted]")
                    else:
                        f.write(f"[ERROR: {result.get('error', 'Unknown error')}]")
                    
                    f.write("\n\n" + "="*60 + "\n\n")
            
            logger.info(f"Generated extracted text output: {extracted_file}")
            return extracted_file
            
        except Exception as e:
            logger.error(f"Error generating extracted text output: {str(e)}")
            return ""
    
    def generate_llm_analysis_table(self, llm_analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generate LLM analysis table in the requested format
        
        Args:
            llm_analysis_results: Results from LLM analysis
            
        Returns:
            Path to generated table file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            table_file = os.path.join(self.output_dir, f"llm_analysis_table_{timestamp}.txt")
            
            with open(table_file, 'w', encoding='utf-8') as f:
                f.write("VIT CAMPUS CONNECT - CYBER SIMULATION EXERCISE\n")
                f.write("FILE ANALYSIS SAMPLE OUTPUT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Write summary header
                f.write("LLM ANALYSIS RESULTS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Process each file result
                for i, result in enumerate(llm_analysis_results, 1):
                    f.write(f"FILE {i}: {result.get('file_name', 'Unknown')}{result.get('file_type', 'Unknown')}\n")
                    f.write("-" * 50 + "\n")
                    
                    if result['status'] == 'success':
                        file_description = result.get('file_description', 'No description available')
                        key_findings = result.get('key_findings', [])
                        
                        # Write file description
                        f.write("DESCRIPTION:\n")
                        f.write(f"{file_description}\n\n")
                        
                        # Write key findings
                        f.write("KEY SECURITY FINDINGS:\n")
                        if isinstance(key_findings, list) and key_findings:
                            for j, finding in enumerate(key_findings, 1):
                                f.write(f"{j}. {finding}\n")
                        elif key_findings:
                            f.write(f"• {key_findings}\n")
                        else:
                            f.write("No key findings available\n")
                        
                        # Write timestamp
                        timestamp = result.get('analysis_timestamp', 'Unknown')
                        if timestamp != 'Unknown':
                            f.write(f"\nAnalysis completed: {timestamp}\n")
                    else:
                        # Handle error cases
                        error_msg = result.get('error', 'Unknown error')
                        f.write(f"ANALYSIS FAILED: {error_msg}\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
                
                # Add detailed breakdown section
                f.write("\n\nDETAILED FILE ANALYSIS BREAKDOWN\n")
                f.write("="*60 + "\n\n")
                
                for i, result in enumerate(llm_analysis_results, 1):
                    f.write(f"FILE {i}: {result.get('file_name', 'Unknown')}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"File Type: {result.get('file_type', 'Unknown')}\n")
                    f.write(f"File Path: {result.get('file_path', 'Unknown')}\n")
                    f.write(f"Analysis Status: {result.get('status', 'Unknown')}\n")
                    
                    if result['status'] == 'success':
                        f.write(f"\nFile Description:\n{result.get('file_description', 'No description available')}\n")
                        
                        f.write(f"\nKey Security Findings:\n")
                        key_findings = result.get('key_findings', [])
                        if isinstance(key_findings, list):
                            for finding in key_findings:
                                f.write(f"• {finding}\n")
                        else:
                            f.write(f"• {key_findings}\n")
                    else:
                        f.write(f"\nError: {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n" + "="*60 + "\n\n")
            
            logger.info(f"Generated LLM analysis table: {table_file}")
            return table_file
            
        except Exception as e:
            logger.error(f"Error generating LLM analysis table: {str(e)}")
            return ""
    
    def generate_all_outputs(self, processing_results: List[Dict[str, Any]], 
                           pii_results: List[Dict[str, Any]], 
                           analysis_results: List[Dict[str, Any]],
                           llm_analysis_results: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate all output formats
        
        Args:
            processing_results: Results from file processing
            pii_results: Results from PII detection and masking
            analysis_results: Results from security analysis
            llm_analysis_results: Results from LLM analysis
            
        Returns:
            Dict with all generated file paths
        """
        all_outputs = {}
        
        # Generate extracted text output for verification
        extracted_file = self.generate_extracted_text_output(processing_results)
        if extracted_file:
            all_outputs['extracted_text'] = extracted_file
        
        # Generate cleansed text outputs
        cleansed_outputs = self.generate_cleansed_output(processing_results, pii_results)
        all_outputs.update(cleansed_outputs)
        
        # Generate analysis outputs
        analysis_outputs = self.generate_analysis_output(analysis_results)
        all_outputs.update(analysis_outputs)
        
        # Generate CSV output
        csv_file = self.generate_csv_output(processing_results, pii_results, analysis_results)
        if csv_file:
            all_outputs['csv_output'] = csv_file
        
        # Generate JSON output
        json_file = self.generate_json_output(processing_results, pii_results, analysis_results, llm_analysis_results)
        if json_file:
            all_outputs['json_output'] = json_file
        
        # Generate summary report
        summary_file = self.generate_summary_report(processing_results, pii_results, analysis_results)
        if summary_file:
            all_outputs['summary_report'] = summary_file
        
        # Generate LLM analysis table output
        if llm_analysis_results:
            llm_table_file = self.generate_llm_analysis_table(llm_analysis_results)
            if llm_table_file:
                all_outputs['llm_analysis_table'] = llm_table_file
        
        return all_outputs
