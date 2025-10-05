"""
LLM Analysis Module for VIT Campus Connect
Uses Gemma3 model to analyze extracted text and image descriptions
to generate file descriptions and security findings
"""

import requests
import logging
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Class for analyzing file content using Gemma3 model to generate descriptions and security findings"""
    
    def __init__(self, gemma_url: str = "http://127.0.0.1:11500"):
        """
        Initialize the LLM Analyzer
        
        Args:
            gemma_url (str): URL of the Gemma3 Ollama service
        """
        self.gemma_url = gemma_url
        self.model_name = "gemma3:4b"  # Adjust based on your actual model name
        
    def analyze_file_content(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single file's content to generate description and security findings
        
        Args:
            file_data (Dict): File data containing extracted text and metadata
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Extract file information
            file_path = file_data.get('file_path', 'Unknown')
            file_type = file_data.get('file_type', 'Unknown')
            content = file_data.get('content', '')
            
            # Get file name without path
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            file_name = file_name.split('.')[0]  # Remove extension
            
            # Prepare the analysis prompt
            prompt = self._create_analysis_prompt(file_name, file_type, content)
            
            # Send request to Gemma3
            response = self._send_llm_request(prompt)
            
            if response['status'] == 'success':
                # Parse the LLM response
                analysis_result = self._parse_llm_response(response['content'], file_name, file_type)
                
                return {
                    'status': 'success',
                    'file_name': file_name,
                    'file_type': file_type,
                    'file_path': file_path,
                    'file_description': analysis_result['description'],
                    'key_findings': analysis_result['findings'],
                    'raw_llm_response': response['content'],
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'file_name': file_name,
                    'file_type': file_type,
                    'file_path': file_path,
                    'file_description': 'Analysis failed',
                    'key_findings': ['Error: Could not analyze file content'],
                    'error': response.get('error', 'Unknown error'),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing file content: {str(e)}")
            return {
                'status': 'error',
                'file_name': file_data.get('file_path', 'Unknown').split('/')[-1].split('\\')[-1].split('.')[0],
                'file_type': file_data.get('file_type', 'Unknown'),
                'file_path': file_data.get('file_path', 'Unknown'),
                'file_description': 'Analysis failed',
                'key_findings': ['Error: Analysis processing failed'],
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def batch_analyze_files(self, file_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple files in batch
        
        Args:
            file_data_list (List[Dict]): List of file data dictionaries
            
        Returns:
            List of analysis results
        """
        results = []
        for file_data in file_data_list:
            result = self.analyze_file_content(file_data)
            results.append(result)
        return results
    
    def _create_analysis_prompt(self, file_name: str, file_type: str, content: str) -> str:
        """
        Create a structured prompt for LLM analysis
        
        Args:
            file_name (str): Name of the file
            file_type (str): Type/extension of the file
            content (str): Extracted content from the file
            
        Returns:
            str: Formatted prompt for the LLM
        """
        prompt = f"""You are a cybersecurity analyst reviewing files for a security assessment. Analyze the following file content and provide:

1. A concise file description (2-3 sentences)
2. Key security findings (4-6 bullet points covering both security issues and existing security features)

IMPORTANT: The content below has been MASKED for privacy protection. Sensitive information like names, emails, phone numbers, and other PII have been replaced with [MASKED] or similar placeholders. Do NOT make assumptions about missing information - what appears as [MASKED] may actually contain detailed information in the original document.

File Information:
- File Name: {file_name}
- File Type: {file_type}

File Content (MASKED for privacy):
{content[:3000]}  # Limit content to avoid token limits

Please respond in the following JSON format:
{{
    "file_description": "Brief description of what this file contains and its purpose",
    "key_findings": [
        "Security finding 1 - focus on access control, authentication, or security implications",
        "Security finding 2 - focus on data protection, vulnerabilities, or compliance",
        "Security finding 3 - focus on operational security or risk factors",
        "Existing security feature 1 - identify current security measures or controls in place",
        "Existing security feature 2 - highlight positive security implementations or protocols",
        "Additional finding - any other security-relevant observation"
    ]
}}

CRITICAL INSTRUCTIONS FOR MASKED CONTENT:
- Do NOT assume information is missing just because you see [MASKED] placeholders
- Focus on the STRUCTURE, PATTERNS, and VISIBLE CONTENT rather than what appears to be missing
- If you see [MASKED] in what appears to be a user field, assume the original contains actual user information
- If you see [MASKED] in what appears to be a timestamp or ID field, assume the original contains actual data
- Base your analysis on what IS visible, not what appears to be missing due to masking

Focus on both security issues AND existing security features:

Security Issues to identify (based on visible structure and patterns):
- Access control vulnerabilities or weaknesses visible in the structure
- Authentication mechanism flaws evident from the content layout
- Data protection gaps observable in the document structure
- Compliance and policy violations apparent from visible content
- Operational security risks indicated by the document format
- Physical security deficiencies shown in the structure

Existing Security Features to identify (based on visible elements):
- Access control systems (card readers, biometric systems, keypads) mentioned or referenced
- Authentication mechanisms already in place (visible in structure or labels)
- Data protection measures implemented (evident from document structure)
- Security protocols and procedures (visible in content organization)
- Physical security measures (mentioned or referenced)
- Monitoring and surveillance systems (indicated by structure or labels)
- Security policies and compliance measures (visible in document format)

If the content appears to be a visitors log, access control system, or security-related document, provide a balanced analysis based on the visible structure and patterns, acknowledging that detailed information may be masked."""
        
        return prompt
    
    def _send_llm_request(self, prompt: str) -> Dict[str, Any]:
        """
        Send request to Gemma3 model via Ollama API
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            Dict containing response or error information
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.gemma_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '')
                if not content:
                    content = result.get('response', '')
                
                return {
                    'status': 'success',
                    'content': content.strip(),
                    'metadata': {
                        'total_duration': result.get('total_duration', 0),
                        'eval_count': result.get('eval_count', 0)
                    }
                }
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'error': error_msg
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Request to Gemma3 service timed out"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to Gemma3 service at {self.gemma_url}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error in LLM request: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
    
    def _parse_llm_response(self, response_content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured information
        
        Args:
            response_content (str): Raw response from the LLM
            file_name (str): Name of the file being analyzed
            file_type (str): Type of the file being analyzed
            
        Returns:
            Dict containing parsed description and findings
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    return {
                        'description': json_data.get('file_description', 'No description provided'),
                        'findings': json_data.get('key_findings', ['No findings provided'])
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Parse text response manually
            return self._parse_text_response(response_content, file_name, file_type)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                'description': 'Error parsing analysis response',
                'findings': ['Error: Could not parse analysis results']
            }
    
    def _parse_text_response(self, content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """
        Parse text response when JSON parsing fails
        
        Args:
            content (str): Text response from LLM
            file_name (str): Name of the file
            file_type (str): Type of the file
            
        Returns:
            Dict containing parsed description and findings
        """
        try:
            lines = content.split('\n')
            description = ""
            findings = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for description section
                if any(keyword in line.lower() for keyword in ['description', 'file contains', 'this file']):
                    current_section = 'description'
                    description = line
                    continue
                
                # Look for findings section
                if any(keyword in line.lower() for keyword in ['findings', 'security', 'key points', 'implications']):
                    current_section = 'findings'
                    continue
                
                # Add content based on current section
                if current_section == 'description' and not any(keyword in line.lower() for keyword in ['findings', 'security', 'key']):
                    if description:
                        description += " " + line
                    else:
                        description = line
                elif current_section == 'findings' or line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    # Clean up bullet points
                    clean_line = re.sub(r'^[-•*]\s*', '', line)
                    if clean_line and len(clean_line) > 10:  # Only add substantial findings
                        findings.append(clean_line)
            
            # If no structured content found, create basic analysis
            if not description:
                if file_type in ['.png', '.jpg', '.jpeg']:
                    description = f"Image file {file_name} containing visual content that may relate to security systems or access control."
                else:
                    description = f"File {file_name} of type {file_type} containing text or data content."
            
            if not findings:
                findings = ["No specific security findings identified in the content analysis."]
            
            return {
                'description': description,
                'findings': findings[:5]  # Limit to 5 findings
            }
            
        except Exception as e:
            logger.error(f"Error in text response parsing: {str(e)}")
            return {
                'description': f"Analysis of {file_name} file",
                'findings': ['Error: Could not parse analysis response']
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Gemma3 service
        
        Returns:
            Dict containing connection test results
        """
        try:
            # Try to get model list to test connection
            response = requests.get(f"{self.gemma_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                return {
                    'status': 'success',
                    'connection': True,
                    'available_models': model_names,
                    'target_model': self.model_name,
                    'model_available': self.model_name in model_names
                }
            else:
                return {
                    'status': 'error',
                    'connection': False,
                    'error': f"API returned status {response.status_code}",
                    'available_models': [],
                    'target_model': self.model_name,
                    'model_available': False
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'connection': False,
                'error': f"Could not connect to {self.gemma_url}",
                'available_models': [],
                'target_model': self.model_name,
                'model_available': False
            }
        except Exception as e:
            return {
                'status': 'error',
                'connection': False,
                'error': str(e),
                'available_models': [],
                'target_model': self.model_name,
                'model_available': False
            }
