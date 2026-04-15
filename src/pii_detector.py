"""
PII Detection and Masking Module for VIT Campus Connect
Detects and masks Personally Identifiable Information (PII) and sensitive data
Uses Microsoft Presidio for advanced PII detection and masking
"""

import re
import logging
import uuid
from typing import Dict, List, Tuple, Any
import hashlib
import random
import string

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from crypto_utils import CryptoUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIDetector:
    """PII detection and masking class"""
    
    def __init__(self):
        # Admin password for ChaCha20 encryption
        self.admin_password = "admin123"
        
        # Initialize Presidio engines
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.use_presidio = True
            logger.info("Presidio engines initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {str(e)}")
            self.analyzer = None
            self.anonymizer = None
            self.use_presidio = False
        
        # Keep existing regex patterns for additional custom detection
        self.pii_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'date_of_birth': r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b'
        }
        
        # Common company/client names to mask
        self.client_indicators = [
            r'\b(?:client|company|organization|corp|inc|ltd|llc)\b',
            r'\b(?:confidential|proprietary|internal)\b'
        ]
        
        # Common names database (basic fallback)
        self.common_names = {
            'john', 'jane', 'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia',
            'miller', 'davis', 'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez',
            'wilson', 'anderson', 'thomas', 'taylor', 'moore', 'jackson', 'martin',
            'lee', 'perez', 'thompson', 'white', 'harris', 'sanchez', 'clark'
        }
    
    def detect_and_mask_pii(self, text: str) -> Dict[str, Any]:
        """
        Main method to detect and mask PII in text using Presidio
        
        Args:
            text (str): Input text to process
            
        Returns:
            Dict containing masked text and detection results
        """
        try:
            original_text = text
            masked_text = text
            detected_items = []
            
            # Use Presidio for primary PII detection
            if self.use_presidio:
                masked_text, presidio_detections = self._detect_and_mask_with_presidio(text)
                detected_items.extend(presidio_detections)
            else:
                logger.warning("Presidio not available, falling back to regex-only detection")
            
            # Additional custom regex patterns for specific PII types
            masked_text, regex_detections = self._detect_custom_patterns(masked_text)
            detected_items.extend(regex_detections)
            
            # Detect and mask client-specific information
            masked_text, client_detections = self._mask_client_info(masked_text)
            detected_items.extend(client_detections)
            
            return {
                'original_text': original_text,
                'masked_text': masked_text,
                'detected_items': detected_items,
                'detection_count': len(detected_items),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in PII detection: {str(e)}")
            return {
                'original_text': text,
                'masked_text': text,
                'detected_items': [],
                'detection_count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _detect_and_mask_with_presidio(self, text: str) -> Tuple[str, List[Dict]]:
        """Detect and mask PII using Presidio with unique UUID placeholders and ChaCha20 encryption"""
        detected_items = []
        try:
            results = self.analyzer.analyze(text=text, language='en')
            # Filter out URL detections
            results = [r for r in results if r.entity_type != 'URL']
            # Sort descending to safely replace substrings in text modifying the tail first.
            results.sort(key=lambda x: x.start, reverse=True)
            
            masked_text = text
            for result in results:
                original_value = text[result.start:result.end]
                token_id = str(uuid.uuid4())
                placeholder = f"<PII_TOKEN_{token_id}>"
                
                nonce, ciphertext = CryptoUtils.encrypt_data(self.admin_password, original_value)
                
                # Replace the exact substring
                masked_text = masked_text[:result.start] + placeholder + masked_text[result.end:]
                
                detected_items.append({
                    'type': result.entity_type.lower(),
                    'subtype': result.entity_type,
                    'original': original_value,
                    'masked': placeholder,
                    'placeholder': placeholder,
                    'encrypted_nonce': nonce,
                    'encrypted_ciphertext': ciphertext,
                    'position': (result.start, result.start + len(placeholder)),
                    'confidence': result.score
                })
            
            # Return reversed list to keep timeline logic somewhat standard if needed
            detected_items.reverse()
            return masked_text, detected_items
            
        except Exception as e:
            logger.error(f"Error in Presidio detection: {str(e)}")
            return text, []
    
    def _detect_custom_patterns(self, text: str) -> Tuple[str, List[Dict]]:
        """Detect additional PII using custom regex patterns"""
        detected_items = []
        masked_text = text
        
        # Detect patterns not covered by Presidio or to ensure better coverage
        additional_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'date_of_birth': r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b'
        }
        
        for pii_type, pattern in additional_patterns.items():
            matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
            matches.sort(key=lambda x: x.start(), reverse=True)
            
            for match in matches:
                original_value = match.group()
                token_id = str(uuid.uuid4())
                placeholder = f"<PII_TOKEN_{token_id}>"
                nonce, ciphertext = CryptoUtils.encrypt_data(self.admin_password, original_value)
                
                start, end = match.span()
                masked_text = masked_text[:start] + placeholder + masked_text[end:]
                
                detected_items.append({
                    'type': pii_type,
                    'subtype': 'custom_regex',
                    'original': original_value,
                    'masked': placeholder,
                    'placeholder': placeholder,
                    'encrypted_nonce': nonce,
                    'encrypted_ciphertext': ciphertext,
                    'position': (start, start + len(placeholder))
                })
        
        detected_items.reverse()
        return masked_text, detected_items
    
    def _mask_client_info(self, text: str) -> Tuple[str, List[Dict]]:
        """Mask client-specific information using regex patterns"""
        detected_items = []
        masked_text = text
        
        for pattern in self.client_indicators:
            matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
            matches.sort(key=lambda x: x.start(), reverse=True)
            
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(masked_text), match.end() + 50)
                context = masked_text[start:end]
                
                company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
                company_matches = list(re.finditer(company_pattern, context))
                company_matches.sort(key=lambda x: x.start(), reverse=True)
                
                for company_match in company_matches:
                    company_name = company_match.group()
                    if len(company_name.split()) >= 2:
                        full_start = start + company_match.start()
                        full_end = start + company_match.end()
                        
                        token_id = str(uuid.uuid4())
                        placeholder = f"<PII_TOKEN_{token_id}>"
                        nonce, ciphertext = CryptoUtils.encrypt_data(self.admin_password, company_name)
                        
                        masked_text = masked_text[:full_start] + placeholder + masked_text[full_end:]
                        detected_items.append({
                            'type': 'client_info',
                            'subtype': 'organization',
                            'original': company_name,
                            'masked': placeholder,
                            'placeholder': placeholder,
                            'encrypted_nonce': nonce,
                            'encrypted_ciphertext': ciphertext,
                            'position': (full_start, full_start + len(placeholder))
                        })
                        break  
        
        detected_items.reverse()
        return masked_text, detected_items
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts (List[str]): List of texts to process
            
        Returns:
            List of processing results
        """
        results = []
        for text in texts:
            result = self.detect_and_mask_pii(text)
            results.append(result)
        
        return results
    
    def get_detection_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing results"""
        total_detections = sum(result['detection_count'] for result in results)
        pii_types = {}
        
        for result in results:
            for item in result['detected_items']:
                pii_type = item['type']
                pii_types[pii_type] = pii_types.get(pii_type, 0) + 1
        
        return {
            'total_files_processed': len(results),
            'total_detections': total_detections,
            'detection_types': pii_types,
            'average_detections_per_file': total_detections / len(results) if results else 0
        }
