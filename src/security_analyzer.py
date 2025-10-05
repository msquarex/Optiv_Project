"""
Security Content Analysis Module for VIT Campus Connect
Analyzes and extracts security-related information from processed text
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAnalyzer:
    """Security content analysis class for extracting security insights"""
    
    def __init__(self):
        # IAM Policy patterns
        self.iam_patterns = {
            'policy_statement': r'"(?:Effect|Action|Resource|Principal)":\s*"[^"]*"',
            'permissions': r'(?:Allow|Deny|Read|Write|Execute|FullAccess)',
            'resources': r'arn:aws:[^:]+:[^:]*:[^:]*:[^:]*',
            'actions': r'(?:s3|ec2|iam|lambda|rds|dynamodb):[a-zA-Z0-9-]+',
            'conditions': r'"Condition":\s*\{[^}]+\}',
            'principals': r'"Principal":\s*\{[^}]+\}'
        }
        
        # Firewall rule patterns
        self.firewall_patterns = {
            'rule_number': r'^\s*\d+\s+',
            'action': r'(?:permit|deny|allow|block|drop|reject)',
            'protocol': r'(?:tcp|udp|icmp|ip|any)',
            'source_ip': r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?',
            'dest_ip': r'(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?',
            'port': r'(?:\d{1,5}|any|any)',
            'interface': r'(?:in|out|inside|outside)',
            'service': r'(?:http|https|ssh|ftp|telnet|smtp|dns|dhcp)'
        }
        
        # IDS/IPS log patterns
        self.ids_patterns = {
            'timestamp': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            'severity': r'(?:critical|high|medium|low|info)',
            'signature_id': r'(?:sid|signature):\s*\d+',
            'source_ip': r'(?:src|source):\s*(?:\d{1,3}\.){3}\d{1,3}',
            'dest_ip': r'(?:dst|dest|destination):\s*(?:\d{1,3}\.){3}\d{1,3}',
            'port': r'(?:sport|dport|port):\s*\d+',
            'protocol': r'(?:proto|protocol):\s*(?:tcp|udp|icmp)',
            'attack_type': r'(?:attack|threat|malware|exploit|intrusion)',
            'rule_name': r'(?:rule|signature):\s*[^\s]+'
        }
        
        # Network configuration patterns
        self.network_patterns = {
            'vlan': r'vlan\s+\d+',
            'subnet': r'(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}',
            'gateway': r'gateway\s+(?:\d{1,3}\.){3}\d{1,3}',
            'dns_server': r'dns\s+(?:\d{1,3}\.){3}\d{1,3}',
            'dhcp_pool': r'dhcp\s+pool\s+[^\s]+',
            'access_list': r'access-list\s+\d+',
            'route': r'ip\s+route\s+(?:\d{1,3}\.){3}\d{1,3}'
        }
        
        # Security vulnerability patterns
        self.vulnerability_patterns = {
            'cve': r'CVE-\d{4}-\d{4,7}',
            'cvss': r'CVSS:\s*\d+\.\d+',
            'severity': r'(?:critical|high|medium|low)',
            'vulnerability_type': r'(?:sql injection|xss|csrf|buffer overflow|privilege escalation)',
            'affected_software': r'(?:apache|nginx|mysql|postgresql|windows|linux|java|php|python)'
        }
    
    def analyze_security_content(self, text: str) -> Dict[str, Any]:
        """
        Main method to analyze security content in text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing security analysis results
        """
        try:
            analysis_results = {
                'iam_policies': self._analyze_iam_policies(text),
                'firewall_rules': self._analyze_firewall_rules(text),
                'ids_ips_logs': self._analyze_ids_ips_logs(text),
                'network_config': self._analyze_network_config(text),
                'vulnerabilities': self._analyze_vulnerabilities(text),
                'summary': {}
            }
            
            # Generate summary
            analysis_results['summary'] = self._generate_summary(analysis_results)
            
            return {
                'text': text,
                'analysis': analysis_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in security analysis: {str(e)}")
            return {
                'text': text,
                'analysis': {},
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_iam_policies(self, text: str) -> Dict[str, Any]:
        """Analyze IAM policies in text"""
        results = {
            'policies_found': [],
            'permissions': [],
            'resources': [],
            'actions': [],
            'conditions': [],
            'principals': []
        }
        
        # Find policy statements
        policy_matches = re.finditer(self.iam_patterns['policy_statement'], text, re.IGNORECASE)
        for match in policy_matches:
            results['policies_found'].append({
                'content': match.group(),
                'position': match.span()
            })
        
        # Find permissions
        permission_matches = re.finditer(self.iam_patterns['permissions'], text, re.IGNORECASE)
        for match in permission_matches:
            results['permissions'].append({
                'permission': match.group(),
                'position': match.span()
            })
        
        # Find AWS resources
        resource_matches = re.finditer(self.iam_patterns['resources'], text, re.IGNORECASE)
        for match in resource_matches:
            results['resources'].append({
                'arn': match.group(),
                'position': match.span()
            })
        
        # Find actions
        action_matches = re.finditer(self.iam_patterns['actions'], text, re.IGNORECASE)
        for match in action_matches:
            results['actions'].append({
                'action': match.group(),
                'position': match.span()
            })
        
        return results
    
    def _analyze_firewall_rules(self, text: str) -> Dict[str, Any]:
        """Analyze firewall rules in text"""
        results = {
            'rules_found': [],
            'actions': [],
            'protocols': [],
            'source_ips': [],
            'dest_ips': [],
            'ports': [],
            'services': []
        }
        
        # Find firewall rules (lines that start with numbers)
        rule_matches = re.finditer(self.firewall_patterns['rule_number'], text, re.MULTILINE)
        for match in rule_matches:
            # Extract the full rule line
            line_start = match.start()
            line_end = text.find('\n', line_start)
            if line_end == -1:
                line_end = len(text)
            
            rule_line = text[line_start:line_end].strip()
            results['rules_found'].append({
                'rule': rule_line,
                'position': (line_start, line_end)
            })
        
        # Find actions
        action_matches = re.finditer(self.firewall_patterns['action'], text, re.IGNORECASE)
        for match in action_matches:
            results['actions'].append({
                'action': match.group(),
                'position': match.span()
            })
        
        # Find protocols
        protocol_matches = re.finditer(self.firewall_patterns['protocol'], text, re.IGNORECASE)
        for match in protocol_matches:
            results['protocols'].append({
                'protocol': match.group(),
                'position': match.span()
            })
        
        # Find IP addresses
        ip_matches = re.finditer(self.firewall_patterns['source_ip'], text)
        for match in ip_matches:
            results['source_ips'].append({
                'ip': match.group(),
                'position': match.span()
            })
        
        ip_matches = re.finditer(self.firewall_patterns['dest_ip'], text)
        for match in ip_matches:
            results['dest_ips'].append({
                'ip': match.group(),
                'position': match.span()
            })
        
        # Find ports
        port_matches = re.finditer(self.firewall_patterns['port'], text, re.IGNORECASE)
        for match in port_matches:
            results['ports'].append({
                'port': match.group(),
                'position': match.span()
            })
        
        return results
    
    def _analyze_ids_ips_logs(self, text: str) -> Dict[str, Any]:
        """Analyze IDS/IPS logs in text"""
        results = {
            'log_entries': [],
            'timestamps': [],
            'severities': [],
            'signatures': [],
            'source_ips': [],
            'dest_ips': [],
            'attack_types': [],
            'rule_names': []
        }
        
        # Find log entries (lines with timestamps)
        timestamp_matches = re.finditer(self.ids_patterns['timestamp'], text)
        for match in timestamp_matches:
            # Extract the full log line
            line_start = text.rfind('\n', 0, match.start()) + 1
            line_end = text.find('\n', match.end())
            if line_end == -1:
                line_end = len(text)
            
            log_line = text[line_start:line_end].strip()
            results['log_entries'].append({
                'entry': log_line,
                'timestamp': match.group(),
                'position': (line_start, line_end)
            })
        
        # Find severities
        severity_matches = re.finditer(self.ids_patterns['severity'], text, re.IGNORECASE)
        for match in severity_matches:
            results['severities'].append({
                'severity': match.group(),
                'position': match.span()
            })
        
        # Find signature IDs
        sig_matches = re.finditer(self.ids_patterns['signature_id'], text, re.IGNORECASE)
        for match in sig_matches:
            results['signatures'].append({
                'signature': match.group(),
                'position': match.span()
            })
        
        # Find attack types
        attack_matches = re.finditer(self.ids_patterns['attack_type'], text, re.IGNORECASE)
        for match in attack_matches:
            results['attack_types'].append({
                'attack': match.group(),
                'position': match.span()
            })
        
        return results
    
    def _analyze_network_config(self, text: str) -> Dict[str, Any]:
        """Analyze network configuration in text"""
        results = {
            'vlans': [],
            'subnets': [],
            'gateways': [],
            'dns_servers': [],
            'dhcp_pools': [],
            'access_lists': [],
            'routes': []
        }
        
        # Find VLANs
        vlan_matches = re.finditer(self.network_patterns['vlan'], text, re.IGNORECASE)
        for match in vlan_matches:
            results['vlans'].append({
                'vlan': match.group(),
                'position': match.span()
            })
        
        # Find subnets
        subnet_matches = re.finditer(self.network_patterns['subnet'], text)
        for match in subnet_matches:
            results['subnets'].append({
                'subnet': match.group(),
                'position': match.span()
            })
        
        # Find gateways
        gateway_matches = re.finditer(self.network_patterns['gateway'], text, re.IGNORECASE)
        for match in gateway_matches:
            results['gateways'].append({
                'gateway': match.group(),
                'position': match.span()
            })
        
        # Find DNS servers
        dns_matches = re.finditer(self.network_patterns['dns_server'], text, re.IGNORECASE)
        for match in dns_matches:
            results['dns_servers'].append({
                'dns': match.group(),
                'position': match.span()
            })
        
        return results
    
    def _analyze_vulnerabilities(self, text: str) -> Dict[str, Any]:
        """Analyze security vulnerabilities in text"""
        results = {
            'cves': [],
            'cvss_scores': [],
            'severities': [],
            'vulnerability_types': [],
            'affected_software': []
        }
        
        # Find CVEs
        cve_matches = re.finditer(self.vulnerability_patterns['cve'], text, re.IGNORECASE)
        for match in cve_matches:
            results['cves'].append({
                'cve': match.group(),
                'position': match.span()
            })
        
        # Find CVSS scores
        cvss_matches = re.finditer(self.vulnerability_patterns['cvss'], text, re.IGNORECASE)
        for match in cvss_matches:
            results['cvss_scores'].append({
                'cvss': match.group(),
                'position': match.span()
            })
        
        # Find vulnerability types
        vuln_matches = re.finditer(self.vulnerability_patterns['vulnerability_type'], text, re.IGNORECASE)
        for match in vuln_matches:
            results['vulnerability_types'].append({
                'type': match.group(),
                'position': match.span()
            })
        
        # Find affected software
        software_matches = re.finditer(self.vulnerability_patterns['affected_software'], text, re.IGNORECASE)
        for match in software_matches:
            results['affected_software'].append({
                'software': match.group(),
                'position': match.span()
            })
        
        return results
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for analysis results"""
        summary = {
            'total_iam_policies': len(analysis_results['iam_policies']['policies_found']),
            'total_firewall_rules': len(analysis_results['firewall_rules']['rules_found']),
            'total_ids_logs': len(analysis_results['ids_ips_logs']['log_entries']),
            'total_vulnerabilities': len(analysis_results['vulnerabilities']['cves']),
            'total_network_configs': (
                len(analysis_results['network_config']['vlans']) +
                len(analysis_results['network_config']['subnets']) +
                len(analysis_results['network_config']['gateways'])
            ),
            'security_insights': []
        }
        
        # Generate security insights
        if summary['total_iam_policies'] > 0:
            summary['security_insights'].append(f"Found {summary['total_iam_policies']} IAM policy statements")
        
        if summary['total_firewall_rules'] > 0:
            summary['security_insights'].append(f"Identified {summary['total_firewall_rules']} firewall rules")
        
        if summary['total_ids_logs'] > 0:
            summary['security_insights'].append(f"Detected {summary['total_ids_logs']} IDS/IPS log entries")
        
        if summary['total_vulnerabilities'] > 0:
            summary['security_insights'].append(f"Found {summary['total_vulnerabilities']} CVE references")
        
        return summary
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_security_content(text)
            results.append(result)
        
        return results
