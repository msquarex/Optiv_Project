# VIT Campus Connect - Cyber Simulation Exercise

## Automating File Cleansing and Analysis leveraging AI

### Project Overview

This project implements an automated solution for security consultants to cleanse and analyze heterogeneous file formats containing sensitive client data. The system removes PII, client-specific information, and extracts security-related insights while ensuring data privacy and compliance.

### Key Features

- **Multi-format File Processing**: Supports .xlsx, .txt, .pdf, .jpeg, .png, .pptx files
- **PII Detection & Masking**: Automated detection and anonymization of sensitive information
- **Security Content Analysis**: Extracts IAM policies, firewall rules, IDS/IPS logs
- **Advanced OCR Capabilities**: Text extraction from scanned documents and images
- **AI-Powered Image Description**: Uses Gemma3 model for intelligent image analysis
- **Standardized Output**: Generates clean, structured results in multiple formats
- **Comprehensive Reporting**: Detailed analysis with visualizations and summaries

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  File Processing │───▶│  PII Detection  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Output Generation│◀───│ Security Analysis│◀───│   Content Mask  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Optiv
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model** (for advanced NER):
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Install Tesseract OCR** (for image processing):
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

5. **Setup Gemma3 for AI Image Description** (optional but recommended):
   ```bash
   # Install Ollama
   # Windows: Download from https://ollama.ai/download
   # macOS: brew install ollama
   # Linux: curl -fsSL https://ollama.ai/install.sh | sh
   
   # Download Gemma3 model
   ollama pull gemma3:4b
   
   # Start Ollama service (default port 11434)
   ollama serve
   
   # Optional: Configure custom port (e.g., 11500)
   OLLAMA_HOST=127.0.0.1:11500 ollama serve
   ```

### Quick Start
#### Frontend
```python
python run_streamlit.py



#### Option 1: Jupyter Notebook Demo
```bash
jupyter notebook VIT_Campus_Connect_Demo.ipynb
```

#### Option 2: Python Script
```python
from src.main_workflow import CyberSimulationWorkflow

# Initialize the workflow
workflow = CyberSimulationWorkflow(output_dir="output")

# Process files
file_paths = ["path/to/your/files/file1.txt", "path/to/your/files/file2.xlsx"]
results = workflow.process_files(file_paths)

# View results
print(f"Processing Status: {results['status']}")
print(f"Files Processed: {results['processing_stats']['total_files']}")
print(f"PII Detections: {results['processing_stats']['total_pii_detections']}")
```

#### Option 3: Run Demo with Sample Files
```python
from src.main_workflow import CyberSimulationWorkflow

workflow = CyberSimulationWorkflow()
results = workflow.run_demo()
```

#### Option 4: Image Processing Demo
```python
# Run the image processing demo
python demo_image_processing.py
```

### Supported File Formats

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| Excel | .xlsx, .xls | pandas.read_excel() |
| Text | .txt | Direct text reading |
| PDF | .pdf | PyPDF2 + OCR fallback |
| Images | .jpeg, .jpg, .png | OpenCV + Tesseract OCR + Gemma3 AI Description |
| PowerPoint | .pptx | python-pptx |
| Word | .docx | python-docx |

### PII Detection Capabilities

The system detects and masks the following types of sensitive information:

- **Personal Information**: Names, SSNs, phone numbers, email addresses
- **Financial Data**: Credit card numbers, bank account information
- **Network Information**: IP addresses, MAC addresses, URLs
- **Geographic Data**: ZIP codes, addresses
- **Client Information**: Company names, organization identifiers
- **Dates**: Birth dates, sensitive timestamps

### AI-Powered Image Description

The system now includes advanced AI capabilities for image analysis using the Gemma3 model:

#### Features
- **Intelligent Image Understanding**: Analyzes image content beyond just text extraction
- **Contextual Descriptions**: Provides detailed descriptions of visual elements
- **Security-Relevant Analysis**: Identifies security-related content in images
- **Combined Processing**: Works alongside OCR for comprehensive image analysis

#### How It Works
1. **OCR Processing**: Extracts text from images using Tesseract
2. **AI Description**: Sends image to Gemma3 for intelligent analysis
3. **Combined Output**: Merges OCR text and AI description for complete analysis

#### Configuration
```python
# Initialize with custom Gemma3 URL
workflow = CyberSimulationWorkflow(gemma_url="http://127.0.0.1:11500")

# Test connection
test_result = workflow.file_processor.image_descriptor.test_connection()
```

#### Example Output
```
=== OCR EXTRACTED TEXT ===
VISITORS LOG BOOK - PLEASE SIGN IN
DATE: 22/03/23
Name: Kate Hopkins
Reason: Sales Demo

=== IMAGE DESCRIPTION (Gemma3) ===
This image shows a visitors log book with a sign-in sheet. The document appears to be a standard office visitor registration form with fields for date, name, and reason for visit. The form is printed on white paper with black text and has a professional appearance typical of corporate visitor management systems.
```

### Security Analysis Features

#### IAM Policy Analysis
- Extracts AWS IAM policy statements
- Identifies permissions, resources, and conditions
- Analyzes principal configurations

#### Firewall Rule Analysis
- Parses firewall rule configurations
- Extracts source/destination IPs, ports, protocols
- Identifies access control actions

#### IDS/IPS Log Analysis
- Processes security log entries
- Extracts timestamps, severity levels, signature IDs
- Identifies attack types and threat indicators

#### Vulnerability Assessment
- Detects CVE references
- Extracts CVSS scores and severity levels
- Identifies affected software components

### Output Formats

The system generates multiple output formats:

1. **Cleansed Text Files**: Individual and consolidated anonymized content
2. **Security Analysis Report**: Detailed security findings and insights
3. **CSV Export**: Structured data for further analysis
4. **JSON Export**: Complete results with metadata
5. **Executive Summary**: High-level overview and statistics

### Configuration

#### Environment Variables
```bash
# Optional: Set custom output directory
export OUTPUT_DIR="/path/to/output"

# Optional: Set Tesseract path (if not in system PATH)
export TESSERACT_CMD="/usr/local/bin/tesseract"
```

#### Custom PII Patterns
You can extend PII detection by modifying `src/pii_detector.py`:

```python
# Add custom patterns
self.pii_patterns['custom_pattern'] = r'your_regex_pattern_here'
```

### Performance Metrics

- **Processing Speed**: ~30 seconds for 20MB files
- **Accuracy**: ≥90% PII detection rate
- **Scalability**: Handles up to 100 files per batch
- **Memory Usage**: Optimized for large file processing

### Error Handling

The system includes comprehensive error handling:

- **File Format Validation**: Checks supported formats before processing
- **Graceful Degradation**: Continues processing even if individual files fail
- **Detailed Logging**: Provides clear error messages and debugging information
- **Recovery Mechanisms**: Handles corrupted or malformed files

### Security Considerations

- **Temporary File Encryption**: All temporary files are encrypted during processing
- **Memory Management**: Sensitive data is cleared from memory after processing
- **Access Control**: Output files are generated with appropriate permissions
- **Audit Logging**: Complete processing logs for compliance

### API Reference

#### CyberSimulationWorkflow

```python
class CyberSimulationWorkflow:
    def __init__(self, output_dir: str = "output")
    def process_files(self, file_paths: List[str]) -> Dict[str, Any]
    def process_single_file(self, file_path: str) -> Dict[str, Any]
    def run_demo(self, sample_dir: str = "sample_files") -> Dict[str, Any]
    def get_processing_summary(self) -> Dict[str, Any]
    def validate_file_paths(self, file_paths: List[str]) -> Dict[str, List[str]]
```

#### FileProcessor

```python
class FileProcessor:
    def process_file(self, file_path: str) -> Dict[str, Any]
    def batch_process(self, file_paths: List[str]) -> List[Dict[str, Any]]
```

#### PIIDetector

```python
class PIIDetector:
    def detect_and_mask_pii(self, text: str) -> Dict[str, Any]
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]
    def get_detection_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]
```

#### SecurityAnalyzer

```python
class SecurityAnalyzer:
    def analyze_security_content(self, text: str) -> Dict[str, Any]
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]
```

### Troubleshooting

#### Common Issues

1. **Tesseract not found**:
   ```bash
   # Install Tesseract and add to PATH
   # Or set TESSERACT_CMD environment variable
   ```

2. **spaCy model missing**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory issues with large files**:
   - Process files in smaller batches
   - Increase system memory allocation
   - Use file streaming for very large files

4. **OCR accuracy issues**:
   - Ensure images are high quality
   - Preprocess images for better contrast
   - Use appropriate Tesseract configuration

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### License

This project is developed for the VIT Campus Connect - Cyber Simulation Exercise.

### Support

For questions or issues:
- Check the troubleshooting section
- Review the Jupyter notebook demo
- Examine the source code documentation

---

**VIT Campus Connect - Cyber Simulation Exercise**  
*Automating File Cleansing and Analysis leveraging AI*

**Developed by**: Optiv Product Management  
**Date**: September 5, 2025
