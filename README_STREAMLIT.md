# VIT Campus Connect - Streamlit Frontend

A modern web interface for the VIT Campus Connect file processing system that allows users to upload files and get masked versions with security analysis results.

## Features

🛡️ **PII Detection & Masking**: Automatically detects and masks personally identifiable information
🔐 **Security Analysis**: Analyzes files for security-related content (IAM policies, firewall rules, etc.)
📁 **Multiple File Formats**: Supports TXT, PDF, DOCX, XLSX, PPTX, PNG, JPG, JPEG, CSV
📊 **Detailed Reports**: Comprehensive analysis with downloadable results
🎨 **Modern UI**: Clean, responsive interface built with Streamlit

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

**Option 1: Using the startup script (recommended)**
```bash
python run_streamlit_simple.py
```

**Option 2: Using the batch file (Windows)**
```bash
start_app.bat
```

**Option 3: Direct Streamlit command**
```bash
streamlit run streamlit_app.py --server.port 8501
```

**Option 4: If you encounter PyTorch errors**
```bash
python run_streamlit.py
```

### 3. Access the Web Interface
- Open your browser and go to: http://localhost:8501
- Upload files using the file uploader
- Click "Process Files" to analyze your documents
- Download masked files and analysis reports

## Usage Guide

### Uploading Files
1. Click "Choose files to process" in the main interface
2. Select one or more files from your computer
3. Supported formats: TXT, PDF, DOCX, XLSX, PPTX, PNG, JPG, JPEG, CSV

### Processing Files
1. After uploading, click the "🚀 Process Files" button
2. The system will:
   - Extract text content from your files
   - Detect and mask PII (emails, phone numbers, SSNs, etc.)
   - Analyze security-related content
   - Generate comprehensive reports

### Viewing Results
- **Processing Statistics**: See how many files were processed, PII items found, etc.
- **File Details**: Expand each file to see original content, masked content, and security findings
- **Download Options**: Download individual masked files or complete analysis reports

### Output Files
- **Masked Files**: Clean versions with PII replaced by placeholders
- **Analysis Reports**: JSON reports with detailed security findings
- **Processing Statistics**: Summary of the entire processing session

## File Processing Pipeline

```
Upload Files → Extract Content → Detect PII → Mask Sensitive Data → Security Analysis → Generate Reports
```

### PII Detection
The system automatically detects:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- Names and addresses
- And more...

### Security Analysis
Identifies and analyzes:
- IAM (Identity and Access Management) policies
- Firewall rules and configurations
- IDS (Intrusion Detection System) logs
- Security vulnerabilities
- Network configurations

## Configuration

### Supported File Formats
- **Text Files**: .txt, .csv
- **Documents**: .pdf, .docx, .xlsx, .pptx
- **Images**: .png, .jpg, .jpeg (OCR text extraction)

### Processing Options
- Batch processing of multiple files
- Automatic file format detection
- Error handling for unsupported formats
- Progress tracking and status updates

## Troubleshooting

### Common Issues

**PyTorch/Streamlit Conflicts**
- If you see PyTorch-related errors, use: `python run_streamlit_simple.py`
- The app will still work despite PyTorch warnings
- These warnings don't affect functionality

**Import Errors**
- Make sure you're running from the project root directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**File Upload Issues**
- Check file format is supported
- Ensure files are not corrupted
- Try smaller files if experiencing timeout issues

**Processing Errors**
- Some file formats may not extract text properly
- Very large files may take longer to process
- Check the error messages in the results section

**Port Already in Use**
- If port 8501 is busy, try: `streamlit run streamlit_app.py --server.port 8502`
- Or kill existing Streamlit processes and try again

### Performance Tips
- Process files in smaller batches for better performance
- Use supported file formats for optimal results
- Monitor system resources during large file processing

## Development

### Project Structure
```
├── streamlit_app.py          # Main Streamlit application
├── run_streamlit.py          # Startup script
├── src/                      # Core processing modules
│   ├── main_workflow.py      # Main workflow orchestrator
│   ├── file_processor.py     # File content extraction
│   ├── pii_detector.py       # PII detection and masking
│   ├── security_analyzer.py  # Security content analysis
│   └── output_generator.py   # Report generation
└── requirements.txt          # Python dependencies
```

### Adding New Features
1. Modify the core modules in `src/` for new processing capabilities
2. Update `streamlit_app.py` for new UI components
3. Test with various file formats and edge cases

## Security Considerations

- Uploaded files are processed in temporary directories
- Files are automatically cleaned up after processing
- No sensitive data is permanently stored
- All processing happens locally on your machine

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Try with sample files first to verify the system is working

---

**🛡️ VIT Campus Connect - Secure File Processing Made Simple**
