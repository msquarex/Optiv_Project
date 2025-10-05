#!/usr/bin/env python3
"""
Streamlit Frontend for VIT Campus Connect - File Processing System
Upload files and get masked versions with security analysis results
"""

import warnings
import os
import sys

# Suppress PyTorch warnings that can interfere with Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
os.environ["TORCH_LOGS"] = "-dynamo"

import streamlit as st
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import json

# Add src directory to path
sys.path.append('src')

# Import our modules
from main_workflow import CyberSimulationWorkflow

# Page configuration
st.set_page_config(
    page_title="VIT Campus Connect - File Processor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .download-button {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

def initialize_workflow() -> CyberSimulationWorkflow:
    """Initialize the workflow system"""
    if 'workflow' not in st.session_state:
        st.session_state.workflow = CyberSimulationWorkflow(
            output_dir="streamlit_output",
            gemma_url="http://127.0.0.1:11500"
        )
    return st.session_state.workflow

def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """Save uploaded files to temporary directory and return file paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Create file path - preserve directory structure for folder uploads
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Ensure directory exists for nested files
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file content
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_paths.append(file_path)
    
    return file_paths, temp_dir

def display_processing_results(results: Dict[str, Any]):
    """Display processing results in a user-friendly format"""
    
    if results['status'] == 'error':
        st.error(f"❌ Processing Error: {results.get('error', 'Unknown error')}")
        return
    
    # Processing Statistics
    stats = results['processing_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Files Processed",
            value=stats['successful_files'],
            delta=f"{stats['failed_files']} failed" if stats['failed_files'] > 0 else None
        )
    
    with col2:
        st.metric(
            label="PII Items Found",
            value=stats['total_pii_detections']
        )
    
    with col3:
        st.metric(
            label="Security Findings",
            value=stats['total_security_findings']
        )
    
    with col4:
        duration = results.get('processing_duration_seconds', 0)
        st.metric(
            label="Processing Time",
            value=f"{duration:.2f}s"
        )
    
    # Add LLM Analysis metric if available
    if 'llm_analysis_results' in results and results['llm_analysis_results']:
        col5 = st.columns(1)[0]
        with col5:
            st.metric(
                label="LLM Analyses",
                value=stats.get('total_llm_analyses', 0)
            )
    
    # Detailed Results
    st.subheader("📄 File Processing Details")
    
    # Get LLM analysis results if available
    llm_results = results.get('llm_analysis_results', [])
    
    for i, (proc_result, pii_result, analysis_result) in enumerate(zip(
        results['processing_results'],
        results['pii_results'],
        results['analysis_results']
    )):
        with st.expander(f"File {i+1}: {os.path.basename(proc_result['file_path'])}", expanded=False):
            
            # File processing status
            if proc_result['status'] == 'success':
                st.success(f"✅ File processed successfully")
                
                # Show original content (first 500 chars)
                st.subheader("📝 Original Content (Preview)")
                original_preview = proc_result['content'][:500]
                if len(proc_result['content']) > 500:
                    original_preview += "..."
                st.text_area("Original Content Preview", original_preview, height=100, disabled=True, label_visibility="collapsed")
                
                # Show image-specific information if this is an image file
                if proc_result.get('file_type', '').lower() in ['.png', '.jpg', '.jpeg']:
                    st.subheader("🖼️ Image Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**OCR Extracted Text:**")
                        ocr_text = proc_result.get('ocr_text', 'No OCR text available')
                        ocr_preview = ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text
                        st.text_area("OCR Text", ocr_preview, height=100, disabled=True, label_visibility="collapsed")
                    
                    with col2:
                        st.write("**Gemma3 Image Description:**")
                        description_status = proc_result.get('description_status', 'error')
                        if description_status == 'success':
                            image_desc = proc_result.get('image_description', 'No description available')
                            desc_preview = image_desc[:300] + "..." if len(image_desc) > 300 else image_desc
                            st.text_area("Image Description", desc_preview, height=100, disabled=True, label_visibility="collapsed")
                        elif description_status == 'skipped':
                            st.info("ℹ️ Image descriptions disabled by user")
                        else:
                            error_msg = proc_result.get('description_error', 'Unknown error')
                            st.error(f"❌ Description failed: {error_msg}")
                
                # PII Detection Results
                if pii_result['status'] == 'success':
                    st.subheader("🔒 PII Detection & Masking")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**PII Items Detected:** {pii_result['detection_count']}")
                        if pii_result['detected_items']:
                            for item in pii_result['detected_items'][:5]:  # Show first 5
                                original_value = item.get('original', 'N/A')
                                st.write(f"• {item['type']}: {original_value}")
                            if len(pii_result['detected_items']) > 5:
                                st.write(f"... and {len(pii_result['detected_items']) - 5} more")
                    
                    with col2:
                        # Show masked content (first 500 chars)
                        st.write("**Masked Content (Preview):**")
                        masked_preview = pii_result['masked_text'][:500]
                        if len(pii_result['masked_text']) > 500:
                            masked_preview += "..."
                        st.text_area("Masked Content Preview", masked_preview, height=100, disabled=True, label_visibility="collapsed")
                    
                    # Download masked content
                    st.download_button(
                        label="📥 Download Masked File",
                        data=pii_result['masked_text'],
                        file_name=f"masked_{os.path.basename(proc_result['file_path'])}.txt",
                        mime="text/plain"
                    )
                
                # Security Analysis Results
                if analysis_result['status'] == 'success':
                    st.subheader("🔐 Security Analysis")
                    analysis = analysis_result['analysis']
                    summary = analysis.get('summary', {})
                    
                    # Security findings metrics
                    sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
                    
                    with sec_col1:
                        st.metric("IAM Policies", summary.get('total_iam_policies', 0))
                    
                    with sec_col2:
                        st.metric("Firewall Rules", summary.get('total_firewall_rules', 0))
                    
                    with sec_col3:
                        st.metric("IDS Logs", summary.get('total_ids_logs', 0))
                    
                    with sec_col4:
                        st.metric("Vulnerabilities", summary.get('total_vulnerabilities', 0))
                    
                    # Detailed findings
                    if analysis.get('findings'):
                        st.write("**Detailed Security Findings:**")
                        for finding in analysis['findings'][:10]:  # Show first 10
                            st.write(f"• {finding.get('type', 'Unknown')}: {finding.get('description', 'No description')}")
                        
                        if len(analysis['findings']) > 10:
                            st.write(f"... and {len(analysis['findings']) - 10} more findings")
                
                # LLM Analysis Results
                if i < len(llm_results) and llm_results[i]['status'] == 'success':
                    st.subheader("🤖 LLM Analysis")
                    llm_result = llm_results[i]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**File Description:**")
                        description = llm_result.get('file_description', 'No description available')
                        desc_preview = description[:400] + "..." if len(description) > 400 else description
                        st.text_area("Description", desc_preview, height=120, disabled=True, label_visibility="collapsed")
                    
                    with col2:
                        st.write("**Key Security Findings:**")
                        key_findings = llm_result.get('key_findings', [])
                        if key_findings:
                            for j, finding in enumerate(key_findings[:3]):  # Show first 3 findings
                                finding_preview = finding[:200] + "..." if len(finding) > 200 else finding
                                st.write(f"**{j+1}.** {finding_preview}")
                            if len(key_findings) > 3:
                                st.write(f"... and {len(key_findings) - 3} more findings")
                        else:
                            st.write("No key findings available")
                    
                    # Show all findings in expandable section
                    if len(key_findings) > 3:
                        with st.expander("📋 View All Security Findings", expanded=False):
                            for j, finding in enumerate(key_findings, 1):
                                st.write(f"**{j}.** {finding}")
                
                elif i < len(llm_results) and llm_results[i]['status'] == 'error':
                    st.subheader("🤖 LLM Analysis")
                    st.error(f"❌ LLM Analysis failed: {llm_results[i].get('error', 'Unknown error')}")
                
            else:
                st.error(f"❌ File processing failed: {proc_result.get('error', 'Unknown error')}")

def display_summary_report(results: Dict[str, Any]):
    """Display a comprehensive summary report"""
    
    if results['status'] == 'error':
        return
    
    st.subheader("📊 Comprehensive Analysis Report")
    
    # Create summary data with consistent string values
    summary_data = {
        'Metric': [
            'Total Files Processed',
            'Successful Files',
            'Failed Files',
            'Total PII Detections',
            'Total Security Findings',
            'Total LLM Analyses',
            'Processing Duration (seconds)'
        ],
        'Value': [
            str(results['processing_stats']['total_files']),
            str(results['processing_stats']['successful_files']),
            str(results['processing_stats']['failed_files']),
            str(results['processing_stats']['total_pii_detections']),
            str(results['processing_stats']['total_security_findings']),
            str(results['processing_stats'].get('total_llm_analyses', 0)),
            f"{results.get('processing_duration_seconds', 0):.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download comprehensive report with serializable data
    processing_stats = results['processing_stats'].copy()
    
    # Convert datetime objects to ISO format strings
    if processing_stats.get('start_time'):
        processing_stats['start_time'] = processing_stats['start_time'].isoformat() if hasattr(processing_stats['start_time'], 'isoformat') else str(processing_stats['start_time'])
    if processing_stats.get('end_time'):
        processing_stats['end_time'] = processing_stats['end_time'].isoformat() if hasattr(processing_stats['end_time'], 'isoformat') else str(processing_stats['end_time'])
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'processing_stats': processing_stats,
        'processing_duration_seconds': float(results.get('processing_duration_seconds', 0)),
        'file_count': len(results['processing_results']),
        'summary': {
            'total_pii_items': int(results['processing_stats']['total_pii_detections']),
            'total_security_findings': int(results['processing_stats']['total_security_findings']),
            'total_llm_analyses': int(results['processing_stats'].get('total_llm_analyses', 0)),
            'success_rate': float((results['processing_stats']['successful_files'] / results['processing_stats']['total_files'] * 100) if results['processing_stats']['total_files'] > 0 else 0)
        }
    }
    
    report_json = json.dumps(report_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Complete Report (JSON)",
        data=report_json,
        file_name=f"security_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def display_llm_analysis_table(llm_results: List[Dict[str, Any]]):
    """Display LLM analysis results in table format"""
    
    st.subheader("🤖 LLM Analysis Results")
    st.markdown("File descriptions and security findings generated by AI analysis:")
    
    # Display each result in a card format for better readability
    for i, result in enumerate(llm_results, 1):
        with st.container():
            st.markdown("---")
            
            if result['status'] == 'success':
                file_name = result.get('file_name', 'Unknown')
                file_type = result.get('file_type', 'Unknown')
                description = result.get('file_description', 'No description available')
                key_findings = result.get('key_findings', [])
                
                # File header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### 📄 {file_name}{file_type}")
                with col2:
                    st.markdown(f"**Type:** {file_type}")
                
                # File description
                st.markdown("**📝 File Description:**")
                st.info(description)
                
                # Key findings
                st.markdown("**🔍 Key Security Findings:**")
                if isinstance(key_findings, list) and key_findings:
                    for j, finding in enumerate(key_findings, 1):
                        st.markdown(f"**{j}.** {finding}")
                elif key_findings:
                    st.markdown(f"• {key_findings}")
                else:
                    st.warning("No key findings available")
                
                # Analysis timestamp
                timestamp = result.get('analysis_timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    st.caption(f"Analysis completed: {timestamp}")
                    
            else:
                # Handle error cases
                file_name = result.get('file_name', 'Unknown')
                file_type = result.get('file_type', 'Unknown')
                error_msg = result.get('error', 'Unknown error')
                
                st.markdown(f"### ❌ {file_name}{file_type}")
                st.error(f"Analysis failed: {error_msg}")
    
    # Download section
    if llm_results:
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Create CSV data for download
        csv_data = []
        for result in llm_results:
            if result['status'] == 'success':
                file_name = result.get('file_name', 'Unknown')
                file_type = result.get('file_type', 'Unknown')
                description = result.get('file_description', 'No description available')
                key_findings = result.get('key_findings', [])
                
                # Format key findings for CSV
                if isinstance(key_findings, list):
                    formatted_findings = ' | '.join([f"{i+1}. {finding}" for i, finding in enumerate(key_findings)])
                else:
                    formatted_findings = f"• {key_findings}"
                
                csv_data.append({
                    'File Name': file_name,
                    'File Type': file_type,
                    'File Description': description,
                    'Key Findings': formatted_findings,
                    'Status': 'Success'
                })
            else:
                file_name = result.get('file_name', 'Unknown')
                file_type = result.get('file_type', 'Unknown')
                error_msg = result.get('error', 'Unknown error')
                
                csv_data.append({
                    'File Name': file_name,
                    'File Type': file_type,
                    'File Description': 'Analysis failed',
                    'Key Findings': f'Error: {error_msg}',
                    'Status': 'Failed'
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download LLM Analysis Results (CSV)",
                data=csv_string,
                file_name=f"llm_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ VIT Campus Connect - File Processor</h1>', unsafe_allow_html=True)
    st.markdown("Upload files to detect PII, mask sensitive information, and analyze security content.")
    
    # Initialize workflow
    workflow = initialize_workflow()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Upload method info
    st.sidebar.markdown("### 📁 Upload Methods")
    st.sidebar.markdown("""
    **Single Files:** Upload individual files one by one or select multiple files at once.
    
    **Entire Folder:** Upload a complete folder with all its contents. The system will automatically process all supported files within the folder and its subdirectories.
    """)
    
    # Supported file types
    st.sidebar.markdown("### 📄 Supported File Types")
    st.sidebar.markdown("""
    - **Documents:** TXT, PDF, DOCX, XLSX, PPTX, CSV
    - **Images:** PNG, JPG, JPEG (with OCR + AI Description)
    """)
    
    # Image processing options
    st.sidebar.markdown("### 🤖 AI Image Analysis")
    
    # Checkbox for enabling/disabling image descriptions
    enable_image_descriptions = st.sidebar.checkbox(
        "Enable AI Image Descriptions",
        value=True,
        help="When enabled, image files will be processed with both OCR (text extraction) and AI description using Gemma3. When disabled, only OCR will be performed."
    )
    
    if enable_image_descriptions:
        if st.sidebar.button("Test Gemma3 Connection"):
            with st.spinner("Testing connection to Gemma3..."):
                workflow = initialize_workflow()
                test_result = workflow.file_processor.image_descriptor.test_connection()
                
                if test_result['status'] == 'success':
                    st.sidebar.success("✅ Gemma3 Connected")
                    if test_result['model_available']:
                        st.sidebar.success(f"✅ Model {test_result['target_model']} Available")
                    else:
                        st.sidebar.warning(f"⚠️ Model {test_result['target_model']} Not Found")
                        st.sidebar.write("Available models:")
                        for model in test_result['available_models']:
                            st.sidebar.write(f"• {model}")
                else:
                    st.sidebar.error("❌ Gemma3 Connection Failed")
                    st.sidebar.error(test_result.get('error', 'Unknown error'))
    
    # LLM Analysis section
    st.sidebar.markdown("### 🤖 LLM Analysis")
    st.sidebar.markdown("""
    **AI-Powered Analysis:** The system now includes LLM analysis that provides:
    - **File Descriptions:** AI-generated descriptions of file content
    - **Security Findings:** Security-focused analysis and risk assessment
    - **Structured Output:** Table format with key findings
    """)
    
    if st.sidebar.button("Test LLM Analyzer"):
        with st.spinner("Testing LLM analyzer connection..."):
            workflow = initialize_workflow()
            test_result = workflow.llm_analyzer.test_connection()
            
            if test_result['status'] == 'success':
                st.sidebar.success("✅ LLM Analyzer Connected")
                if test_result['model_available']:
                    st.sidebar.success(f"✅ Model {test_result['target_model']} Available")
                else:
                    st.sidebar.warning(f"⚠️ Model {test_result['target_model']} Not Found")
            else:
                st.sidebar.error("❌ LLM Analyzer Connection Failed")
                st.sidebar.error(test_result.get('error', 'Unknown error'))
    
    if enable_image_descriptions:
        st.sidebar.markdown("**Note:** Image files will be processed with both OCR (text extraction) and AI description using Gemma3.")
    else:
        st.sidebar.markdown("**Note:** Image files will be processed with OCR only (no AI descriptions).")
    
    # File upload section
    st.subheader("📁 File Upload")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["Single Files", "Entire Folder"],
        horizontal=True,
        help="Select whether to upload individual files or an entire folder"
    )
    
    uploaded_files = None
    
    if upload_method == "Single Files":
        uploaded_files = st.file_uploader(
            "Choose files to process",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'xlsx', 'pptx', 'png', 'jpg', 'jpeg', 'csv'],
            help="Supported formats: TXT, PDF, DOCX, XLSX, PPTX, PNG, JPG, JPEG, CSV"
        )
    else:  # Entire Folder
        uploaded_files = st.file_uploader(
            "Choose folder to process",
            accept_multiple_files="directory",
            type=['txt', 'pdf', 'docx', 'xlsx', 'pptx', 'png', 'jpg', 'jpeg', 'csv'],
            help="Select a folder to upload all supported files from it. Supported formats: TXT, PDF, DOCX, XLSX, PPTX, PNG, JPG, JPEG, CSV"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        if upload_method == "Entire Folder":
            st.success(f"✅ Folder uploaded successfully! Found {len(uploaded_files)} file(s) in the folder.")
        else:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Show uploaded files with enhanced display for folder uploads
        with st.expander("📋 Uploaded Files", expanded=False):
            if upload_method == "Entire Folder":
                # Group files by directory for better organization
                files_by_dir = {}
                for file in uploaded_files:
                    dir_path = os.path.dirname(file.name)
                    if dir_path not in files_by_dir:
                        files_by_dir[dir_path] = []
                    files_by_dir[dir_path].append(file)
                
                for dir_path, files in files_by_dir.items():
                    if dir_path:
                        st.write(f"📁 **{dir_path}**")
                    else:
                        st.write("📁 **Root Directory**")
                    
                    for file in files:
                        filename = os.path.basename(file.name)
                        st.write(f"  • {filename} ({file.size} bytes)")
            else:
                # Regular file list for single file uploads
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. **{file.name}** ({file.size} bytes)")
        
        # Process files button
        button_text = f"🚀 Process {len(uploaded_files)} File{'s' if len(uploaded_files) > 1 else ''}"
        if upload_method == "Entire Folder":
            button_text += " from Folder"
        
        if st.button(button_text, type="primary", use_container_width=True):
            
            # Save uploaded files
            save_message = f"Saving {len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''}..."
            if upload_method == "Entire Folder":
                save_message = f"Saving folder contents ({len(uploaded_files)} files)..."
            
            with st.spinner(save_message):
                file_paths, temp_dir = save_uploaded_files(uploaded_files)
            
            # Initialize progress tracking
            progress_container = st.container()
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            
            # Progress callback function
            def progress_callback(progress_data):
                # Update progress bar
                overall_progress = progress_data.get('progress_percentage', 0) / 100
                progress_bar.progress(overall_progress)
                
                # Update status text
                current_step = progress_data.get('current_step', 'Processing')
                current_file = progress_data.get('current_file_name', '')
                file_index = progress_data.get('current_file_index', 0) + 1
                total_files = progress_data.get('total_files', 1)
                
                status_text.text(f"🔄 {current_step}")
                
                # Update details
                if current_file:
                    details = f"📄 Processing file {file_index}/{total_files}: {current_file}"
                else:
                    step_index = progress_data.get('current_step_index', 0)
                    details = f"📊 Step {step_index + 1}/7: {current_step}"
                
                # Add processing stats
                stats = progress_data.get('processing_stats', {})
                if stats.get('files_processed', 0) > 0 or stats.get('pii_items_found', 0) > 0 or stats.get('security_findings', 0) > 0:
                    details += f"\n📈 Stats: {stats.get('files_processed', 0)} files processed, {stats.get('pii_items_found', 0)} PII items found, {stats.get('security_findings', 0)} security findings"
                
                details_text.text(details)
            
            # Set progress callback
            workflow.set_progress_callback(progress_callback)
            
            # Process files with detailed progress
            try:
                results = workflow.process_files(file_paths, enable_image_descriptions=enable_image_descriptions)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                details_text.empty()
                
                # Store results in session state
                st.session_state.processing_results = results
                st.session_state.temp_dir = temp_dir
                st.session_state.upload_method = upload_method
                
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
                # Show completion message
                st.success(f"✅ Processing completed! {results['processing_stats']['successful_files']} files processed successfully.")
                
                # Refresh the page to show results
                st.rerun()
                
            except Exception as e:
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()
                details_text.empty()
                st.error(f"❌ Processing failed: {str(e)}")
                st.rerun()
    
    # Display results if available
    if 'processing_results' in st.session_state:
        results = st.session_state.processing_results
        upload_method = st.session_state.get('upload_method', 'Single Files')
        
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        # Show upload method info
        if upload_method == "Entire Folder":
            st.info(f"📁 Results from folder upload - {results['processing_stats']['total_files']} files processed")
        else:
            st.info(f"📄 Results from individual file upload - {results['processing_stats']['total_files']} files processed")
        
        # Processing Results
        display_processing_results(results)
        
        # Summary Report
        display_summary_report(results)
        
        # LLM Analysis Table
        if 'llm_analysis_results' in results and results['llm_analysis_results']:
            display_llm_analysis_table(results['llm_analysis_results'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            if 'processing_results' in st.session_state:
                del st.session_state.processing_results
            if 'temp_dir' in st.session_state:
                del st.session_state.temp_dir
            if 'upload_method' in st.session_state:
                del st.session_state.upload_method
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🛡️ VIT Campus Connect - Cybersecurity File Processing System</p>
        <p>Upload individual files or entire folders to automatically detect PII, mask sensitive data, and analyze security content</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
