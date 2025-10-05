"""
Progress Tracking Module for VIT Campus Connect
Provides detailed progress information for frontend display
"""

import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Class for tracking detailed progress of file processing operations"""
    
    def __init__(self, total_files: int):
        """
        Initialize the progress tracker
        
        Args:
            total_files (int): Total number of files to process
        """
        self.total_files = total_files
        self.current_file_index = 0
        self.current_step = "Initializing"
        self.current_file_name = ""
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.progress_callback: Optional[Callable] = None
        
        # Step tracking
        self.steps = [
            "Initializing",
            "Processing Files",
            "Detecting PII",
            "Analyzing Security Content", 
            "LLM Analysis",
            "Generating Outputs",
            "Completed"
        ]
        self.current_step_index = 0
        
        # File processing details
        self.file_details: List[Dict[str, Any]] = []
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'pii_items_found': 0,
            'security_findings': 0,
            'llm_analyses': 0
        }
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function to receive progress updates"""
        self.progress_callback = callback
    
    def _update_progress(self):
        """Send progress update to callback if available"""
        if self.progress_callback:
            progress_data = {
                'current_step': self.current_step,
                'current_file_name': self.current_file_name,
                'current_file_index': self.current_file_index,
                'total_files': self.total_files,
                'progress_percentage': (self.current_file_index / self.total_files) * 100 if self.total_files > 0 else 0,
                'step_progress': (self.current_step_index / len(self.steps)) * 100,
                'processing_stats': self.processing_stats.copy(),
                'elapsed_time': time.time() - self.start_time,
                'file_details': self.file_details.copy()
            }
            self.progress_callback(progress_data)
    
    def start_step(self, step_name: str):
        """Start a new processing step"""
        self.current_step = step_name
        self.step_start_time = time.time()
        if step_name in self.steps:
            self.current_step_index = self.steps.index(step_name)
        else:
            # If step not found, increment the index
            self.current_step_index = min(self.current_step_index + 1, len(self.steps) - 1)
        logger.info(f"Starting: {step_name}")
        self._update_progress()
    
    def start_file_processing(self, file_name: str, file_index: int):
        """Start processing a specific file"""
        self.current_file_name = file_name
        self.current_file_index = file_index
        logger.info(f"Processing file {file_index + 1}/{self.total_files}: {file_name}")
        self._update_progress()
    
    def complete_file_processing(self, file_name: str, success: bool, details: Dict[str, Any] = None):
        """Mark a file as completed"""
        file_detail = {
            'file_name': file_name,
            'status': 'success' if success else 'failed',
            'completed_at': datetime.now().isoformat(),
            'processing_time': time.time() - self.step_start_time,
            'details': details or {}
        }
        self.file_details.append(file_detail)
        
        if success:
            self.processing_stats['files_processed'] += 1
        else:
            self.processing_stats['files_failed'] += 1
        
        logger.info(f"Completed file: {file_name} ({'success' if success else 'failed'})")
        self._update_progress()
    
    def update_pii_detection(self, count: int):
        """Update PII detection count"""
        self.processing_stats['pii_items_found'] += count
        self._update_progress()
    
    def update_security_findings(self, count: int):
        """Update security findings count"""
        self.processing_stats['security_findings'] += count
        self._update_progress()
    
    def update_llm_analyses(self, count: int = 1):
        """Update LLM analysis count"""
        self.processing_stats['llm_analyses'] += count
        self._update_progress()
    
    def complete_step(self, step_name: str):
        """Complete a processing step"""
        step_duration = time.time() - self.step_start_time
        logger.info(f"Completed: {step_name} (took {step_duration:.2f}s)")
        self._update_progress()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        return {
            'current_step': self.current_step,
            'current_file_name': self.current_file_name,
            'current_file_index': self.current_file_index,
            'total_files': self.total_files,
            'progress_percentage': (self.current_file_index / self.total_files) * 100 if self.total_files > 0 else 0,
            'step_progress': (self.current_step_index / len(self.steps)) * 100,
            'processing_stats': self.processing_stats.copy(),
            'elapsed_time': time.time() - self.start_time,
            'file_details': self.file_details.copy()
        }
