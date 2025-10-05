"""
Image Description Module for VIT Campus Connect
Uses Gemma3 model to generate descriptions of images
"""

import base64
import requests
import logging
from typing import Dict, Any, Optional
import os
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDescriptor:
    """Class for generating image descriptions using Gemma3 model"""
    
    def __init__(self, gemma_url: str = "http://127.0.0.1:11500"):
        """
        Initialize the ImageDescriptor
        
        Args:
            gemma_url (str): URL of the Gemma3 Ollama service
        """
        self.gemma_url = gemma_url
        self.model_name = "gemma3:4b"  # Adjust based on your actual model name
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string efficiently
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
    
    def resize_image_if_needed(self, image_path: str, max_size: tuple = (1024, 1024)) -> str:
        """
        Resize image if it's too large for the model
        
        Args:
            image_path (str): Path to the image file
            max_size (tuple): Maximum size (width, height)
            
        Returns:
            str: Path to the resized image (or original if no resize needed)
        """
        try:
            with Image.open(image_path) as img:
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    # Calculate new size maintaining aspect ratio
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Save resized image to temporary file
                    temp_path = image_path.replace('.', '_resized.')
                    img.save(temp_path, format=img.format)
                    logger.info(f"Resized image from {img.size} to {max_size}")
                    return temp_path
                else:
                    return image_path
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {str(e)}")
            return image_path
    
    def describe_image(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Generate description for an image using Gemma3
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Custom prompt for image description
            
        Returns:
            Dict containing description and metadata
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                return {
                    'status': 'error',
                    'error': f'Image file not found: {image_path}',
                    'description': '',
                    'image_path': image_path
                }
            
            # Resize image if needed
            processed_image_path = self.resize_image_if_needed(image_path)
            logger.info(f"Processing image: {processed_image_path}")
            
            # Encode image to base64 (required by Ollama API)
            base64_image = self.encode_image_to_base64(processed_image_path)
            logger.info(f"Image encoded to base64, length: {len(base64_image)} characters")
            
            # Default prompt for image description (similar to ollama run default)
            if prompt is None:
                prompt = ""
            
            # Try using the chat endpoint which might handle images better
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            # Make request to Gemma3 service using chat endpoint
            response = requests.post(
                f"{self.gemma_url}/api/chat",
                json=payload,
                timeout=120  # 120 second timeout for image processing
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle chat endpoint response format
                if 'message' in result and 'content' in result['message']:
                    description = result['message']['content'].strip()
                else:
                    description = result.get('response', '').strip()
                
                # Clean up temporary resized image if created
                if processed_image_path != image_path and os.path.exists(processed_image_path):
                    try:
                        os.remove(processed_image_path)
                    except:
                        pass
                
                return {
                    'status': 'success',
                    'description': description,
                    'image_path': image_path,
                    'model_used': self.model_name,
                    'prompt_used': prompt,
                    'response_metadata': {
                        'total_duration': result.get('total_duration', 0),
                        'load_duration': result.get('load_duration', 0),
                        'prompt_eval_count': result.get('prompt_eval_count', 0),
                        'prompt_eval_duration': result.get('prompt_eval_duration', 0),
                        'eval_count': result.get('eval_count', 0),
                        'eval_duration': result.get('eval_duration', 0)
                    }
                }
            else:
                error_msg = f"Gemma3 API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'error': error_msg,
                    'description': '',
                    'image_path': image_path
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Request to Gemma3 service timed out"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'description': '',
                'image_path': image_path
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to Gemma3 service at {self.gemma_url}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'description': '',
                'image_path': image_path
            }
        except Exception as e:
            error_msg = f"Unexpected error describing image: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'description': '',
                'image_path': image_path
            }
    
    def batch_describe_images(self, image_paths: list, prompt: str = None) -> list:
        """
        Generate descriptions for multiple images
        
        Args:
            image_paths (list): List of image file paths
            prompt (str): Custom prompt for image descriptions
            
        Returns:
            list: List of description results
        """
        results = []
        for image_path in image_paths:
            result = self.describe_image(image_path, prompt)
            results.append(result)
        return results
    
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
