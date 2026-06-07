#!/usr/bin/env python3
"""
API Key Manager with Quota Handling and Fallback
Manages multiple API keys and handles quota limits gracefully
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import google.generativeai as genai

logger = logging.getLogger(__name__)

class APIKeyManager:
    """
    Manages API keys with quota handling and fallback mechanisms
    """
    
    def __init__(self):
        self.primary_key = os.getenv('GOOGLE_API_KEY_PRIMARY', 'your-primary-google-api-key-here')
        self.fallback_key = os.getenv('GOOGLE_API_KEY_FALLBACK', 'your-fallback-google-api-key-here')
        self.current_key = self.primary_key
        self.quota_errors = {}
        self.retry_delays = {}
        self.max_retries = 3
        self.retry_delay = 60  # seconds
        
    def get_available_key(self) -> Optional[str]:
        """
        Get an available API key, handling quota limits
        """
        # Check if current key is available
        if self._is_key_available(self.current_key):
            return self.current_key
            
        # Try fallback key
        if self._is_key_available(self.fallback_key):
            logger.info("🔄 Switching to fallback API key")
            self.current_key = self.fallback_key
            return self.fallback_key
            
        # Both keys are unavailable
        logger.warning("⚠️ All API keys are currently unavailable due to quota limits")
        return None
    
    def _is_key_available(self, key: str) -> bool:
        """
        Check if a key is available (not hitting quota limits)
        """
        if key not in self.quota_errors:
            return True
            
        error_time = self.quota_errors[key]
        retry_after = self.retry_delays.get(key, self.retry_delay)
        
        # Check if enough time has passed
        if datetime.now() - error_time > timedelta(seconds=retry_after):
            # Reset the error
            del self.quota_errors[key]
            if key in self.retry_delays:
                del self.retry_delays[key]
            return True
            
        return False
    
    def handle_quota_error(self, key: str, error: Exception, retry_after: int = None):
        """
        Handle quota error and set retry delay
        """
        self.quota_errors[key] = datetime.now()
        if retry_after:
            self.retry_delays[key] = retry_after
        else:
            self.retry_delays[key] = self.retry_delay
            
        logger.warning(f"⚠️ API key {key[:10]}... hit quota limit. Retry after {retry_after or self.retry_delay} seconds")
    
    def get_gemini_model(self, model_name: str = "gemini-1.5-flash"):
        """
        Get a Gemini model with available API key
        """
        key = self.get_available_key()
        if not key:
            raise Exception("No available API keys. All keys are hitting quota limits.")
            
        genai.configure(api_key=key)
        return genai.GenerativeModel(model_name)
    
    def test_key(self, key: str) -> bool:
        """
        Test if an API key is working
        """
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Test")
            return True
        except Exception as e:
            logger.error(f"❌ API key test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of all API keys
        """
        status = {
            "primary_key": {
                "key": self.primary_key[:10] + "...",
                "available": self._is_key_available(self.primary_key),
                "quota_error": self.primary_key in self.quota_errors,
                "retry_after": self.retry_delays.get(self.primary_key, 0)
            },
            "fallback_key": {
                "key": self.fallback_key[:10] + "...",
                "available": self._is_key_available(self.fallback_key),
                "quota_error": self.fallback_key in self.quota_errors,
                "retry_after": self.retry_delays.get(self.fallback_key, 0)
            },
            "current_key": self.current_key[:10] + "..." if self.current_key else None
        }
        return status

# Global instance
api_key_manager = APIKeyManager()
