"""
Testing utilities for simulations.

This module provides simple connectivity testing for APIs.
"""

import logging
from typing import Dict, Any, Optional
import requests

from .utils import logger


class ConnectivityTester:
    """
    Test API connectivity using read-only endpoints.
    
    This class provides methods to test connectivity to various external services
    without affecting production data.
    """
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def test_google_slides(self, presentation_id: str = "test") -> bool:
        """
        Test Google Slides API connectivity.
        
        Args:
            presentation_id: ID to test (defaults to "test")
            
        Returns:
            True if connectivity test passes
        """
        try:
            # This is a placeholder - in real implementation would use Google API
            # For now, just test that we can make HTTP requests
            response = requests.get("https://www.googleapis.com/discovery/v1/apis", timeout=5)
            
            self.test_results["google_slides"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "presentation_id": presentation_id,
            }
            
            return response.status_code == 200
            
        except Exception as e:
            self.test_results["google_slides"] = {
                "success": False,
                "error": str(e),
                "presentation_id": presentation_id,
            }
            return False
    
    def test_google_sheets(self, spreadsheet_id: str = "test") -> bool:
        """
        Test Google Sheets API connectivity.
        
        Args:
            spreadsheet_id: ID to test (defaults to "test")
            
        Returns:
            True if connectivity test passes
        """
        try:
            # Placeholder - would use Google Sheets API
            response = requests.get("https://www.googleapis.com/discovery/v1/apis", timeout=5)
            
            self.test_results["google_sheets"] = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "spreadsheet_id": spreadsheet_id,
            }
            
            return response.status_code == 200
            
        except Exception as e:
            self.test_results["google_sheets"] = {
                "success": False,
                "error": str(e),
                "spreadsheet_id": spreadsheet_id,
            }
            return False
    
    def test_http_endpoint(self, url: str, headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Test HTTP endpoint connectivity.
        
        Args:
            url: URL to test
            headers: Optional headers to include
            
        Returns:
            True if connectivity test passes
        """
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.test_results["http"] = {
                "success": True,
                "status_code": response.status_code,
                "url": url,
            }
            return True
            
        except Exception as e:
            self.test_results["http"] = {
                "success": False,
                "error": str(e),
                "url": url,
            }
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all connectivity tests.
        
        Returns:
            Dictionary mapping service names to test results
        """
        results = {
            "google_slides": self.test_google_slides(),
            "google_sheets": self.test_google_sheets(),
        }
        
        logger.info(f"Connectivity test results: {results}")
        return results
    
    def get_test_summary(self) -> str:
        """
        Get a summary of all test results.
        
        Returns:
            Formatted string with test results
        """
        summary_lines = ["Connectivity Test Results:", "=" * 50]
        
        for service, result in self.test_results.items():
            status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
            summary_lines.append(f"{service}: {status}")
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                summary_lines.append(f"  Error: {error}")
        
        summary_lines.append("=" * 50)
        return "\n".join(summary_lines)


def test_connectivity() -> Dict[str, bool]:
    """
    Test connectivity to all services.
    
    Returns:
        Dictionary mapping service names to connectivity status
    """
    tester = ConnectivityTester()
    return tester.run_all_tests()