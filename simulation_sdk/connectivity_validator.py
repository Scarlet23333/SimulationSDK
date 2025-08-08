"""
Testing utilities for connectivity and validation.

This module provides ConnectivityTester for testing API connectivity
and SimulationValidator for validating simulation outputs.
"""

import json
import smtplib
import ssl
import traceback
from datetime import datetime
from typing import Dict, List, Any

from .models import (
    SimulationResponse,
    AgentSimulatedResult,
)


class ConnectivityTester:
    """
    Tests API connectivity using read-only endpoints.
    
    This class provides methods to test connectivity to various external services
    without affecting production data.
    """
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def test_google_slides(self, credentials: Dict[str, Any]) -> bool:
        """
        Test Google Slides API connectivity.
        
        Args:
            credentials: Dictionary containing Google API credentials
                - Should include 'client_id', 'client_secret', 'refresh_token'
                - Or 'service_account_key' for service account auth
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import Google API libraries only when needed
            try:
                from googleapiclient.discovery import build
                from google.oauth2.credentials import Credentials
                from google.oauth2 import service_account
            except ImportError:
                self.test_results['google_slides'] = {
                    'success': False,
                    'error': 'Google API client libraries not installed. Run: pip install google-api-python-client google-auth',
                    'timestamp': datetime.now().isoformat()
                }
                return False
            
            # Build credentials based on auth type
            if 'service_account_key' in credentials:
                # Service account authentication
                creds = service_account.Credentials.from_service_account_info(
                    credentials['service_account_key'],
                    scopes=['https://www.googleapis.com/auth/presentations.readonly']
                )
            else:
                # OAuth2 credentials
                creds = Credentials(
                    token=credentials.get('access_token'),
                    refresh_token=credentials.get('refresh_token'),
                    token_uri='https://oauth2.googleapis.com/token',
                    client_id=credentials.get('client_id'),
                    client_secret=credentials.get('client_secret'),
                    scopes=['https://www.googleapis.com/auth/presentations.readonly']
                )
            
            # Build the service
            service = build('slides', 'v1', credentials=creds)
            
            # Test API by listing presentations (limited to 1)
            # This is a read-only operation that doesn't affect production
            result = service.presentations().get(
                presentationId='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'  # Google's public template
            ).execute()
            
            self.test_results['google_slides'] = {
                'success': True,
                'message': 'Successfully connected to Google Slides API',
                'api_response': 'Presentation retrieved successfully',
                'timestamp': datetime.now().isoformat()
            }
            return True
            
        except Exception as e:
            self.test_results['google_slides'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_google_sheets(self, credentials: Dict[str, Any]) -> bool:
        """
        Test Google Sheets API connectivity.
        
        Args:
            credentials: Dictionary containing Google API credentials
                - Should include 'client_id', 'client_secret', 'refresh_token'
                - Or 'service_account_key' for service account auth
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import Google API libraries only when needed
            try:
                from googleapiclient.discovery import build
                from google.oauth2.credentials import Credentials
                from google.oauth2 import service_account
            except ImportError:
                self.test_results['google_sheets'] = {
                    'success': False,
                    'error': 'Google API client libraries not installed. Run: pip install google-api-python-client google-auth',
                    'timestamp': datetime.now().isoformat()
                }
                return False
            
            # Build credentials based on auth type
            if 'service_account_key' in credentials:
                # Service account authentication
                creds = service_account.Credentials.from_service_account_info(
                    credentials['service_account_key'],
                    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
                )
            else:
                # OAuth2 credentials
                creds = Credentials(
                    token=credentials.get('access_token'),
                    refresh_token=credentials.get('refresh_token'),
                    token_uri='https://oauth2.googleapis.com/token',
                    client_id=credentials.get('client_id'),
                    client_secret=credentials.get('client_secret'),
                    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
                )
            
            # Build the service
            service = build('sheets', 'v4', credentials=creds)
            
            # Test API by reading from a public Google Sheets template
            # This is a read-only operation that doesn't affect production
            spreadsheet_id = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'  # Public template
            range_name = 'A1:A1'
            
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            self.test_results['google_sheets'] = {
                'success': True,
                'message': 'Successfully connected to Google Sheets API',
                'api_response': 'Spreadsheet data retrieved successfully',
                'timestamp': datetime.now().isoformat()
            }
            return True
            
        except Exception as e:
            self.test_results['google_sheets'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_email_smtp(self, config: Dict[str, Any]) -> bool:
        """
        Test SMTP email connectivity.
        
        Args:
            config: Dictionary containing SMTP configuration
                - smtp_server: SMTP server hostname
                - smtp_port: SMTP server port (e.g., 587 for TLS, 465 for SSL)
                - username: SMTP username
                - password: SMTP password
                - use_tls: Boolean, whether to use TLS (default True)
                - use_ssl: Boolean, whether to use SSL (default False)
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            smtp_server = config.get('smtp_server')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            use_tls = config.get('use_tls', True)
            use_ssl = config.get('use_ssl', False)
            
            if not all([smtp_server, smtp_port, username, password]):
                raise ValueError("Missing required SMTP configuration parameters")
            
            # Create SMTP connection
            if use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=context)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
            
            server.set_debuglevel(0)  # Disable debug output
            
            if use_tls and not use_ssl:
                context = ssl.create_default_context()
                server.starttls(context=context)
            
            # Test authentication
            server.login(username, password)
            
            # Test connection with NOOP command (doesn't send email)
            status = server.noop()
            
            # Close connection
            server.quit()
            
            self.test_results['email_smtp'] = {
                'success': True,
                'message': f'Successfully connected to SMTP server {smtp_server}:{smtp_port}',
                'smtp_response': status,
                'timestamp': datetime.now().isoformat()
            }
            return True
            
        except Exception as e:
            self.test_results['email_smtp'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all configured connectivity tests.
        
        Returns:
            Dictionary with test results for all services
        """
        # This method would be called with appropriate credentials
        # Example usage:
        # tester = ConnectivityTester()
        # tester.test_google_slides(google_creds)
        # tester.test_google_sheets(google_creds)
        # tester.test_email_smtp(smtp_config)
        # results = tester.run_all_tests()
        
        return self.test_results
    
    def get_summary(self) -> str:
        """
        Get a summary of all test results.
        
        Returns:
            Formatted string with test results summary
        """
        if not self.test_results:
            return "No tests have been run yet."
        
        lines = [
            "Connectivity Test Results",
            "=" * 50,
            f"Total Tests: {len(self.test_results)}",
            f"Passed: {sum(1 for r in self.test_results.values() if r.get('success', False))}",
            f"Failed: {sum(1 for r in self.test_results.values() if not r.get('success', False))}",
            "",
        ]
        
        for service, result in self.test_results.items():
            status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
            lines.append(f"{status} {service}")
            
            if result.get('success'):
                lines.append(f"  Message: {result.get('message', 'N/A')}")
            else:
                lines.append(f"  Error: {result.get('error', 'Unknown error')}")
            
            lines.append(f"  Timestamp: {result.get('timestamp', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)


class SimulationValidator:
    """
    Validates simulation outputs match production format.
    
    This class provides methods to validate that simulated agent outputs
    and response templates match the expected production formats.
    """
    
    def __init__(self):
        self.validation_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def validate_agent_output(self, agent_name: str, output: Any) -> bool:
        """
        Validate that agent output matches expected format.
        
        Args:
            agent_name: Name of the agent being validated
            output: The output to validate (can be AgentSimulatedResult or raw output)
        
        Returns:
            True if validation passes, False otherwise
        """
        validation_errors = []
        validation_passed = True
        
        try:
            # Check if output is an AgentSimulatedResult
            if isinstance(output, AgentSimulatedResult):
                # Validate required fields
                if not isinstance(output.tool_calls, list):
                    validation_errors.append("tool_calls must be a list")
                    validation_passed = False
                
                # Validate tool calls format
                for i, tool_call in enumerate(output.tool_calls):
                    if not isinstance(tool_call, dict):
                        validation_errors.append(f"tool_call[{i}] must be a dictionary")
                        validation_passed = False
                    else:
                        # Check required tool call fields
                        if 'tool_name' not in tool_call:
                            validation_errors.append(f"tool_call[{i}] missing 'tool_name'")
                            validation_passed = False
                        if 'parameters' not in tool_call:
                            validation_errors.append(f"tool_call[{i}] missing 'parameters'")
                            validation_passed = False
                
                # Check final_output exists
                if output.final_output is None:
                    validation_errors.append("final_output cannot be None")
                    validation_passed = False
                
            elif isinstance(output, dict):
                # Validate dictionary format for raw output
                # This would be customized based on specific agent requirements
                if 'result' not in output and 'data' not in output:
                    validation_errors.append("Output dictionary must contain 'result' or 'data' field")
                    validation_passed = False
                
            else:
                # For other types, ensure they're serializable
                try:
                    json.dumps(output)
                except (TypeError, ValueError) as e:
                    validation_errors.append(f"Output must be JSON serializable: {str(e)}")
                    validation_passed = False
            
            # Record validation result
            result = {
                'timestamp': datetime.now().isoformat(),
                'passed': validation_passed,
                'output_type': type(output).__name__,
                'errors': validation_errors,
                'warnings': []
            }
            
            if agent_name not in self.validation_results:
                self.validation_results[agent_name] = []
            self.validation_results[agent_name].append(result)
            
            return validation_passed
            
        except Exception as e:
            # Record unexpected validation error
            result = {
                'timestamp': datetime.now().isoformat(),
                'passed': False,
                'output_type': type(output).__name__,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'traceback': traceback.format_exc()
            }
            
            if agent_name not in self.validation_results:
                self.validation_results[agent_name] = []
            self.validation_results[agent_name].append(result)
            
            return False
    
    def validate_response_template(self, template: Any, expected_format: Dict[str, Any]) -> bool:
        """
        Validate that a response template matches expected format.
        
        Args:
            template: The response template to validate (SimulationResponse or dict)
            expected_format: Dictionary describing the expected format
                Example: {
                    'type': 'SimulationResponse',
                    'required_fields': ['success', 'response_data'],
                    'response_data_schema': {
                        'id': 'string',
                        'status': 'string'
                    }
                }
        
        Returns:
            True if validation passes, False otherwise
        """
        validation_errors = []
        validation_passed = True
        
        try:
            # Check if template is a SimulationResponse
            if isinstance(template, SimulationResponse):
                # Validate required fields
                if template.success is None:
                    validation_errors.append("Missing required field: success")
                    validation_passed = False
                
                if not isinstance(template.response_data, dict):
                    validation_errors.append("response_data must be a dictionary")
                    validation_passed = False
                
                # Validate against expected schema if provided
                if 'response_data_schema' in expected_format:
                    schema = expected_format['response_data_schema']
                    for field, field_type in schema.items():
                        if field not in template.response_data:
                            validation_errors.append(f"Missing field in response_data: {field}")
                            validation_passed = False
                        else:
                            # Basic type checking
                            value = template.response_data[field]
                            if field_type == 'string' and not isinstance(value, str):
                                validation_errors.append(f"Field {field} should be string, got {type(value).__name__}")
                                validation_passed = False
                            elif field_type == 'number' and not isinstance(value, (int, float)):
                                validation_errors.append(f"Field {field} should be number, got {type(value).__name__}")
                                validation_passed = False
                            elif field_type == 'boolean' and not isinstance(value, bool):
                                validation_errors.append(f"Field {field} should be boolean, got {type(value).__name__}")
                                validation_passed = False
                            elif field_type == 'array' and not isinstance(value, list):
                                validation_errors.append(f"Field {field} should be array, got {type(value).__name__}")
                                validation_passed = False
                            elif field_type == 'object' and not isinstance(value, dict):
                                validation_errors.append(f"Field {field} should be object, got {type(value).__name__}")
                                validation_passed = False
                
            elif isinstance(template, dict):
                # Validate dictionary format
                required_fields = expected_format.get('required_fields', [])
                for field in required_fields:
                    if field not in template:
                        validation_errors.append(f"Missing required field: {field}")
                        validation_passed = False
                
            else:
                validation_errors.append(f"Template must be SimulationResponse or dict, got {type(template).__name__}")
                validation_passed = False
            
            # Record validation result
            result = {
                'timestamp': datetime.now().isoformat(),
                'passed': validation_passed,
                'template_type': type(template).__name__,
                'errors': validation_errors,
                'expected_format': expected_format
            }
            
            # Store under special key for templates
            if '_templates' not in self.validation_results:
                self.validation_results['_templates'] = []
            self.validation_results['_templates'].append(result)
            
            return validation_passed
            
        except Exception as e:
            # Record unexpected validation error
            result = {
                'timestamp': datetime.now().isoformat(),
                'passed': False,
                'template_type': type(template).__name__ if template else 'None',
                'errors': [f"Validation exception: {str(e)}"],
                'expected_format': expected_format,
                'traceback': traceback.format_exc()
            }
            
            if '_templates' not in self.validation_results:
                self.validation_results['_templates'] = []
            self.validation_results['_templates'].append(result)
            
            return False
    
    def get_validation_report(self) -> str:
        """
        Generate a detailed validation report.
        
        Returns:
            Formatted string with validation results
        """
        if not self.validation_results:
            return "No validations have been performed yet."
        
        lines = [
            "Simulation Validation Report",
            "=" * 50,
        ]
        
        # Count totals
        total_validations = 0
        total_passed = 0
        
        for key, results in self.validation_results.items():
            total_validations += len(results)
            total_passed += sum(1 for r in results if r.get('passed', False))
        
        lines.extend([
            f"Total Validations: {total_validations}",
            f"Passed: {total_passed}",
            f"Failed: {total_validations - total_passed}",
            "",
        ])
        
        # Report by agent/category
        for key, results in self.validation_results.items():
            if key == '_templates':
                lines.append("Template Validations:")
            else:
                lines.append(f"Agent: {key}")
            
            for result in results:
                status = "✓" if result.get('passed', False) else "✗"
                lines.append(f"  {status} {result.get('timestamp', 'N/A')}")
                
                if not result.get('passed', False):
                    for error in result.get('errors', []):
                        lines.append(f"    ERROR: {error}")
                
                for warning in result.get('warnings', []):
                    lines.append(f"    WARN: {warning}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def clear_results(self) -> None:
        """Clear all validation results."""
        self.validation_results.clear()