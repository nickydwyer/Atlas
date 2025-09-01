#!/usr/bin/env python3
"""
Environment Validator Module for Atlas

Provides comprehensive validation of environment configuration,
including connectivity checks for Neo4j, LLM providers, and MCP servers.
"""

import os
import sys
import json
import asyncio
import subprocess
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Neo4j imports
try:
    from langchain_neo4j import Neo4jGraph
    neo4j_available = True
except ImportError:
    neo4j_available = False

# LLM client imports for validation
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from env_config import EnvironmentConfig


class EnvironmentValidator:
    """Validates environment configuration and tests connectivity."""
    
    def __init__(self, env_config: EnvironmentConfig):
        """
        Initialize validator with environment configuration.
        
        Args:
            env_config: EnvironmentConfig instance with loaded environment
        """
        self.env_config = env_config
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': env_config.env,
            'loaded_file': env_config.loaded_env_file,
            'components': {},
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
    
    async def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Complete validation results
        """
        print("\n🔍 Starting comprehensive environment validation...\n")
        
        # Validate environment variables first
        self._validate_environment_vars()
        
        # Validate each component
        await self._validate_neo4j()
        await self._validate_llm_providers()
        await self._validate_mcp_servers()
        
        # Update summary
        self._update_summary()
        
        return self.validation_results
    
    async def validate_component(self, component: str) -> Dict[str, Any]:
        """
        Validate a specific component.
        
        Args:
            component: Component to validate (neo4j, llm, mcp)
            
        Returns:
            Validation results for the component
        """
        if component == 'neo4j':
            await self._validate_neo4j()
        elif component == 'llm':
            await self._validate_llm_providers()
        elif component == 'mcp':
            await self._validate_mcp_servers()
        else:
            raise ValueError(f"Unknown component: {component}")
        
        self._update_summary()
        return self.validation_results
    
    def _validate_environment_vars(self):
        """Validate required environment variables."""
        print("📋 Validating environment variables...")
        
        var_results = self.env_config.validate_required_vars()
        
        self.validation_results['components']['environment_vars'] = {
            'status': 'passed' if var_results['valid'] else 'failed',
            'details': var_results,
            'timestamp': datetime.now().isoformat()
        }
        
        if var_results['valid']:
            print("   ✅ All required environment variables are set")
        else:
            print("   ❌ Missing required environment variables")
            for component, missing in var_results['missing'].items():
                print(f"      - {component}: {', '.join(missing)}")
        
        if var_results.get('warnings'):
            for warning in var_results['warnings']:
                print(f"   ⚠️  {warning}")
    
    async def _validate_neo4j(self):
        """Validate Neo4j connectivity and configuration."""
        print("\n🗄️  Validating Neo4j connection...")
        
        result = {
            'status': 'unknown',
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if Neo4j vars are configured
        neo4j_config = self.env_config.get_neo4j_config()
        if not all(neo4j_config.values()):
            result['status'] = 'skipped'
            result['details']['message'] = 'Neo4j not configured'
            print("   ⏭️  Neo4j validation skipped (not configured)")
        elif not neo4j_available:
            result['status'] = 'failed'
            result['details']['error'] = 'langchain_neo4j not installed'
            print("   ❌ Neo4j validation failed: langchain_neo4j package not installed")
        else:
            try:
                # Test connection
                graph = Neo4jGraph(
                    url=neo4j_config['uri'],
                    username=neo4j_config['username'],
                    password=neo4j_config['password'],
                    enhanced_schema=True,
                    refresh_schema=False
                )
                
                # Run a simple query to test connectivity
                test_result = graph.query("RETURN 1 as test")
                if test_result and test_result[0]['test'] == 1:
                    result['status'] = 'passed'
                    print(f"   ✅ Connected to Neo4j at {neo4j_config['uri']}")
                    
                    # Get additional info
                    try:
                        # Get database version
                        version_result = graph.query(
                            "CALL dbms.components() YIELD name, versions "
                            "WHERE name = 'Neo4j Kernel' "
                            "RETURN versions[0] as version"
                        )
                        if version_result:
                            result['details']['version'] = version_result[0]['version']
                            print(f"   📌 Neo4j version: {version_result[0]['version']}")
                        
                        # Check for APOC
                        apoc_result = graph.query(
                            "CALL dbms.procedures() YIELD name "
                            "WHERE name STARTS WITH 'apoc.' "
                            "RETURN count(*) as apoc_count"
                        )
                        if apoc_result and apoc_result[0]['apoc_count'] > 0:
                            result['details']['apoc_installed'] = True
                            print(f"   ✅ APOC plugin installed ({apoc_result[0]['apoc_count']} procedures)")
                        else:
                            result['details']['apoc_installed'] = False
                            result['details']['warning'] = 'APOC plugin not found'
                            print("   ⚠️  APOC plugin not found (required for schema export)")
                        
                        # Get node and relationship counts
                        stats = graph.query(
                            "MATCH (n) WITH count(n) as nodes "
                            "MATCH ()-[r]->() RETURN nodes, count(r) as relationships"
                        )
                        if stats:
                            result['details']['statistics'] = {
                                'nodes': stats[0]['nodes'],
                                'relationships': stats[0]['relationships']
                            }
                            print(f"   📊 Database stats: {stats[0]['nodes']:,} nodes, {stats[0]['relationships']:,} relationships")
                    
                    except Exception as e:
                        result['details']['info_error'] = str(e)
                        
            except Exception as e:
                result['status'] = 'failed'
                result['details']['error'] = str(e)
                print(f"   ❌ Neo4j connection failed: {str(e)}")
        
        self.validation_results['components']['neo4j'] = result
    
    async def _validate_llm_providers(self):
        """Validate LLM provider connectivity."""
        print("\n🤖 Validating LLM providers...")
        
        providers_to_check = []
        
        # Check which providers have configuration
        for provider in ['openai', 'anthropic', 'google', 'ollama']:
            config = self.env_config.get_llm_config(provider)
            if provider == 'ollama' or (config.get('api_key')):
                providers_to_check.append(provider)
        
        if not providers_to_check:
            print("   ⏭️  No LLM providers configured")
            self.validation_results['components']['llm_providers'] = {
                'status': 'skipped',
                'details': {'message': 'No LLM providers configured'},
                'timestamp': datetime.now().isoformat()
            }
            return
        
        provider_results = {}
        
        for provider in providers_to_check:
            print(f"\n   🔧 Checking {provider}...")
            result = await self._validate_llm_provider(provider)
            provider_results[provider] = result
            
            if result['status'] == 'passed':
                print(f"   ✅ {provider} is working correctly")
                if 'models' in result['details']:
                    models = result['details']['models']
                    if models:
                        print(f"      Available models: {', '.join(models[:5])}")
                        if len(models) > 5:
                            print(f"      ... and {len(models) - 5} more")
                    else:
                        print(f"      No models installed")
                if 'warning' in result['details']:
                    print(f"      ⚠️  {result['details']['warning']}")
                if provider == 'ollama' and 'base_url' in result['details']:
                    print(f"      Service URL: {result['details']['base_url']}")
            elif result['status'] == 'failed':
                print(f"   ❌ {provider} validation failed: {result['details'].get('error', 'Unknown error')}")
                if provider == 'ollama' and 'base_url' in result['details']:
                    print(f"      Attempted URL: {result['details']['base_url']}")
        
        # Aggregate results
        all_passed = all(r['status'] == 'passed' for r in provider_results.values())
        any_failed = any(r['status'] == 'failed' for r in provider_results.values())
        
        self.validation_results['components']['llm_providers'] = {
            'status': 'passed' if all_passed else ('failed' if any_failed else 'partial'),
            'providers': provider_results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _validate_llm_provider(self, provider: str) -> Dict[str, Any]:
        """Validate a specific LLM provider."""
        result = {
            'status': 'unknown',
            'provider': provider,
            'details': {}
        }
        
        try:
            if provider == 'openai':
                if not openai:
                    raise ImportError("openai package not installed")
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                
                client = openai.OpenAI(api_key=api_key)
                # List models to verify API key
                models = client.models.list()
                model_ids = [m.id for m in models.data if 'gpt' in m.id]
                
                result['status'] = 'passed'
                result['details']['models'] = model_ids
                
            elif provider == 'anthropic':
                if not anthropic:
                    raise ImportError("anthropic package not installed")
                
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                
                client = anthropic.Anthropic(api_key=api_key)
                # Simple test - create a message with minimal tokens
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                
                result['status'] = 'passed'
                result['details']['models'] = [
                    'claude-2.0', 
                    'claude-2.1',
                    'claude-sonnet-4-20250514'
                ]
                
            elif provider == 'google':
                if not genai:
                    raise ImportError("google-generativeai package not installed")
                
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                
                genai.configure(api_key=api_key)
                # List available models
                models = genai.list_models()
                model_names = [m.name.split('/')[-1] for m in models if 'generateContent' in m.supported_generation_methods]
                
                result['status'] = 'passed'
                result['details']['models'] = model_names
                
            elif provider == 'ollama':
                if not ollama:
                    raise ImportError("ollama package not installed")
                
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                
                # Test basic connectivity first with a simple HTTP check
                try:
                    # Try a simple HTTP request to check if service is running
                    health_url = f"{base_url.rstrip('/')}/api/version"
                    response = requests.get(health_url, timeout=5)
                    if response.status_code != 200:
                        raise Exception(f"Ollama service not responding properly (HTTP {response.status_code})")
                except requests.exceptions.ConnectionError:
                    raise Exception(f"Ollama service is not running at {base_url}. Start it with 'ollama serve'")
                except requests.exceptions.Timeout:
                    raise Exception(f"Timeout connecting to Ollama service at {base_url}")
                except Exception as http_error:
                    if "connection" in str(http_error).lower():
                        raise Exception(f"Cannot reach Ollama service at {base_url}")
                    # Continue with ollama client if HTTP check has issues
                
                # Try to connect to Ollama with proper error handling
                try:
                    client = ollama.Client(host=base_url)
                    
                    # Try to list models - this tests both connectivity and service availability
                    try:
                        models_response = client.list()
                        
                        if models_response and hasattr(models_response, 'models'):
                            # Handle new ollama client response format (objects)
                            model_names = [m.model for m in models_response.models]
                        elif models_response and 'models' in models_response:
                            # Handle dict response format
                            model_names = [m['name'] for m in models_response['models']]
                        elif hasattr(models_response, '__dict__') and 'models' in models_response.__dict__:
                            # Handle object with models attribute
                            models_list = getattr(models_response, 'models', [])
                            model_names = [getattr(m, 'model', str(m)) for m in models_list]
                        else:
                            # Fallback: empty list if we can't parse
                            model_names = []
                        
                        if model_names:
                            result['status'] = 'passed'
                            result['details']['models'] = model_names
                            result['details']['base_url'] = base_url
                            result['details']['model_count'] = len(model_names)
                        else:
                            # No models found but service is reachable
                            result['status'] = 'passed'
                            result['details']['models'] = []
                            result['details']['base_url'] = base_url
                            result['details']['warning'] = "Ollama service is running but no models are installed. Install models with 'ollama pull <model>'"
                            
                    except Exception as model_error:
                        # Ollama service might be running but having issues
                        error_str = str(model_error).lower()
                        if "connection" in error_str or "refused" in error_str:
                            raise Exception(f"Cannot connect to Ollama service at {base_url}. Is Ollama running?")
                        else:
                            raise Exception(f"Ollama model list error: {str(model_error)}")
                            
                except Exception as conn_error:
                    # Re-raise with more helpful context
                    error_msg = str(conn_error).lower()
                    if "connection refused" in error_msg or "connection error" in error_msg:
                        raise Exception(f"Ollama service is not running at {base_url}. Start it with 'ollama serve'")
                    elif "timeout" in error_msg:
                        raise Exception(f"Timeout connecting to Ollama service at {base_url}")
                    elif "connection" in error_msg:
                        raise Exception(f"Cannot reach Ollama service at {base_url}. Check the URL and network connectivity")
                    else:
                        raise Exception(f"Ollama validation error: {str(conn_error)}")
                    
        except Exception as e:
            result['status'] = 'failed'
            result['details']['error'] = str(e)
        
        return result
    
    async def _validate_mcp_servers(self):
        """Validate MCP server configuration."""
        print("\n🔌 Validating MCP servers...")
        
        # Check if mcp_servers.json exists
        mcp_config_path = Path(__file__).parent / 'mcp_servers.json'
        if not mcp_config_path.exists():
            self.validation_results['components']['mcp_servers'] = {
                'status': 'skipped',
                'details': {'message': 'mcp_servers.json not found'},
                'timestamp': datetime.now().isoformat()
            }
            print("   ⏭️  MCP validation skipped (mcp_servers.json not found)")
            return
        
        try:
            with open(mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
            
            servers = mcp_config.get('servers', {})
            server_results = {}
            
            for server_name, server_config in servers.items():
                if not server_config.get('enabled', False):
                    server_results[server_name] = {
                        'status': 'disabled',
                        'details': {'message': 'Server is disabled in configuration'}
                    }
                    continue
                
                print(f"\n   🔧 Checking {server_name}...")
                
                # Validate environment variables
                env_vars = server_config.get('environment', {})
                required_vars = env_vars.get('required', [])
                missing_vars = [var for var in required_vars if not os.getenv(var)]
                
                if missing_vars:
                    server_results[server_name] = {
                        'status': 'failed',
                        'details': {
                            'error': f"Missing required environment variables: {', '.join(missing_vars)}"
                        }
                    }
                    print(f"   ❌ {server_name}: Missing environment variables: {', '.join(missing_vars)}")
                else:
                    server_results[server_name] = {
                        'status': 'configured',
                        'details': {
                            'command': server_config.get('command', {}).get('executable', 'Unknown'),
                            'description': server_config.get('description', 'No description')
                        }
                    }
                    print(f"   ✅ {server_name}: Configuration valid")
            
            # Aggregate results
            all_ok = all(r['status'] in ['configured', 'disabled'] for r in server_results.values())
            
            self.validation_results['components']['mcp_servers'] = {
                'status': 'passed' if all_ok else 'failed',
                'servers': server_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.validation_results['components']['mcp_servers'] = {
                'status': 'failed',
                'details': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
            print(f"   ❌ MCP validation failed: {str(e)}")
    
    def _update_summary(self):
        """Update the summary statistics."""
        total = passed = failed = warnings = 0
        
        for component, result in self.validation_results['components'].items():
            total += 1
            status = result.get('status', 'unknown')
            
            if status == 'passed':
                passed += 1
            elif status == 'failed':
                failed += 1
            elif status in ['partial', 'warning']:
                warnings += 1
            elif status == 'skipped':
                total -= 1  # Don't count skipped in total
        
        self.validation_results['summary'] = {
            'total_checks': total,
            'passed': passed,
            'failed': failed,
            'warnings': warnings
        }
    
    def display_validation_report(self):
        """Display a formatted validation report."""
        print("\n" + "="*80)
        print("📊 ENVIRONMENT VALIDATION REPORT")
        print("="*80)
        
        print(f"\n🌍 Environment: {self.validation_results['environment']}")
        print(f"📁 Config File: {self.validation_results['loaded_file']}")
        print(f"🕐 Timestamp: {self.validation_results['timestamp']}")
        
        summary = self.validation_results['summary']
        print(f"\n📈 Summary:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⚠️  Warnings: {summary['warnings']}")
        
        print("\n🔍 Component Details:")
        
        for component, result in self.validation_results['components'].items():
            status = result.get('status', 'unknown')
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'warning': '⚠️',
                'partial': '⚠️',
                'skipped': '⏭️',
                'unknown': '❓'
            }.get(status, '❓')
            
            print(f"\n{status_icon} {component.replace('_', ' ').title()}:")
            
            if component == 'environment_vars' and 'details' in result:
                details = result['details']
                if not details['valid']:
                    for comp, missing in details.get('missing', {}).items():
                        print(f"   - {comp}: Missing {', '.join(missing)}")
                        
            elif component == 'neo4j' and 'details' in result:
                details = result.get('details', {})
                if 'version' in details:
                    print(f"   - Version: {details['version']}")
                if 'apoc_installed' in details:
                    print(f"   - APOC: {'Installed' if details['apoc_installed'] else 'Not installed'}")
                if 'statistics' in details:
                    stats = details['statistics']
                    print(f"   - Database: {stats['nodes']:,} nodes, {stats['relationships']:,} relationships")
                    
            elif component == 'llm_providers' and 'providers' in result:
                for provider, prov_result in result['providers'].items():
                    prov_status = prov_result.get('status', 'unknown')
                    prov_icon = '✅' if prov_status == 'passed' else '❌'
                    print(f"   {prov_icon} {provider}")
                    if prov_status == 'failed' and 'error' in prov_result.get('details', {}):
                        print(f"      Error: {prov_result['details']['error']}")
                        
            elif component == 'mcp_servers' and 'servers' in result:
                for server, serv_result in result['servers'].items():
                    serv_status = serv_result.get('status', 'unknown')
                    serv_icon = '✅' if serv_status == 'configured' else ('⏸️' if serv_status == 'disabled' else '❌')
                    print(f"   {serv_icon} {server}")
                    if 'error' in serv_result.get('details', {}):
                        print(f"      Error: {serv_result['details']['error']}")
        
        # Overall status
        print("\n" + "="*80)
        if summary['failed'] > 0:
            print("❌ VALIDATION FAILED - Please fix the issues above")
        elif summary['warnings'] > 0:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
        else:
            print("✅ VALIDATION PASSED - Environment is properly configured")
        print("="*80 + "\n")
    
    def save_validation_report(self, output_file: Optional[str] = None):
        """
        Save validation report to a JSON file.
        
        Args:
            output_file: Path to save the report (default: validation_report_{timestamp}.json)
        """
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"validation_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"📝 Validation report saved to: {output_file}")


async def main():
    """Test the environment validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Atlas environment configuration')
    parser.add_argument('--env', help='Environment name')
    parser.add_argument('--env-file', help='Custom environment file path')
    parser.add_argument('--component', choices=['neo4j', 'llm', 'mcp', 'all'], 
                       default='all', help='Component to validate')
    parser.add_argument('--save-report', help='Save report to specified file')
    args = parser.parse_args()
    
    # Load environment
    env_config = EnvironmentConfig(env=args.env, env_file=args.env_file)
    result = env_config.load_environment()
    
    if not result['success']:
        print(f"❌ Failed to load environment: {result['error']}")
        sys.exit(1)
    
    # Validate
    validator = EnvironmentValidator(env_config)
    
    if args.component == 'all':
        await validator.validate_all()
    else:
        await validator.validate_component(args.component)
    
    # Display report
    validator.display_validation_report()
    
    # Save report if requested
    if args.save_report:
        validator.save_validation_report(args.save_report)
    
    # Exit with appropriate code
    if validator.validation_results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    asyncio.run(main())