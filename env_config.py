#!/usr/bin/env python3
"""
Environment Configuration Module for Atlas

Handles loading and managing multiple environment configurations,
including support for different environments (dev, staging, prod)
and custom environment files.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Manages environment configuration and loading for Atlas."""
    
    # Default environment file mappings for common environment names
    # Note: Any environment name not in this mapping will look for .env.{env}
    # For example: --env client1 will look for .env.client1
    ENV_FILE_MAPPING = {
        'development': '.env.dev',
        'dev': '.env.dev',
        'staging': '.env.staging',
        'stage': '.env.staging',
        'production': '.env.prod',
        'prod': '.env.prod'
    }
    
    # Required variables for each component
    REQUIRED_VARS = {
        'neo4j': ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD'],
        'openai': ['OPENAI_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY'],
        'google': ['GEMINI_API_KEY'],
        'ollama': [],  # Ollama doesn't require API keys
    }
    
    # Optional but recommended variables
    OPTIONAL_VARS = {
        'defaults': ['DEFAULT_LLM_PROVIDER', 'DEFAULT_LLM_MODEL', 'DEFAULT_ANALYSIS_CONTEXT'],
        'langsmith': ['LANGSMITH_API_KEY', 'LANGSMITH_PROJECT'],
        'atlas': ['ATLAS_LOG_LEVEL', 'ATLAS_OUTPUT_DIR'],
        'ollama': ['OLLAMA_BASE_URL'],
        'mcp': ['LOCAL_NEO4J_MCP_SERVER_PATH']
    }
    
    def __init__(self, env: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize environment configuration.
        
        Args:
            env: Environment name (development, staging, production)
            env_file: Path to a custom environment file
        """
        self.base_dir = Path(__file__).parent
        self.env = env or os.getenv('ATLAS_ENV', 'development')
        self.custom_env_file = env_file
        self.loaded_env_file = None
        self.validation_results = {}
        
    def load_environment(self) -> Dict[str, Any]:
        """
        Load environment variables from the appropriate .env file.
        
        Returns:
            Dictionary with load status and details
        """
        # Clear any previous validation results
        self.validation_results = {}
        
        # Determine which env file to load
        if self.custom_env_file:
            # Use explicit env file path
            env_path = Path(self.custom_env_file)
            if not env_path.is_absolute():
                env_path = self.base_dir / env_path
        else:
            # Use environment-based file
            env_filename = self.ENV_FILE_MAPPING.get(self.env.lower())
            if env_filename:
                env_path = self.base_dir / env_filename
            else:
                # For any environment name, try .env.{env}
                env_path = self.base_dir / f'.env.{self.env.lower()}'
        
        # Try to load the environment file
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            self.loaded_env_file = str(env_path)
            logger.info(f"Loaded environment from: {env_path}")
            return {
                'success': True, 
                'loaded_file': str(env_path),
                'environment': self.env
            }
        else:
            # Fall back to default .env if it exists
            default_path = self.base_dir / '.env'
            if default_path.exists():
                load_dotenv(dotenv_path=default_path, override=True)
                self.loaded_env_file = str(default_path)
                logger.warning(f"Environment file {env_path} not found, using default .env")
                return {
                    'success': True, 
                    'loaded_file': str(default_path), 
                    'environment': 'default',
                    'fallback': True,
                    'attempted_file': str(env_path)
                }
            else:
                logger.error(f"No environment file found at {env_path} or {default_path}")
                return {
                    'success': False, 
                    'error': f'No environment file found',
                    'attempted_files': [str(env_path), str(default_path)]
                }
    
    def validate_required_vars(self, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate that required environment variables are set.
        
        Args:
            components: List of components to validate (e.g., ['neo4j', 'openai'])
                       If None, validates based on what's configured
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'missing': {},
            'warnings': [],
            'components': {}
        }
        
        # If no components specified, check all that have some configuration
        if not components:
            components = []
            if any(os.getenv(var) for var in self.REQUIRED_VARS['neo4j']):
                components.append('neo4j')
            for provider in ['openai', 'anthropic', 'google', 'ollama']:
                if provider in self.REQUIRED_VARS and any(os.getenv(var) for var in self.REQUIRED_VARS.get(provider, [])):
                    components.append(provider)
        
        # Validate each component
        for component in components:
            if component not in self.REQUIRED_VARS:
                results['warnings'].append(f"Unknown component: {component}")
                continue
                
            required = self.REQUIRED_VARS[component]
            missing = []
            
            for var in required:
                if not os.getenv(var):
                    missing.append(var)
            
            results['components'][component] = {
                'valid': len(missing) == 0,
                'missing': missing,
                'required': required
            }
            
            if missing:
                results['missing'][component] = missing
                results['valid'] = False
        
        # Check optional variables and add warnings
        for category, vars in self.OPTIONAL_VARS.items():
            for var in vars:
                if not os.getenv(var) and category in ['defaults']:
                    results['warnings'].append(
                        f"Optional variable {var} not set. "
                        f"You may need to specify --llm-provider and --model on each command."
                    )
        
        self.validation_results = results
        return results
    
    def get_default_llm_config(self) -> Dict[str, Optional[str]]:
        """
        Get default LLM provider and model from environment.
        
        Returns:
            Dictionary with 'provider' and 'model' keys
        """
        return {
            'provider': os.getenv('DEFAULT_LLM_PROVIDER'),
            'model': os.getenv('DEFAULT_LLM_MODEL')
        }
    
    def get_default_analysis_context(self) -> str:
        """
        Get default analysis context from environment.
        
        Returns:
            Analysis context string (defaults to 'legacy' if not set)
        """
        return os.getenv('DEFAULT_ANALYSIS_CONTEXT', 'legacy')
    
    def get_neo4j_config(self) -> Dict[str, Optional[str]]:
        """
        Get Neo4j configuration from environment.
        
        Returns:
            Dictionary with Neo4j connection settings
        """
        return {
            'uri': os.getenv('NEO4J_URI'),
            'username': os.getenv('NEO4J_USERNAME'),
            'password': os.getenv('NEO4J_PASSWORD')
        }
    
    def get_llm_config(self, provider: str) -> Dict[str, Optional[str]]:
        """
        Get LLM provider configuration from environment.
        
        Args:
            provider: LLM provider name
            
        Returns:
            Dictionary with provider-specific configuration
        """
        config = {}
        
        if provider == 'openai':
            config['api_key'] = os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            config['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        elif provider == 'google':
            config['api_key'] = os.getenv('GEMINI_API_KEY')
        elif provider == 'ollama':
            config['base_url'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
        return config
    
    def display_config_summary(self):
        """Display a summary of the loaded configuration."""
        print("\n" + "="*60)
        print("🔧 ENVIRONMENT CONFIGURATION SUMMARY")
        print("="*60)
        
        if self.loaded_env_file:
            print(f"\n📁 Loaded from: {self.loaded_env_file}")
            print(f"🌍 Environment: {self.env}")
        else:
            print("\n❌ No environment file loaded")
        
        # Display default LLM settings
        defaults = self.get_default_llm_config()
        analysis_context = self.get_default_analysis_context()
        print(f"\n🤖 Default Settings:")
        print(f"   LLM Provider: {defaults['provider'] or 'Not set'}")
        print(f"   LLM Model: {defaults['model'] or 'Not set'}")
        print(f"   Analysis Context: {analysis_context}")
        
        # Display Neo4j settings (masked)
        neo4j = self.get_neo4j_config()
        print(f"\n🗄️  Neo4j Configuration:")
        print(f"   URI: {neo4j['uri'] or 'Not set'}")
        print(f"   Username: {neo4j['username'] or 'Not set'}")
        print(f"   Password: {'***' if neo4j['password'] else 'Not set'}")
        
        # Display LLM provider status
        print(f"\n🔑 LLM Providers:")
        for provider in ['openai', 'anthropic', 'google', 'ollama']:
            config = self.get_llm_config(provider)
            if provider == 'ollama':
                status = f"Base URL: {config.get('base_url', 'Not set')}"
            else:
                api_key = config.get('api_key')
                status = '✅ Configured' if api_key else '❌ Not configured'
            print(f"   {provider.capitalize()}: {status}")
        
        # Display validation results if available
        if self.validation_results:
            print(f"\n✓ Validation Results:")
            if self.validation_results.get('valid'):
                print("   ✅ All required variables are set")
            else:
                print("   ❌ Missing required variables:")
                for component, missing in self.validation_results.get('missing', {}).items():
                    print(f"      {component}: {', '.join(missing)}")
            
            if self.validation_results.get('warnings'):
                print("\n⚠️  Warnings:")
                for warning in self.validation_results['warnings']:
                    print(f"   - {warning}")
        
        print("="*60 + "\n")


def main():
    """Test the environment configuration module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test environment configuration')
    parser.add_argument('--env', help='Environment name')
    parser.add_argument('--env-file', help='Custom environment file path')
    args = parser.parse_args()
    
    config = EnvironmentConfig(env=args.env, env_file=args.env_file)
    result = config.load_environment()
    
    if result['success']:
        print(f"✅ Successfully loaded environment from: {result['loaded_file']}")
        if result.get('fallback'):
            print(f"   (Fallback from attempted: {result.get('attempted_file')})")
    else:
        print(f"❌ Failed to load environment: {result['error']}")
        print(f"   Attempted files: {', '.join(result['attempted_files'])}")
    
    # Validate configuration
    validation = config.validate_required_vars()
    config.display_config_summary()


if __name__ == '__main__':
    main()