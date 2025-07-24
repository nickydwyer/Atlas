#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an individual MCP server"""
    name: str
    enabled: bool
    description: str
    executable: str
    args: List[str]
    transport: str
    environment: Dict[str, List[str]]
    fallback_executable: Optional[str] = None
    fallback_args: Optional[List[str]] = None

@dataclass
class MCPGlobalSettings:
    """Global settings for MCP server management"""
    connection_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    environment_variable_interpolation: bool = True
    fallback_behavior: str = "warn"

class MCPConfigLoader:
    """Loads and manages MCP server configurations from external files"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the MCP configuration loader
        
        Args:
            config_path: Path to the MCP configuration file. If None, uses default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config_data: Dict[str, Any] = {}
        self.servers: Dict[str, MCPServerConfig] = {}
        self.global_settings: MCPGlobalSettings = MCPGlobalSettings()
        
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the configuration file path"""
        if config_path:
            return Path(config_path)
        
        # Try default locations
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "mcp_servers.json",
            current_dir / "config" / "mcp_servers.json",
            current_dir / ".config" / "mcp_servers.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Return the default path even if it doesn't exist
        return current_dir / "mcp_servers.json"
    
    def load_config(self) -> None:
        """Load configuration from the JSON file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"MCP configuration file not found at {self.config_path}")
                return
                
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
                
            self._parse_global_settings()
            self._parse_servers()
            
            logger.info(f"Loaded MCP configuration from {self.config_path}")
            logger.info(f"Found {len(self.servers)} server configurations, {len(self.get_enabled_servers())} enabled")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {e}")
            raise
    
    def _parse_global_settings(self) -> None:
        """Parse global settings from configuration"""
        global_config = self.config_data.get("global_settings", {})
        self.global_settings = MCPGlobalSettings(
            connection_timeout=global_config.get("connection_timeout", 30),
            max_retries=global_config.get("max_retries", 3),
            retry_delay=global_config.get("retry_delay", 1.0),
            environment_variable_interpolation=global_config.get("environment_variable_interpolation", True),
            fallback_behavior=global_config.get("fallback_behavior", "warn")
        )
    
    def _parse_servers(self) -> None:
        """Parse server configurations"""
        servers_config = self.config_data.get("servers", {})
        
        for server_name, server_config in servers_config.items():
            try:
                parsed_server = self._parse_single_server(server_name, server_config)
                self.servers[server_name] = parsed_server
            except Exception as e:
                logger.error(f"Error parsing server config for '{server_name}': {e}")
                if self.global_settings.fallback_behavior == "error":
                    raise
    
    def _parse_single_server(self, server_name: str, config: Dict[str, Any]) -> MCPServerConfig:
        """Parse configuration for a single server"""
        command_config = config.get("command", {})
        
        # Handle executable and fallback
        executable = command_config.get("executable", "")
        fallback_executable = command_config.get("fallback_executable")
        fallback_args = command_config.get("fallback_args", [])
        
        # If executable is an environment variable, try to resolve it
        if self.global_settings.environment_variable_interpolation:
            try:
                executable = self._interpolate_variables(executable)
            except ValueError:
                # If interpolation fails, executable remains as-is (will trigger fallback)
                pass
            
        # If executable is empty or not found, use fallback
        if not executable or (executable.startswith("${") and executable.endswith("}")):
            if fallback_executable:
                executable = fallback_executable
                # Prepend fallback args to regular args
                base_args = command_config.get("args", [])
                config["args"] = fallback_args + base_args + config.get("args", [])
            else:
                raise ValueError(f"No valid executable found for server '{server_name}'")
        
        # Interpolate variables in arguments
        args = config.get("args", [])
        if self.global_settings.environment_variable_interpolation:
            args = [self._interpolate_variables(arg) for arg in args]
        
        return MCPServerConfig(
            name=server_name,
            enabled=config.get("enabled", False),
            description=config.get("description", f"MCP server: {server_name}"),
            executable=executable,
            args=args,
            transport=config.get("transport", "stdio"),
            environment=config.get("environment", {}),
            fallback_executable=fallback_executable,
            fallback_args=fallback_args
        )
    
    def _interpolate_variables(self, value: str) -> str:
        """
        Interpolate environment variables in a string
        
        Supports formats:
        - ${VAR} - Required variable (raises error if not found)
        - ${VAR:default} - Variable with default value
        """
        if not isinstance(value, str):
            return value
            
        def replace_var(match):
            var_spec = match.group(1)
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                var_value = os.getenv(var_spec)
                if var_value is None:
                    raise ValueError(f"Required environment variable '{var_spec}' not found")
                return var_value
        
        # Pattern matches ${VAR} or ${VAR:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, value)
    
    def get_enabled_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all enabled server configurations"""
        return {name: config for name, config in self.servers.items() if config.enabled}
    
    def get_server_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a specific server"""
        return self.servers.get(server_name)
    
    def validate_environment(self, server_config: MCPServerConfig) -> List[str]:
        """
        Validate that required environment variables are available
        
        Returns:
            List of missing required environment variables
        """
        missing_vars = []
        required_vars = server_config.environment.get("required", [])
        
        for var_name in required_vars:
            if not os.getenv(var_name):
                missing_vars.append(var_name)
                
        return missing_vars
    
    def validate_all_enabled_servers(self) -> Dict[str, List[str]]:
        """
        Validate environment for all enabled servers
        
        Returns:
            Dictionary mapping server names to lists of missing variables
        """
        validation_results = {}
        
        for server_name, server_config in self.get_enabled_servers().items():
            missing_vars = self.validate_environment(server_config)
            if missing_vars:
                validation_results[server_name] = missing_vars
                
        return validation_results
    
    def reload_config(self) -> None:
        """Reload configuration from file"""
        logger.info("Reloading MCP configuration")
        self.servers.clear()
        self.load_config()

def load_mcp_config(config_path: Optional[Union[str, Path]] = None) -> MCPConfigLoader:
    """
    Convenience function to load MCP configuration
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Loaded MCPConfigLoader instance
    """
    loader = MCPConfigLoader(config_path)
    loader.load_config()
    return loader

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        config_loader = load_mcp_config()
        
        print(f"Loaded {len(config_loader.servers)} server configurations")
        print(f"Enabled servers: {list(config_loader.get_enabled_servers().keys())}")
        
        # Validate environment
        validation_results = config_loader.validate_all_enabled_servers()
        if validation_results:
            print("\nMissing environment variables:")
            for server_name, missing_vars in validation_results.items():
                print(f"  {server_name}: {missing_vars}")
        else:
            print("\nAll enabled servers have required environment variables")
            
    except Exception as e:
        print(f"Error loading configuration: {e}")