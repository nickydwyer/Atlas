# MCP Server Configuration Guide

This guide explains how to configure Model Context Protocol (MCP) servers for Atlas without requiring code changes.

## Overview

Atlas now supports external MCP server configuration through the `mcp_servers.json` file. This allows you to:

- Add new MCP servers without modifying code
- Enable/disable servers dynamically
- Configure environment variables and command arguments
- Set up fallback mechanisms for better reliability

## Configuration File Structure

### Location
The configuration file should be placed at:
- `./mcp_servers.json` (recommended, in the Atlas root directory)
- `./config/mcp_servers.json`
- `./.config/mcp_servers.json`

### Basic Structure
```json
{
  "servers": {
    "server-name": {
      "enabled": true,
      "description": "Server description",
      "command": {
        "executable": "command",
        "args": ["arg1", "arg2"]
      },
      "args": ["--transport", "stdio"],
      "transport": "stdio",
      "environment": {
        "required": ["REQUIRED_VAR"],
        "optional": ["OPTIONAL_VAR"]
      }
    }
  },
  "global_settings": {
    "connection_timeout": 30,
    "max_retries": 3,
    "retry_delay": 1.0,
    "environment_variable_interpolation": true,
    "fallback_behavior": "warn"
  }
}
```

## Server Configuration Properties

### Required Properties

- **`enabled`** (boolean): Whether the server should be loaded
- **`description`** (string): Human-readable description of the server
- **`command.executable`** (string): The executable command or path
- **`args`** (array): Command-line arguments for the server
- **`transport`** (string): Transport type (typically "stdio")

### Optional Properties

- **`command.args`** (array): Base arguments for the executable
- **`command.fallback_executable`** (string): Alternative executable if primary fails
- **`command.fallback_args`** (array): Arguments for fallback executable
- **`environment.required`** (array): Required environment variables
- **`environment.optional`** (array): Optional environment variables

## Environment Variable Interpolation

Atlas supports environment variable interpolation in configuration values:

### Syntax
- **`${VAR}`**: Required variable (error if not found)
- **`${VAR:default}`**: Variable with default value

### Examples
```json
{
  "args": [
    "--db-url", "${NEO4J_URI:neo4j://localhost:7687}",
    "--username", "${NEO4J_USERNAME:neo4j}",
    "--password", "${NEO4J_PASSWORD}"
  ]
}
```

## Predefined Server Configurations

The default `mcp_servers.json` includes several example servers:

### 1. Local Neo4j (enabled by default)
```json
"local-neo4j": {
  "enabled": true,
  "description": "Neo4j graph database operations via Cypher queries",
  "command": {
    "executable": "${LOCAL_NEO4J_MCP_SERVER_PATH}",
    "fallback_executable": "uvx",
    "fallback_args": ["mcp-neo4j-cypher"]
  },
  "args": [
    "--transport", "stdio",
    "--db-url", "${NEO4J_URI:neo4j://localhost:7687}",
    "--username", "${NEO4J_USERNAME:neo4j}",
    "--password", "${NEO4J_PASSWORD}"
  ]
}
```

### 2. Application Modernization Neo4j (disabled by default)
For specialized Neo4j instances used in application modernization projects.

### 3. Filesystem Operations (disabled by default)
```json
"filesystem": {
  "enabled": false,
  "description": "File system operations for reading and writing files",
  "command": {
    "executable": "uvx",
    "args": ["mcp-filesystem"]
  },
  "args": [
    "--transport", "stdio",
    "--allowed-paths", "${MCP_FILESYSTEM_ALLOWED_PATHS:./}"
  ]
}
```

### 4. Web Search (disabled by default)
```json
"web-search": {
  "enabled": false,
  "description": "Web search capabilities using various search engines",
  "command": {
    "executable": "uvx",
    "args": ["mcp-web-search"]
  },
  "args": [
    "--transport", "stdio",
    "--api-key", "${WEB_SEARCH_API_KEY}"
  ]
}
```

### 5. GitHub Integration (disabled by default)
```json
"github": {
  "enabled": false,
  "description": "GitHub repository and issue management",
  "command": {
    "executable": "uvx",
    "args": ["mcp-github"]
  },
  "args": [
    "--transport", "stdio",
    "--token", "${GITHUB_TOKEN}"
  ]
}
```

## Adding a New MCP Server

### Step 1: Add Server Configuration
Edit `mcp_servers.json` and add your server:

```json
{
  "servers": {
    "my-custom-server": {
      "enabled": true,
      "description": "My custom MCP server for specific tasks",
      "command": {
        "executable": "uvx",
        "args": ["my-mcp-package"]
      },
      "args": [
        "--transport", "stdio",
        "--config-file", "${MY_SERVER_CONFIG:./default-config.json}",
        "--api-key", "${MY_SERVER_API_KEY}"
      ],
      "transport": "stdio",
      "environment": {
        "required": ["MY_SERVER_API_KEY"],
        "optional": ["MY_SERVER_CONFIG"]
      }
    }
  }
}
```

### Step 2: Set Environment Variables
Add required environment variables to your `.env` file:

```bash
# My custom server configuration
MY_SERVER_API_KEY=your_api_key_here
MY_SERVER_CONFIG=/path/to/config.json
```

### Step 3: Test Configuration
You can validate your configuration by running:

```python
python -c "from mcp_config import load_mcp_config; config = load_mcp_config(); print('Enabled servers:', list(config.get_enabled_servers().keys()))"
```

### Step 4: Restart Chat Application
Run the Atlas chat command to load the new server:

```bash
python atlas.py chat --llm-provider openai --model gpt-4o
```

## Global Settings

### Connection Management
- **`connection_timeout`**: Timeout for server connections (default: 30 seconds)
- **`max_retries`**: Maximum connection retry attempts (default: 3)
- **`retry_delay`**: Delay between retry attempts (default: 1.0 seconds)

### Behavior Control
- **`environment_variable_interpolation`**: Enable/disable variable substitution (default: true)
- **`fallback_behavior`**: How to handle configuration errors ("warn" or "error", default: "warn")

## Troubleshooting

### Common Issues

1. **Server Not Loading**
   - Check that `enabled: true` is set
   - Verify all required environment variables are set
   - Check Atlas logs for error messages

2. **Environment Variable Not Found**
   - Ensure variable is defined in `.env` file
   - Check variable name spelling in configuration
   - Verify `.env` file is in the correct location

3. **Executable Not Found**
   - Verify the executable is installed and in PATH
   - Check if fallback executable is properly configured
   - Ensure proper permissions on executable files

### Validation Commands

```bash
# Check configuration syntax
python -m json.tool mcp_servers.json

# Validate environment variables
python mcp_config.py

# Test server connections
python atlas.py chat --llm-provider openai --model gpt-4o-mini
```

## Migration from Hardcoded Configuration

If you're upgrading from a version with hardcoded MCP servers:

1. **Backup**: Save your current `.env` file
2. **Configuration**: The system will automatically fall back to hardcoded Neo4j if `mcp_servers.json` is missing
3. **Migration**: Enable additional servers by setting `enabled: true` in the configuration
4. **Testing**: Verify all expected tools are available in the chat interface

## Security Considerations

- **Environment Variables**: Never commit API keys or passwords to version control
- **File Permissions**: Ensure configuration files have appropriate read permissions
- **Path Restrictions**: Use `allowed-paths` restrictions for filesystem servers
- **API Keys**: Regularly rotate API keys for external services

## Examples and Templates

### Local Development Server
```json
"local-dev": {
  "enabled": true,
  "description": "Local development tools",
  "command": {
    "executable": "python",
    "args": ["-m", "my_local_mcp_server"]
  },
  "args": ["--transport", "stdio", "--debug"],
  "transport": "stdio"
}
```

### Production API Server
```json
"production-api": {
  "enabled": false,
  "description": "Production API integration",
  "command": {
    "executable": "uvx",
    "args": ["production-mcp-client"]
  },
  "args": [
    "--transport", "stdio",
    "--endpoint", "${PROD_API_ENDPOINT}",
    "--token", "${PROD_API_TOKEN}",
    "--timeout", "60"
  ],
  "environment": {
    "required": ["PROD_API_ENDPOINT", "PROD_API_TOKEN"]
  }
}
```

This external configuration system provides flexibility while maintaining backward compatibility and robust error handling.