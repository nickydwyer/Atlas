#!/usr/bin/env python3

import streamlit as st
import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Any, Optional, List
from collections.abc import Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import argparse

# LLM client imports
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

# MCP imports using FastMCP
import shutil
import json
import uuid
import logging
import time

# FastMCP V2 imports
try:
    from fastmcp import Client as FastMCPClient
    from fastmcp.client.transports import (
    SSETransport, 
    PythonStdioTransport, 
    StdioTransport,
    FastMCPTransport
)
    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCPClient = None
    PythonStdioTransport = None
    StdioTransport = None
    FASTMCP_AVAILABLE = False


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata')
        )


class MCPToolManager:
    """Manages MCP tools and connections using FastMCP"""
    
    def __init__(self):
        if not FASTMCP_AVAILABLE:
            logging.warning("FastMCP library not available. MCP functionality will be limited.")
        
        self.servers = {}  # server_name -> config
        self.transports = {}  # server_name -> transport config for reuse
        self.connection_status = {}  # server_name -> status
        self.error_messages = {}  # server_name -> error message
        self.available_tools = {}  # server_name -> list of tools
    
    def add_tool_server(self, name: str, command: str, args: list[str] | None = None, description: str = ""):
        """Add an MCP tool server configuration"""
        self.servers[name] = {
            'command': command,
            'args': args or [],
            'description': description
        }
        self.connection_status[name] = 'disconnected'
        self.error_messages[name] = None
        self.available_tools[name] = []
    
    def check_command_availability(self, command: str) -> bool:
        """Check if a command is available in the system"""
        return shutil.which(command) is not None
    
    def resolve_command_path(self, command: str) -> str:
        """Resolve command to full path"""
        full_path = shutil.which(command)
        return full_path if full_path else command
    
    async def connect_to_server(self, name: str) -> tuple[bool, str]:
        """Connect to an MCP server using FastMCP V2"""
        if not FASTMCP_AVAILABLE:
            return False, "FastMCP library not available"
            
        if name not in self.servers:
            return False, f"Server '{name}' not configured"
        
        config = self.servers[name]
        command = config['command']
        args = config['args']
        
        # Check if command exists
        if not self.check_command_availability(command):
            return False, f"Command '{command}' not found in PATH"
        
        # Resolve command to full path
        full_command_path = self.resolve_command_path(command)
        logging.info(f"Connecting to {name}: {full_command_path} {args}")
        
        try:
            self.connection_status[name] = 'connecting'
            
            # Store transport configuration for reuse
            transport_config = {
                'command_path': full_command_path,
                'args': args,
                'transport_type': 'python' if command.endswith('.py') or 'python' in command else 'stdio'
            }
            self.transports[name] = transport_config
            
            # Create transport and client for testing connection
            if transport_config['transport_type'] == 'python':
                transport = PythonStdioTransport(full_command_path, args)
            else:
                transport = StdioTransport(full_command_path, args)
            client = FastMCPClient(transport)
            
            # Test connection and get available tools
            async with client as session:
                tools_response = await session.list_tools()
                # FastMCP returns a list directly, not an object with .tools attribute
                if isinstance(tools_response, list):
                    tools = tools_response
                elif hasattr(tools_response, 'tools'):
                    tools = tools_response.tools
                else:
                    tools = []
                self.available_tools[name] = tools
                
                self.connection_status[name] = 'connected'
                self.error_messages[name] = None
                
                logging.info(f"Connected to {name} with {len(tools)} tools: {[t.name for t in tools]}")
                return True, f"Connected successfully with {len(tools)} tools"
            
        except Exception as e:
            self.connection_status[name] = 'failed'
            error_msg = str(e)
            self.error_messages[name] = error_msg
            logging.error(f"Failed to connect to {name}: {error_msg}")
            return False, error_msg
    
    async def connect_tools(self):
        """Connect to all configured MCP servers"""
        connection_results = []
        
        for name in self.servers:
            try:
                success, message = await self.connect_to_server(name)
                
                if success:
                    connection_results.append(f"✅ {name}: {message}")
                else:
                    connection_results.append(f"❌ {name}: {message}")
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.error_messages[name] = error_msg
                connection_results.append(f"❌ {name}: {error_msg}")
        
        return connection_results
    
    def get_available_tools(self) -> list[str]:
        """Get list of connected servers"""
        return [name for name, status in self.connection_status.items() if status == 'connected']
    
    def get_connection_summary(self) -> dict[str, dict]:
        """Get summary of all server connections"""
        summary = {}
        for name, config in self.servers.items():
            summary[name] = {
                'status': self.connection_status.get(name, 'unknown'),
                'description': config.get('description', ''),
                'error': self.error_messages.get(name),
                'command': config['command'],
                'tool_count': len(self.available_tools.get(name, []))
            }
        return summary
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> tuple[bool, Any]:
        """Call a tool on an MCP server using FastMCP V2"""
        if not FASTMCP_AVAILABLE:
            return False, "FastMCP library not available"
            
        if server_name not in self.servers:
            return False, f"Server '{server_name}' not configured"
        
        # Check if we have transport configuration for this server
        if server_name not in self.transports or self.connection_status.get(server_name) != 'connected':
            return False, f"Server '{server_name}' not connected. Please connect first."
        
        # Create a fresh client using stored transport configuration
        transport_config = self.transports[server_name]
        
        logging.info(f"Creating fresh client using stored transport config for: {server_name}.{tool_name}")
        logging.info(f"Tool arguments: {arguments}")
        
        try:
            # Create transport and client
            if transport_config['transport_type'] == 'python':
                transport = PythonStdioTransport(transport_config['command_path'], transport_config['args'])
            else:
                transport = StdioTransport(transport_config['command_path'], transport_config['args'])
            client = FastMCPClient(transport)
            
            # Call the tool using fresh client with async context manager
            async with client as session:
                logging.info(f"Session created with fresh client using stored config")
                logging.info(f"Calling tool '{tool_name}' with arguments: {arguments}")
                result = await session.call_tool(tool_name, arguments)
                logging.info(f"Tool call completed, result type: {type(result)}")
                
                # Handle different response formats
                if isinstance(result, list):
                    # Handle list of TextContent objects
                    content_parts = []
                    for item in result:
                        if hasattr(item, 'text'):
                            content_parts.append(item.text)
                        elif hasattr(item, 'content'):
                            content_parts.append(str(item.content))
                        else:
                            content_parts.append(str(item))
                    return True, '\n'.join(content_parts)
                elif hasattr(result, 'content'):
                    return True, result.content
                elif hasattr(result, 'result'):
                    return True, result.result
                elif hasattr(result, 'text'):
                    return True, result.text
                else:
                    return True, str(result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            logging.error(f"Full traceback: {error_details}")
            
            # If the error might be due to a stale connection, mark as disconnected
            if "Connection" in str(e) or "Transport" in str(e):
                logging.warning(f"Marking {server_name} as disconnected due to connection error")
                self.connection_status[server_name] = 'failed'
                self.error_messages[server_name] = str(e)
            
            return False, str(e)
    
    def get_all_available_tools(self) -> dict[str, list[dict]]:
        """Get all available tools from all connected MCP servers"""
        all_tools = {}
        
        # Get tools from connected servers
        for server_name in self.get_available_tools():
            tools = self.available_tools.get(server_name, [])
            if tools:
                # Convert FastMCP tool objects to dicts
                tool_dicts = []
                for tool in tools:
                    tool_dict = {
                        'name': tool.name,
                        'description': tool.description or f"Tool {tool.name} from {server_name}",
                        'inputSchema': tool.inputSchema or {
                            'type': 'object',
                            'properties': {},
                            'required': []
                        }
                    }
                    tool_dicts.append(tool_dict)
                
                all_tools[server_name] = tool_dicts
                logging.info(f"Found {len(tools)} tools for {server_name}: {[t.name for t in tools]}")
        
        return all_tools
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for name in list(self.transports.keys()):
            try:
                # Clear transport configuration and mark as disconnected
                if name in self.transports:
                    del self.transports[name]
                self.connection_status[name] = 'disconnected'
                logging.info(f"Disconnected from {name}")
            except Exception as e:
                logging.error(f"Error disconnecting from {name}: {e}")
                self.connection_status[name] = 'disconnected'


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
    
    async def generate_response(self, messages: list[ChatMessage], **kwargs) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None, mcp_manager: Optional['MCPToolManager'] = None):
        super().__init__(model, api_key or os.getenv('ANTHROPIC_API_KEY'))
        if not anthropic:
            raise ImportError("anthropic package not installed")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.mcp_manager = mcp_manager
    
    def _convert_mcp_tools_to_anthropic_format(self) -> list[dict]:
        """Convert MCP tools to Anthropic tool format"""
        anthropic_tools = []
        
        logging.info(f"_convert_mcp_tools_to_anthropic_format called. MCP manager: {self.mcp_manager is not None}")
        
        if not self.mcp_manager:
            logging.warning("No MCP manager available")
            return anthropic_tools
        
        all_tools = self.mcp_manager.get_all_available_tools()
        logging.info(f"Converting MCP tools to Anthropic format. Found tools: {all_tools}")
        logging.info(f"MCP manager connection status: {self.mcp_manager.connection_status}")
        logging.info(f"MCP manager available tools cache: {self.mcp_manager.available_tools}")
        
        for server_name, tools in all_tools.items():
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                anthropic_tool = {
                    "name": f"{server_name}_{tool_name}",
                    "description": tool.get('description', f"Tool {tool_name} from {server_name}"),
                    "input_schema": tool.get('inputSchema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
                anthropic_tools.append(anthropic_tool)
                logging.info(f"Added tool: {anthropic_tool['name']}")
        
        logging.info(f"Total Anthropic tools: {len(anthropic_tools)}")
        return anthropic_tools
    
    async def _handle_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Handle tool calls and return results"""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', '')
            tool_input = tool_call.get('input', {})
            
            # Parse server and tool name
            if '_' in tool_name:
                server_name, actual_tool_name = tool_name.split('_', 1)
            else:
                server_name = 'unknown'
                actual_tool_name = tool_name
            
            # Call the tool (either simple or MCP)
            logging.info(f"Calling tool: {server_name}.{actual_tool_name} with input: {tool_input}")
            success, result = await self.mcp_manager.call_tool(
                server_name, actual_tool_name, tool_input
            )
            logging.info(f"Tool call result: success={success}, result={str(result)[:100]}...")
            
            if success:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.get('id'),
                    "content": str(result)
                })
            else:
                tool_results.append({
                    "type": "tool_result", 
                    "tool_use_id": tool_call.get('id'),
                    "content": f"Error: {result}",
                    "is_error": True
                })
        
        return tool_results
    
    async def generate_response(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append(
                    { "role": msg.role, "content": msg.content})
            


            # Get available MCP tools
            tools = self._convert_mcp_tools_to_anthropic_format()
            logging.info(f"Available tools for LLM: {len(tools)} tools")
            if tools:
                logging.info(f"Tool names: {[t['name'] for t in tools]}")
            else:
                logging.warning("NO TOOLS AVAILABLE! Debugging...")
                if self.mcp_manager:
                    available_servers = self.mcp_manager.get_available_tools()
                    all_tools = self.mcp_manager.get_all_available_tools()
                    logging.warning(f"MCP manager servers: {available_servers}")
                    logging.warning(f"MCP manager tools: {all_tools}")
                else:
                    logging.warning("MCP manager is None!")
            
            # Create the initial request
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 4000),
                "messages": anthropic_messages
            }
            
            # Add tools if available
            if tools:
                request_params["tools"] = tools
                logging.info(f"Adding {len(tools)} tools to request")
            else:
                logging.warning("No tools available for LLM")
            
            response = self.client.messages.create(**request_params,
                                                   system="""focus on application landscape discovery, analysis,            
                    and tool usage, if asked to do anything else respond with 
                    'this is not in my current remit'""")
            
            logging.info(f"=== INITIAL RESPONSE RECEIVED ===")
            logging.info(f"Response object type: {type(response)}")
            logging.info(f"Response content type: {type(response.content)}")
            logging.info(f"Response content length: {len(response.content)}")
            
            # Check if there are tool calls
            response_content = []
            tool_calls = []
            
            for i, content_block in enumerate(response.content):
                logging.info(f"Initial content block {i}: type={content_block.type}, object={type(content_block)}")
                if content_block.type == "text":
                    text_content = content_block.text
                    logging.info(f"Initial text content length: {len(text_content)}")
                    logging.info(f"Initial text content preview: {text_content[:200]}...")
                    response_content.append(text_content)
                elif content_block.type == "tool_use":
                    logging.info(f"Initial tool use: {content_block.name}")
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            logging.info(f"After processing initial response: response_content has {len(response_content)} items, tool_calls has {len(tool_calls)} items")
            
            # If there are tool calls, execute them
            if tool_calls and self.mcp_manager:
                logging.info(f"Executing {len(tool_calls)} tool calls...")
                tool_results = await self._handle_tool_calls(tool_calls)
                logging.info(f"Tool results: {tool_results}")
                
                # Add tool results to the conversation and get final response
                anthropic_messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool results in proper format
                anthropic_messages.append({
                    "role": "user", 
                    "content": tool_results  # tool_results is already a properly formatted list
                })
                
                logging.info(f"Final messages to LLM: {len(anthropic_messages)} messages")
                logging.info(f"Last message content: {anthropic_messages[-1]['content'][:200]}...")
                
                # Get final response
                final_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get('max_tokens', 4000),
                    messages=anthropic_messages,
                    tools=tools
                )
                
                logging.info(f"Final response received: {len(final_response.content)} content blocks")
                
                # Process final response content properly
                final_content = []
                additional_tool_calls = []
                
                logging.info(f"=== PROCESSING FINAL RESPONSE ===")
                logging.info(f"Final response object type: {type(final_response)}")
                logging.info(f"Final response content type: {type(final_response.content)}")
                logging.info(f"Final response content length: {len(final_response.content)}")
                
                for i, content_block in enumerate(final_response.content):
                    logging.info(f"Content block {i}: type={content_block.type}, object={type(content_block)}")
                    if content_block.type == "text":
                        text_content = content_block.text
                        logging.info(f"Text content length: {len(text_content)}")
                        logging.info(f"Text content preview: {text_content[:200]}...")
                        if text_content and text_content.strip():  # Only add non-empty text
                            final_content.append(text_content)
                            logging.info(f"Added text content to final_content. Total items: {len(final_content)}")
                        else:
                            logging.warning(f"Skipping empty text content")
                    elif content_block.type == "tool_use":
                        logging.warning(f"LLM wants to call another tool: {content_block.name}")
                        additional_tool_calls.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })
                    else:
                        logging.warning(f"Unexpected content block type: {content_block.type}")
                
                logging.info(f"After processing blocks: final_content has {len(final_content)} items, additional_tool_calls has {len(additional_tool_calls)} items")
                
                # If LLM wants to call more tools, execute them and ask for a final summary
                if additional_tool_calls:
                    logging.info(f"LLM requesting {len(additional_tool_calls)} additional tool calls")
                    additional_results = await self._handle_tool_calls(additional_tool_calls)
                    
                    # Add this response and tool results to conversation
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": final_response.content
                    })
                    anthropic_messages.append({
                        "role": "user",
                        "content": additional_results
                    })
                    
                    # Request a final summary (explicitly ask for text response only)
                    logging.info(f"Requesting final summary after tool calls")
                    final_summary_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=kwargs.get('max_tokens', 4000),
                        messages=anthropic_messages + [{
                            "role": "user",
                            "content": "Please provide a comprehensive summary of your analysis based on the tool results above. Do not make any additional tool calls - just summarize what you found."
                        }]
                        # Intentionally NOT including tools to prevent more tool calls
                    )
                    
                    # Extract text from the final summary response
                    logging.info(f"Final summary response has {len(final_summary_response.content)} content blocks")
                    for i, content_block in enumerate(final_summary_response.content):
                        logging.info(f"Summary content block {i}: type={content_block.type}")
                        if content_block.type == "text":
                            text_content = content_block.text
                            logging.info(f"Summary text content length: {len(text_content)}")
                            logging.info(f"Summary text content preview: {text_content[:200]}...")
                            if text_content and text_content.strip():
                                final_content.append(text_content)
                                logging.info(f"Added summary text content. Total items: {len(final_content)}")
                        else:
                            logging.warning(f"Summary response has unexpected {content_block.type} - should only be text")
                
                # Join all collected text content
                if final_content:
                    result_text = " ".join(final_content).strip()
                    logging.info(f"=== FINAL RESULT ===")
                    logging.info(f"Joined {len(final_content)} text pieces")
                    logging.info(f"Final result length: {len(result_text)}")
                    logging.info(f"Final result preview: {result_text[:300]}...")
                    
                    if not result_text:  # Check if result is empty after joining and stripping
                        logging.error("FINAL RESULT IS EMPTY AFTER PROCESSING!")
                        # Provide a better fallback that explains what happened
                        tool_summary = f"I executed {len(tool_calls)} tool call(s)"
                        if additional_tool_calls:
                            tool_summary += f" followed by {len(additional_tool_calls)} additional tool call(s)"
                        result_text = f"{tool_summary}, but I did not provide a text summary of the results. Please check the logs for details of what was executed."
                else:
                    logging.error("NO FINAL TEXT CONTENT COLLECTED!")
                    logging.error(f"Total conversation messages: {len(anthropic_messages)}")
                    logging.error(f"Tool calls made: {len(tool_calls)} initial + {len(additional_tool_calls) if additional_tool_calls else 0} additional")
                    logging.error(f"Available final_content: {final_content}")
                    
                    # Provide a more detailed fallback
                    tool_summary = f"I executed {len(tool_calls)} tool call(s)"
                    if additional_tool_calls:
                        tool_summary += f" followed by {len(additional_tool_calls)} additional tool call(s)"
                    result_text = f"{tool_summary}, but no final text content was collected from the LLM responses. This may indicate an issue with response processing."
                
                return result_text
            
            # Handle case where there were no tool calls but we got text content
            logging.info(f"=== NO TOOL CALLS - DIRECT RESPONSE ===")
            logging.info(f"response_content has {len(response_content)} items")
            logging.info(f"response_content: {response_content}")
            
            if response_content:
                result_text = " ".join(response_content).strip()
                logging.info(f"Direct response length: {len(result_text)}")
                logging.info(f"Direct response preview: {result_text[:300]}...")
                
                if not result_text:  # Check if result is empty after joining and stripping
                    logging.error("DIRECT RESPONSE IS EMPTY AFTER PROCESSING!")
                    result_text = "The LLM responded but the content was empty."
            else:
                result_text = "No response generated"
                logging.error("NO RESPONSE CONTENT COLLECTED!")
                logging.error(f"Initial response object: {response}")
                logging.error(f"Initial response content: {response.content}")
            
            return result_text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, mcp_manager: Optional['MCPToolManager'] = None):
        super().__init__(model, api_key or os.getenv('OPENAI_API_KEY'))
        if not openai:
            raise ImportError("openai package not installed")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.mcp_manager = mcp_manager
    
    def _convert_mcp_tools_to_openai_format(self) -> list[dict]:
        """Convert MCP tools to OpenAI function format"""
        openai_tools = []
        
        logging.info(f"_convert_mcp_tools_to_openai_format called. MCP manager: {self.mcp_manager is not None}")
        
        if not self.mcp_manager:
            logging.warning("No MCP manager available")
            return openai_tools
        
        all_tools = self.mcp_manager.get_all_available_tools()
        logging.info(f"Converting MCP tools to OpenAI format. Found tools: {all_tools}")
        
        for server_name, tools in all_tools.items():
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}_{tool_name}",
                        "description": tool.get('description', f"Tool {tool_name} from {server_name}"),
                        "parameters": tool.get('inputSchema', {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                openai_tools.append(openai_tool)
                logging.info(f"Added tool: {openai_tool['function']['name']}")
        
        logging.info(f"Total OpenAI tools: {len(openai_tools)}")
        return openai_tools
    
    async def _handle_tool_calls(self, tool_calls: list) -> list[dict]:
        """Handle tool calls and return results"""
        tool_results = []
        
        for tool_call in tool_calls:
            # OpenAI tool_calls are objects, not dictionaries
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
            
            # Parse server and tool name
            if '_' in function_name:
                server_name, actual_tool_name = function_name.split('_', 1)
            else:
                server_name = 'unknown'
                actual_tool_name = function_name
            
            # Call the tool
            logging.info(f"Calling tool: {server_name}.{actual_tool_name} with args: {function_args}")
            success, result = await self.mcp_manager.call_tool(
                server_name, actual_tool_name, function_args
            )
            logging.info(f"Tool call result: success={success}, result={str(result)[:100]}...")
            
            if success:
                tool_results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": str(result)
                })
            else:
                tool_results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": f"Error: {result}"
                })
        
        return tool_results
    
    async def generate_response(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            openai_messages.append({
                "role": "system",
                "content": "Focus on application landscape discovery, analysis, and tool usage. If asked to do anything else, respond with 'this is not in my current remit'."
            })
            
            # Get available MCP tools
            tools = self._convert_mcp_tools_to_openai_format()
            logging.info(f"Available tools for OpenAI: {len(tools)} tools")
            
            # Create the initial request
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": kwargs.get('max_tokens', 4000)
            }
            
            # Add tools if available
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                logging.info(f"Adding {len(tools)} tools to OpenAI request")
            
            response = self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            
            # Check if there are tool calls
            if message.tool_calls and self.mcp_manager:
                logging.info(f"OpenAI requesting {len(message.tool_calls)} tool calls")
                
                # Handle tool calls
                tool_results = await self._handle_tool_calls(message.tool_calls)
                
                # Add assistant message with tool calls
                openai_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Add tool results
                openai_messages.extend(tool_results)
                
                # Get final response
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=kwargs.get('max_tokens', 4000),
                    tools=tools,
                    tool_choice="auto"
                )
                
                return final_response.choices[0].message.content or "No response generated"
            
            return message.content or "No response generated"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""
    
    def __init__(self, model: str = "llama2", base_url: Optional[str] = None, mcp_manager: Optional['MCPToolManager'] = None):
        super().__init__(model)
        if not ollama:
            raise ImportError("ollama package not installed")
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.mcp_manager = mcp_manager
    
    def _convert_mcp_tools_to_ollama_format(self) -> list[dict]:
        """Convert MCP tools to Ollama format (for future tool support)"""
        ollama_tools = []
        
        logging.info(f"_convert_mcp_tools_to_ollama_format called. MCP manager: {self.mcp_manager is not None}")
        
        if not self.mcp_manager:
            logging.warning("No MCP manager available")
            return ollama_tools
        
        all_tools = self.mcp_manager.get_all_available_tools()
        logging.info(f"Converting MCP tools to Ollama format. Found tools: {all_tools}")
        
        for server_name, tools in all_tools.items():
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                ollama_tool = {
                    "name": f"{server_name}_{tool_name}",
                    "description": tool.get('description', f"Tool {tool_name} from {server_name}"),
                    "parameters": tool.get('inputSchema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
                ollama_tools.append(ollama_tool)
                logging.info(f"Added tool: {ollama_tool['name']}")
        
        logging.info(f"Total Ollama tools: {len(ollama_tools)}")
        return ollama_tools
    
    async def _handle_tool_calls_in_text(self, response_text: str) -> str:
        """Handle tool calls that might be embedded in text response"""
        if not self.mcp_manager:
            return response_text
        
        # Look for tool call patterns in the response
        # This is a simple implementation that could be enhanced for specific models
        import re
        
        # Pattern to match tool calls like: TOOL_CALL[server_name.tool_name({"arg": "value"})]
        tool_pattern = r'TOOL_CALL\[([^.]+)\.([^(]+)\(([^)]+)\)\]'
        matches = re.findall(tool_pattern, response_text)
        
        if not matches:
            return response_text
        
        result_text = response_text
        
        for server_name, tool_name, args_str in matches:
            try:
                # Parse arguments
                args = json.loads(args_str)
                
                # Call the tool
                logging.info(f"Calling tool from text: {server_name}.{tool_name} with args: {args}")
                success, result = await self.mcp_manager.call_tool(server_name, tool_name, args)
                
                if success:
                    # Replace the tool call with the result
                    old_pattern = f'TOOL_CALL[{server_name}.{tool_name}({args_str})]'
                    result_text = result_text.replace(old_pattern, f"Tool result: {result}")
                else:
                    # Replace with error message
                    old_pattern = f'TOOL_CALL[{server_name}.{tool_name}({args_str})]'
                    result_text = result_text.replace(old_pattern, f"Tool error: {result}")
                    
            except Exception as e:
                logging.error(f"Error processing tool call: {e}")
                continue
        
        return result_text
    
    async def _handle_ollama_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Handle tool calls from Ollama and return results"""
        tool_messages = []
        
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']
            
            # Parse server and tool name
            if '_' in function_name:
                server_name, actual_tool_name = function_name.split('_', 1)
            else:
                server_name = 'unknown'
                actual_tool_name = function_name
            
            # Call the tool
            logging.info(f"Calling tool: {server_name}.{actual_tool_name} with args: {function_args}")
            success, result = await self.mcp_manager.call_tool(
                server_name, actual_tool_name, function_args
            )
            logging.info(f"Tool call result: success={success}, result={str(result)[:100]}...")
            
            if success:
                tool_messages.append({
                    'role': 'tool',
                    'content': str(result)
                })
            else:
                tool_messages.append({
                    'role': 'tool',
                    'content': f"Error: {result}"
                })
        
        return tool_messages

    async def generate_response(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            # Add system message
            ollama_messages.insert(0, {
                'role': 'system',
                'content': 'You are a helpful assistant focussing on application landscape discovery, analysis, and tool usage. If asked to do anything else decline and respond with "this is not in my current remit".'
            })
            
            # Get available MCP tools
            tools = self._convert_mcp_tools_to_ollama_format()
            logging.info(f"Available tools for Ollama: {len(tools)} tools")
            
            # Prepare request parameters
            request_params = {
                'model': self.model,
                'messages': ollama_messages,
                'stream': False
            }
            
            # Add tools if available
            if tools and self.mcp_manager:
                # Convert tools to Ollama format
                ollama_tools = []
                for tool in tools:
                    ollama_tool = {
                        'type': 'function',
                        'function': {
                            'name': tool['name'],
                            'description': tool['description'],
                            'parameters': tool['parameters']
                        }
                    }
                    ollama_tools.append(ollama_tool)
                
                request_params['tools'] = ollama_tools
                logging.info(f"Adding {len(ollama_tools)} tools to Ollama request")
            
            # Make the initial request
            response = ollama.chat(**request_params)
            
            message = response['message']
            
            # Check if there are tool calls
            if 'tool_calls' in message and message['tool_calls'] and self.mcp_manager:
                logging.info(f"Ollama requesting {len(message['tool_calls'])} tool calls")
                
                # Handle tool calls
                tool_messages = await self._handle_ollama_tool_calls(message['tool_calls'])
                
                # Add assistant message with tool calls
                ollama_messages.append({
                    'role': 'assistant',
                    'content': message.get('content', ''),
                    'tool_calls': message['tool_calls']
                })
                
                # Add tool results
                ollama_messages.extend(tool_messages)
                
                # Get final response
                final_response = ollama.chat(
                    model=self.model,
                    messages=ollama_messages,
                    tools=ollama_tools if tools else None,
                    stream=False
                )
                
                return final_response['message']['content'] or "No response generated"
            
            return message.get('content', 'No response generated')
            
        except Exception as e:
            logging.error(f"Error in Ollama generate_response: {str(e)}")
            return f"Error generating response: {str(e)}"


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation"""
    
    def __init__(self, api_key: str, model: str, mcp_manager: Optional['MCPToolManager'] = None):
        super().__init__(model, api_key)
        if not genai:
            raise ImportError("google-generativeai package not installed")
        self.mcp_manager = mcp_manager
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model)
    
    def _clean_schema_for_google(self, schema: dict) -> dict:
        """Clean schema to remove fields not supported by Google"""
        if not isinstance(schema, dict):
            return schema
        
        # Fields that Google doesn't support
        unsupported_fields = {'title', 'examples', 'default', '$schema', '$id', 'additionalProperties'}
        
        cleaned = {}
        for key, value in schema.items():
            if key not in unsupported_fields:
                if isinstance(value, dict):
                    cleaned[key] = self._clean_schema_for_google(value)
                elif isinstance(value, list):
                    cleaned[key] = [self._clean_schema_for_google(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned[key] = value
        
        return cleaned
    
    def _convert_mcp_tools_to_google_format(self) -> list[dict]:
        """Convert MCP tools to Google function calling format"""
        if not self.mcp_manager:
            return []
        
        google_tools = []
        all_tools = self.mcp_manager.get_all_available_tools()
        logging.info(f"Converting {len(all_tools)} MCP tool servers for Google")
        
        for server_name, tools in all_tools.items():
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                
                # Get and clean the input schema
                raw_schema = tool.get('inputSchema', {
                    'type': 'object',
                    'properties': {},
                    'required': []
                })
                cleaned_schema = self._clean_schema_for_google(raw_schema)
                
                google_tool = {
                    'name': f"{server_name}_{tool_name}",
                    'description': tool.get('description', f'Execute {tool_name} from {server_name}'),
                    'parameters': cleaned_schema
                }
                google_tools.append(google_tool)
                logging.info(f"Added Google tool: {google_tool['name']}")
        
        logging.info(f"Total Google tools: {len(google_tools)}")
        return google_tools
    
    async def generate_response(self, messages: List[ChatMessage]) -> str:
        """Generate response using Google Gemini"""
        try:
            # Convert messages to Google format
            google_messages = []
            
            for msg in messages:
                if msg.role == 'user':
                    google_messages.append({
                        'role': 'user',
                        'parts': [msg.content]
                    })
                elif msg.role == 'assistant':
                    google_messages.append({
                        'role': 'model',
                        'parts': [msg.content]
                    })
            
            # Get available MCP tools
            tools = self._convert_mcp_tools_to_google_format()
            logging.info(f"Available tools for Google: {len(tools)} tools")
            
            # Prepare request parameters
            generation_config = {
                'temperature': 0.1,
                'top_p': 0.95,
                'top_k': 64,
                'max_output_tokens': 8192,
            }
            
            # Create the model with tools if available
            if tools and self.mcp_manager:
                from google.generativeai.types import FunctionDeclaration, Tool
                
                # Convert tools to Google's FunctionDeclaration format
                google_function_declarations = []
                for tool in tools:
                    func_decl = FunctionDeclaration(
                        name=tool['name'],
                        description=tool['description'],
                        parameters=tool['parameters']
                    )
                    google_function_declarations.append(func_decl)
                
                google_tool = Tool(function_declarations=google_function_declarations)
                model = genai.GenerativeModel(
                    model_name=self.model,
                    tools=[google_tool],
                    generation_config=generation_config
                )
            else:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=generation_config
                )
            
            # Generate response
            if google_messages:
                # Use the last message as the prompt
                prompt = google_messages[-1]['parts'][0]
                
                # Include conversation history as context if there are multiple messages
                if len(google_messages) > 1:
                    context = "Previous conversation:\n"
                    for msg in google_messages[:-1]:
                        role = "Human" if msg['role'] == 'user' else "Assistant"
                        context += f"{role}: {msg['parts'][0]}\n"
                    prompt = f"{context}\nCurrent question: {prompt}"
                
                response = model.generate_content(prompt)
                
                # Handle function calls if present
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Check for function calls
                    if hasattr(candidate.content, 'parts'):
                        tool_calls_made = False
                        final_response = ""
                        
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                # Execute the function call
                                function_name = part.function_call.name
                                function_args = dict(part.function_call.args) if part.function_call.args else {}
                                
                                logging.info(f"Google function call: {function_name} with args: {function_args}")
                                
                                # Parse server and tool name
                                if '_' in function_name:
                                    server_name, actual_tool_name = function_name.split('_', 1)
                                else:
                                    server_name = 'unknown'
                                    actual_tool_name = function_name
                                
                                # Execute the MCP tool
                                if self.mcp_manager:
                                    success, tool_result = await self.mcp_manager.call_tool(
                                        server_name, actual_tool_name, function_args
                                    )
                                    logging.info(f"Tool result: success={success}, result={tool_result}")
                                    tool_calls_made = True
                                    
                                    if success:
                                        # Generate follow-up response with tool result using a model WITHOUT tools
                                        # to prevent infinite function calling loops
                                        text_only_model = genai.GenerativeModel(
                                            model_name=self.model,
                                            generation_config=generation_config
                                        )
                                        follow_up_prompt = f"{prompt}\n\nTool '{function_name}' returned: {tool_result}\n\nPlease provide a comprehensive response based on this information. Do not call any tools."
                                        follow_up_response = text_only_model.generate_content(follow_up_prompt)
                                        
                                        try:
                                            if follow_up_response.text:
                                                final_response = follow_up_response.text
                                            else:
                                                final_response = f"Tool '{function_name}' executed successfully. Result: {tool_result}"
                                        except Exception as e:
                                            logging.warning(f"Could not get follow_up_response.text: {e}")
                                            final_response = f"Tool '{function_name}' executed successfully. Result: {tool_result}"
                                    else:
                                        final_response = f"Error executing tool {function_name}: {tool_result}"
                            elif hasattr(part, 'text') and part.text:
                                final_response = part.text
                        
                        if tool_calls_made and final_response:
                            return final_response
                        elif tool_calls_made:
                            return "Tool calls completed successfully."
                
                # Return the text response - handle cases where response.text fails
                try:
                    if response.text:
                        return response.text
                    else:
                        return "No response generated"
                except Exception as e:
                    # If response.text fails (e.g., due to function calls), extract text from parts
                    logging.warning(f"Could not get response.text directly: {e}")
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate.content, 'parts'):
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            if text_parts:
                                return ' '.join(text_parts)
                    return "Could not extract response text"
            else:
                return "No messages to process"
                
        except Exception as e:
            logging.error(f"Error in Google generate_response: {str(e)}")
            return f"Error generating response: {str(e)}"


class ChatHistory:
    """Manages chat history persistence"""
    
    def __init__(self, history_file: str = ".atlas_chat_history.json"):
        self.history_file = Path(history_file)
        self.messages: list[ChatMessage] = []
        self.load_history()
    
    def add_message(self, role: str, content: str, metadata: dict | None = None):
        """Add a message to history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.messages.append(message)
        self.save_history()
    
    def get_messages(self, limit: Optional[int] = None) -> list[ChatMessage]:
        """Get chat messages"""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear_history(self):
        """Clear chat history"""
        self.messages = []
        self.save_history()
    
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([msg.to_dict() for msg in self.messages], f, indent=2)
        except Exception as e:
            st.error(f"Failed to save chat history: {str(e)}")
    
    def load_history(self):
        """Load history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.messages = [ChatMessage.from_dict(msg) for msg in data]
        except Exception as e:
            st.warning(f"Failed to load chat history: {str(e)}")
            self.messages = []


class AtlasDiscoveryAgent:
    """Main Atlas Discovery Agent chat application"""
    
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        self.history = ChatHistory()
        self.mcp_manager = MCPToolManager()
        self.llm_provider: Optional[LLMProvider] = None
        self.setup_page_config()
        self.setup_default_mcp_tools()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Atlas Discovery Agent",
            page_icon="🗺️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_default_mcp_tools(self):
        """Setup default MCP tools - only Neo4j"""
        # Neo4j MCP server (using same config as Claude Desktop)

        # Local Neo4j server
        mcp_server_name = "local-neo4j"
        neo4j_db_url = os.getenv('NEO4J_DB_URL', 'neo4j://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'scale-silence-limit-minus-rent-7661')

        # Try to use the local Neo4j MCP server if available, otherwise fall back to uvx
        local_mcp_path = os.getenv('LOCAL_NEO4J_MCP_SERVER_PATH')
        if local_mcp_path and os.path.exists(local_mcp_path):
            # Use the local Neo4j MCP server executable
            self.mcp_manager.add_tool_server(
                mcp_server_name, local_mcp_path,
                args=["--transport", "stdio",
                      "--db-url", neo4j_db_url, 
                      "--username", neo4j_username, 
                      "--password", neo4j_password],
                description="Neo4j graph database operations via Cypher queries (local server)"
            )
        else:
            # Fallback to uvx (original Claude Desktop config)
            self.mcp_manager.add_tool_server(
                mcp_server_name, "uvx",
                args=["mcp-neo4j-cypher",
                      "--transport", "stdio",
                      "--db-url", neo4j_db_url, 
                      "--username", neo4j_username, 
                      "--password", neo4j_password],
                description="Neo4j graph database operations via Cypher queries"
            )
    
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1>🗺️ Atlas Discovery Agent</h1>
            <p style="color: #666; font-size: 1.1em;">
                AI-powered application landscape discovery and analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # LLM Provider Configuration
        st.sidebar.subheader("🤖 LLM Provider")
        
        provider = st.sidebar.selectbox(
            "Provider",
            ["anthropic", "openai", "ollama", "google"],
            index=0  # Default to anthropic
        )
        
        if provider == "anthropic":
            model = st.sidebar.selectbox(
                "Model",
                ["claude-sonnet-4-20250514", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                index=0
            )
            api_key = st.sidebar.text_input("API Key", type="password", 
                                          value=os.getenv('ANTHROPIC_API_KEY', ''))
        elif provider == "openai":
            model = st.sidebar.selectbox(
                "Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0
            )
            api_key = st.sidebar.text_input("API Key", type="password", 
                                          value=os.getenv('OPENAI_API_KEY', ''))
        elif provider == "google":
            model = st.sidebar.selectbox(
                "Model",
                ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                index=0
            )
            api_key = st.sidebar.text_input("API Key", type="password", 
                                          value=os.getenv('GEMINI_API_KEY', ''))
        else:  # ollama
            model = st.sidebar.text_input("Model", value="llama3.1")
            base_url = st.sidebar.text_input("Base URL", 
                                           value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
            api_key = None
            st.sidebar.info("💡 **Note:** Tool calling requires models like llama3.1 or later that support function calling.")
        
        # Initialize LLM provider (only if not already created or if config changed)
        provider_key = f"{provider}_{model}_{api_key}"
        current_provider_key = getattr(self, '_current_provider_key', None)
        
        # Use session state to persist provider across Streamlit reruns
        if 'llm_provider' not in st.session_state:
            st.session_state.llm_provider = None
            st.session_state.provider_key = None
        
        logging.info(f"Provider key check: current='{current_provider_key}', new='{provider_key}', session_key='{st.session_state.provider_key}'")
        logging.info(f"Provider exists: instance={self.llm_provider is not None}, session={st.session_state.llm_provider is not None}")
        
        # Check if we need to create a new provider or can reuse existing one
        need_new_provider = (
            st.session_state.provider_key != provider_key or 
            st.session_state.llm_provider is None
        )
        
        logging.info(f"Need new provider: {need_new_provider}")
        
        if need_new_provider:
            try:
                if provider == "anthropic":
                    self.llm_provider = AnthropicProvider(model, api_key, self.mcp_manager)
                elif provider == "openai":
                    self.llm_provider = OpenAIProvider(model, api_key, self.mcp_manager)
                elif provider == "google":
                    self.llm_provider = GoogleProvider(api_key, model, self.mcp_manager)
                else:
                    self.llm_provider = OllamaProvider(model, base_url, self.mcp_manager)
                
                # Store in both instance and session state
                self._current_provider_key = provider_key
                st.session_state.llm_provider = self.llm_provider
                st.session_state.provider_key = provider_key
                logging.info(f"Created new LLM provider: {provider} ({model})")
                
            except Exception as e:
                st.sidebar.error(f"❌ Failed to initialize {provider}: {str(e)}")
                self.llm_provider = None
                self._current_provider_key = None
                st.session_state.llm_provider = None
                st.session_state.provider_key = None
        else:
            # Reuse existing provider from session state
            self.llm_provider = st.session_state.llm_provider
            self._current_provider_key = st.session_state.provider_key
            logging.info(f"Reusing existing LLM provider: {provider} ({model})")
            
            # CRITICAL: Update the provider's MCP manager reference to current one
            if hasattr(self.llm_provider, 'mcp_manager'):
                old_manager_id = id(self.llm_provider.mcp_manager)
                new_manager_id = id(self.mcp_manager)
                self.llm_provider.mcp_manager = self.mcp_manager
                logging.info(f"Updated provider MCP manager: {old_manager_id} -> {new_manager_id}")
        
        # Always ensure instance-level provider is set from session state
        if st.session_state.llm_provider is not None and self.llm_provider is None:
            self.llm_provider = st.session_state.llm_provider
            logging.info("Set instance provider from session state")
            
            # Also update MCP manager reference in this case
            if hasattr(self.llm_provider, 'mcp_manager'):
                self.llm_provider.mcp_manager = self.mcp_manager
                logging.info("Updated MCP manager reference after setting from session")
        
        # Show provider status with current tool count
        if self.llm_provider:
            try:
                # Show tool count for LLM
                if hasattr(self.llm_provider, 'mcp_manager'):
                    if provider == "anthropic":
                        tools = self.llm_provider._convert_mcp_tools_to_anthropic_format()
                    elif provider == "openai":
                        tools = self.llm_provider._convert_mcp_tools_to_openai_format()
                    elif provider == "google":
                        tools = self.llm_provider._convert_mcp_tools_to_google_format()
                    else:  # ollama
                        tools = self.llm_provider._convert_mcp_tools_to_ollama_format()
                    tool_count = len(tools)
                    if tool_count > 0:
                        st.sidebar.success(f"✅ {provider} ({model}) ready with {tool_count} MCP tools")
                    else:
                        st.sidebar.success(f"✅ {provider} ({model}) ready")
                        st.sidebar.warning("⚠️ No MCP tools connected. Click 'Connect Tools' below to enable tool calling.")
                else:
                    st.sidebar.success(f"✅ {provider} ({model}) ready")
            except Exception as e:
                st.sidebar.error(f"❌ Error checking tools: {str(e)}")
        else:
            st.sidebar.error("❌ LLM provider not available")
        
        # MCP Tools Configuration
        st.sidebar.subheader("🔧 MCP Tools")
        
        # Show tool connection summary
        connection_summary = self.mcp_manager.get_connection_summary()
        
        if connection_summary:
            for tool_name, info in connection_summary.items():
                status = info['status']
                description = info['description']
                error = info['error']
                
                # Status indicator
                if status == 'connected':
                    status_icon = "✅"
                    status_color = "green"
                elif status == 'connecting':
                    status_icon = "🔄"
                    status_color = "orange"
                elif status == 'failed':
                    status_icon = "❌"
                    status_color = "red"
                else:
                    status_icon = "⚪"
                    status_color = "gray"
                
                # Display tool with status
                st.sidebar.markdown(f"{status_icon} **{tool_name}** ({status})")
                if description:
                    st.sidebar.caption(description)
                
                # Show command details for complex tools
                if tool_name == 'local-neo4j':
                    st.sidebar.caption(f"Command: {info['command']} mcp-neo4j-cypher")
                    st.sidebar.caption(f"Database: {os.getenv('NEO4J_DB_URL', 'neo4j://localhost:7687')}")
                
                if error and status in ['failed', 'error']:
                    st.sidebar.error(f"Error: {error}")
                
                st.sidebar.markdown("---")
        
        # Tool connection button
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔌 Connect Tools", use_container_width=True):
                with st.spinner("Connecting to MCP servers..."):
                    # Store connection results in session state to persist across reruns
                    connection_results = asyncio.run(self.mcp_manager.connect_tools())
                    st.session_state.connection_results = connection_results
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Show connection results if available
        if hasattr(st.session_state, 'connection_results'):
            st.sidebar.subheader("🔗 Connection Results")
            for result in st.session_state.connection_results:
                if result.startswith("✅") or result.startswith("🚀"):
                    st.sidebar.success(result)
                elif result.startswith("⚠️"):
                    st.sidebar.warning(result)
                else:
                    st.sidebar.error(result)
            
            # Clear results after showing
            if st.sidebar.button("Clear Results"):
                del st.session_state.connection_results
                st.rerun()
        
        # Show available MCP tools
        available_tools = self.mcp_manager.get_all_available_tools()
        st.sidebar.subheader("🛠️ Available MCP Tools")
        
        if available_tools:
            for server_name, tools in available_tools.items():
                st.sidebar.write(f"**{server_name}:** {len(tools)} tools")
                for tool in tools:
                    st.sidebar.write(f"  • {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                st.sidebar.markdown("---")
        else:
            st.sidebar.write("Connect to MCP servers to see available tools")
        
        # Chat History Management
        st.sidebar.subheader("💬 Chat History")
        
        message_count = len(self.history.get_messages())
        st.sidebar.write(f"Messages: {message_count}")
        
        if st.sidebar.button("Clear History"):
            self.history.clear_history()
            st.rerun()
    
    def render_chat_interface(self):
        """Render main chat interface"""
        
        # Show MCP tool status prominently
        if self.llm_provider and hasattr(self.llm_provider, 'mcp_manager'):
            available_tools = self.llm_provider.mcp_manager.get_all_available_tools()
            total_tools = sum(len(tools) for tools in available_tools.values())
            
            if total_tools == 0:
                st.info("💡 **Tip:** Connect to MCP tools in the sidebar to enable the LLM to query your Neo4j database, read files, and perform other advanced operations.")
            else:
                st.success(f"🛠️ **{total_tools} MCP tools connected** - The LLM can now use Neo4j queries and other advanced capabilities!")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        messages = self.history.get_messages()
        
        with chat_container:
            for message in messages:
                with st.chat_message(message.role):
                    st.write(message.content)
                    st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")
        
        # Chat input
        if prompt := st.chat_input("Ask Atlas Discovery Agent..."):
            if not self.llm_provider:
                st.error("Please configure an LLM provider in the sidebar first.")
                return
            
            # Add user message
            self.history.add_message("user", prompt)
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        msg = self.history.get_messages(1)
                        if msg:
                            logging.info(f"Generating response for message: {msg[-1].content}")

                        # Get recent messages for context
                        recent_messages = self.history.get_messages(limit=10)
                        
                        # Generate response
                        response = asyncio.run(
                            self.llm_provider.generate_response(recent_messages)
                        )
                        
                        # Debug the response in the UI
                        logging.info(f"=== UI LAYER RESPONSE ===")
                        logging.info(f"Response type: {type(response)}")
                        logging.info(f"Response length: {len(response) if response else 0}")
                        logging.info(f"Response preview: {response[:300] if response else 'None/Empty'}")
                        
                        # Display response with error handling for empty/problematic responses
                        if not response:
                            st.error("❌ **No response received from LLM**")
                            response = "Error: No response received from LLM"
                        elif response.strip() == "":
                            st.error("❌ **Empty response received from LLM**")
                            response = "Error: Empty response received from LLM"
                        elif response == "No text response generated":
                            st.error("❌ **LLM did not generate text response despite completing tool calls**")
                            st.info("💡 This usually means the LLM used tools but didn't provide a final text summary. Check logs for tool call details.")
                        elif response.startswith("Error generating response:"):
                            st.error(f"❌ **LLM Error:** {response}")
                        else:
                            # Normal response - display it
                            st.write(response)
                        
                        # Always add to history, even error responses
                        self.history.add_message("assistant", response)
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        self.history.add_message("assistant", error_msg)
    
    def run(self):
        """Run the Atlas Discovery Agent"""
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Atlas Discovery Agent Chat Interface")
    parser.add_argument('--llm-provider', choices=['anthropic', 'openai', 'ollama', 'google'], 
                       default='anthropic', help='LLM provider')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--port', type=int, default=8501, help='Streamlit port')
    
    args = parser.parse_args()
    
    # Set environment variables from args if provided
    if args.llm_provider:
        os.environ['ATLAS_LLM_PROVIDER'] = args.llm_provider
    if args.model:
        os.environ['ATLAS_MODEL'] = args.model
    
    # Initialize and run the agent (use session state to persist)
    if 'atlas_agent' not in st.session_state:
        st.session_state.atlas_agent = AtlasDiscoveryAgent()
    
    agent = st.session_state.atlas_agent
    agent.run()


if __name__ == "__main__":
    main()