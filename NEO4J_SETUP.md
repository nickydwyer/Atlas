# Neo4j Docker and MCP Server Setup Guide

This guide will walk you through setting up Neo4j using Docker and installing the Neo4j MCP server on Linux or macOS.

## Prerequisites

- Docker installed on your system
- Terminal/Command line access

### Install and run Neo4j Docker image on MacOS

1. Create directories for Neo4j data and logs:

    ```bash
    mkdir -p $HOME/neo4j/data
    mkdir -p $HOME/neo4j/logs
    ```

2. Pull the Neo4j Docker image:

    ```bash
    docker pull neo4j
    ```

3. Run the Neo4j container with the necessary arguments and APOC plugin extensions:

    ```bash
    docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
        --volume=$HOME/neo4j/data:/data \
        --volume=$HOME/neo4j/logs:/logs \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_import_file_enabled=true \
    neo4j:latest 

    ```bash

### Check to see that Neo4j is working, reset the default password

1. Using your browser navigate to ```http://<hostname>:7474``` and login using the default user name 'neo4j' and password 'neo4j'
2. You will be prompted to set or generate a new password. Generate a new password and make a note of it.

## Installing the Neo4j MCP Servers
Install the Neo4j MCP servers locally using the instructions here: [servers/mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher).
You will then need to configure the server with the Neo4j host,username, database and password.

## IDE Integration with Neo4j MCP

After setting up Neo4j and the MCP server, you can configure your IDE to connect directly to the database:

- **VSCode Users**: Follow the [MCP IDE Integration Guide](./docs/mcp-ide-integration.md#vscode-integration) to set up the MCP extension
- **Claude Desktop Users**: See the [Claude Desktop configuration section](./docs/mcp-ide-integration.md#claude-desktop-integration) for setup instructions
- **Configuration Examples**: Find example configuration files in the `.vscode/` directory and `examples/claude_config.json`

This enables context-aware code completion and direct database queries from your development environment.

## Appendix: Installing and Using Docker on a Linux VM

1. SSH into the VM instance.
2. Install Docker on the VM instance.

### Update the package list and upgrade all packages

```bash
    sudo apt-get update
    sudo apt-get upgrade -y
```

### Install necessary packages for Docker

```bash
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
```

### Add Docker's official GPG key

```bash
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    cd /etc/apt
    sudo cp trusted.gpg trusted.gpg.d
```

### Add Docker's official APT repository

```bash
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

### Update the package list again

```bash
    sudo apt-get update
```

### Install Docker

```bash
    sudo apt-get install -y docker-ce
```

### Verify Docker installation, add your username to the docker group (you'll need to logout and login again)

```bash
    sudo systemctl status docker
    sudo usermod -aG docker $USER
```

logout and login.

```bash
    docker ps
```
