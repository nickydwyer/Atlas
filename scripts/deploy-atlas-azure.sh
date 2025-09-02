#!/bin/bash
# Atlas Azure Deployment Script for Linux/macOS

# Default values
LOCATION="northeurope"  # Default: North Europe
RESOURCE_GROUP="atlas-rg"
VM_NAME="atlas-vm"
VM_SIZE="Standard_D8s_v3"
ADMIN_USERNAME="atlasadmin"
OS_IMAGE="MicrosoftWindowsDesktop:windows-11:win11-23h2-pro:latest"
DISK_SIZE_GB=256
SKIP_PORT_OPENING=false
USE_EXISTING_RG=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -l, --location LOCATION         Azure region (default: northeurope)"
    echo "  -g, --resource-group NAME       Resource group name (default: atlas-rg)"
    echo "  -n, --vm-name NAME              VM name (default: atlas-vm)"
    echo "  -s, --vm-size SIZE              VM size (default: Standard_D8s_v3)"
    echo "  -u, --admin-username USERNAME   Admin username (default: atlasadmin)"
    echo "  -p, --admin-password PASSWORD   Admin password (required)"
    echo "  -d, --disk-size SIZE            OS disk size in GB (default: 256)"
    echo "  --skip-ports                    Skip opening ports"
    echo "  --use-existing-rg               Use existing resource group"
    echo "  -h, --help                      Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -g|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -n|--vm-name)
            VM_NAME="$2"
            shift 2
            ;;
        -s|--vm-size)
            VM_SIZE="$2"
            shift 2
            ;;
        -u|--admin-username)
            ADMIN_USERNAME="$2"
            shift 2
            ;;
        -p|--admin-password)
            ADMIN_PASSWORD="$2"
            shift 2
            ;;
        -d|--disk-size)
            DISK_SIZE_GB="$2"
            shift 2
            ;;
        --skip-ports)
            SKIP_PORT_OPENING=true
            shift
            ;;
        --use-existing-rg)
            USE_EXISTING_RG=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required parameters
if [ -z "$ADMIN_PASSWORD" ]; then
    echo -e "${RED}Error: Admin password is required${NC}"
    usage
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed${NC}"
    echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Azure. Running 'az login'...${NC}"
    az login
fi

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}Atlas Azure Deployment${NC}"
echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}Location: $LOCATION${NC}"
echo -e "${CYAN}Resource Group: $RESOURCE_GROUP${NC}"
echo -e "${CYAN}VM Name: $VM_NAME${NC}"
echo -e "${CYAN}VM Size: $VM_SIZE${NC}"
echo -e "${CYAN}Disk Size: $DISK_SIZE_GB GB${NC}"
echo -e "${CYAN}================================${NC}"

# Create or verify resource group
if [ "$USE_EXISTING_RG" = false ]; then
    echo -e "${CYAN}Creating resource group '$RESOURCE_GROUP' in '$LOCATION'...${NC}"
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output table
else
    echo -e "${CYAN}Using existing resource group '$RESOURCE_GROUP'...${NC}"
fi

# Create VM
echo -e "${CYAN}Creating VM '$VM_NAME'...${NC}"
VM_RESULT=$(az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --image "$OS_IMAGE" \
    --size "$VM_SIZE" \
    --admin-username "$ADMIN_USERNAME" \
    --admin-password "$ADMIN_PASSWORD" \
    --public-ip-address-allocation static \
    --os-disk-size-gb "$DISK_SIZE_GB" \
    --location "$LOCATION" \
    --output json)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}VM created successfully!${NC}"
    PUBLIC_IP=$(echo "$VM_RESULT" | jq -r '.publicIpAddress')
    echo -e "${CYAN}Public IP: $PUBLIC_IP${NC}"
else
    echo -e "${RED}Failed to create VM${NC}"
    exit 1
fi

# Open ports if not skipped
if [ "$SKIP_PORT_OPENING" = false ]; then
    echo -e "${CYAN}Opening required ports...${NC}"
    
    # Define ports
    declare -a PORTS=(
        "3389:1000:RDP"
        "7474:1001:Neo4j-Browser"
        "7687:1002:Neo4j-Bolt"
        "8501:1003:Streamlit"
        "7473:1004:Neo4j-HTTPS"
        "7688:1005:Neo4j-Bolt-TLS"
    )
    
    for port_info in "${PORTS[@]}"; do
        IFS=':' read -r port priority name <<< "$port_info"
        echo -e "${CYAN}Opening port $port ($name)...${NC}"
        az vm open-port \
            --resource-group "$RESOURCE_GROUP" \
            --name "$VM_NAME" \
            --port "$port" \
            --priority "$priority" \
            --output none
    done
    
    echo -e "${GREEN}All ports opened successfully!${NC}"
fi

# Generate DNS name
DNS_NAME="$VM_NAME-$(shuf -i 1000-9999 -n 1)"
echo -e "${CYAN}Setting DNS name: $DNS_NAME.$LOCATION.cloudapp.azure.com${NC}"
az network public-ip update \
    --resource-group "$RESOURCE_GROUP" \
    --name "${VM_NAME}PublicIP" \
    --dns-name "$DNS_NAME" \
    --output none

# Get final public IP
VM_PUBLIC_IP=$(az vm show --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --show-details --query publicIps -o tsv)

# Output connection information
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${CYAN}VM Name: $VM_NAME${NC}"
echo -e "${CYAN}Resource Group: $RESOURCE_GROUP${NC}"
echo -e "${CYAN}Location: $LOCATION${NC}"
echo -e "${CYAN}Public IP: $VM_PUBLIC_IP${NC}"
echo -e "${CYAN}DNS Name: $DNS_NAME.$LOCATION.cloudapp.azure.com${NC}"
echo -e "${CYAN}RDP Username: $ADMIN_USERNAME${NC}"
echo ""
echo -e "${CYAN}To connect via RDP:${NC}"
echo -e "${CYAN}Windows: mstsc /v:$VM_PUBLIC_IP${NC}"
echo -e "${CYAN}macOS: Use Microsoft Remote Desktop app${NC}"
echo -e "${CYAN}Linux: rdesktop $VM_PUBLIC_IP${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo -e "${CYAN}1. RDP to the VM using the credentials provided${NC}"
echo -e "${CYAN}2. Run the setup-atlas.ps1 script on the VM${NC}"
echo -e "${CYAN}3. Configure SSL certificates for production use${NC}"
echo -e "${GREEN}================================${NC}"

# Save deployment info
cat > atlas-deployment-info.json <<EOF
{
  "VMName": "$VM_NAME",
  "ResourceGroup": "$RESOURCE_GROUP",
  "Location": "$LOCATION",
  "PublicIP": "$VM_PUBLIC_IP",
  "DNSName": "$DNS_NAME.$LOCATION.cloudapp.azure.com",
  "AdminUsername": "$ADMIN_USERNAME",
  "DeploymentDate": "$(date -u +"%Y-%m-%d %H:%M:%S")"
}
EOF

echo -e "${CYAN}Deployment information saved to atlas-deployment-info.json${NC}"