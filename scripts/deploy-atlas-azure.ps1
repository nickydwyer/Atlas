# Atlas Azure Deployment Script
param(
    [Parameter(Mandatory=$false)]
    [string]$Location = "northeurope",  # Default: North Europe
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "atlas-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$VMName = "atlas-vm",
    
    [Parameter(Mandatory=$false)]
    [string]$VMSize = "Standard_D8s_v3",
    
    [Parameter(Mandatory=$false)]
    [string]$AdminUsername = "atlasadmin",
    
    [Parameter(Mandatory=$true)]
    [string]$AdminPassword,
    
    [Parameter(Mandatory=$false)]
    [string]$OSImage = "MicrosoftWindowsDesktop:windows-11:win11-23h2-pro:latest",
    
    [Parameter(Mandatory=$false)]
    [int]$DiskSizeGB = 256,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipPortOpening = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseExistingResourceGroup = $false
)

# Color output functions
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Validate Azure CLI is installed
if (!(Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI is not installed. Please install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
}

# Check if logged in to Azure
$account = az account show 2>$null
if (!$account) {
    Write-Warning "Not logged in to Azure. Running 'az login'..."
    az login
}

Write-Info "================================"
Write-Info "Atlas Azure Deployment"
Write-Info "================================"
Write-Info "Location: $Location"
Write-Info "Resource Group: $ResourceGroup"
Write-Info "VM Name: $VMName"
Write-Info "VM Size: $VMSize"
Write-Info "Disk Size: $DiskSizeGB GB"
Write-Info "================================"

# Create or verify resource group
if (!$UseExistingResourceGroup) {
    Write-Info "Creating resource group '$ResourceGroup' in '$Location'..."
    az group create --name $ResourceGroup --location $Location --output table
} else {
    Write-Info "Using existing resource group '$ResourceGroup'..."
}

# Create VM
Write-Info "Creating VM '$VMName'..."
$vmCreateParams = @(
    "--resource-group", $ResourceGroup,
    "--name", $VMName,
    "--image", $OSImage,
    "--size", $VMSize,
    "--admin-username", $AdminUsername,
    "--admin-password", $AdminPassword,
    "--public-ip-address-allocation", "static",
    "--os-disk-size-gb", $DiskSizeGB,
    "--location", $Location
)

$result = az vm create @vmCreateParams --output json | ConvertFrom-Json

if ($result) {
    Write-Success "VM created successfully!"
    Write-Info "Public IP: $($result.publicIpAddress)"
} else {
    Write-Error "Failed to create VM"
    exit 1
}

# Open ports if not skipped
if (!$SkipPortOpening) {
    Write-Info "Opening required ports..."
    
    $ports = @(
        @{Port=3389; Priority=1000; Name="RDP"},
        @{Port=7474; Priority=1001; Name="Neo4j-Browser"},
        @{Port=7687; Priority=1002; Name="Neo4j-Bolt"},
        @{Port=8501; Priority=1003; Name="Streamlit"},
        @{Port=7473; Priority=1004; Name="Neo4j-HTTPS"},
        @{Port=7688; Priority=1005; Name="Neo4j-Bolt-TLS"}
    )
    
    foreach ($portInfo in $ports) {
        Write-Info "Opening port $($portInfo.Port) ($($portInfo.Name))..."
        az vm open-port `
            --resource-group $ResourceGroup `
            --name $VMName `
            --port $portInfo.Port `
            --priority $portInfo.Priority `
            --output none
    }
    
    Write-Success "All ports opened successfully!"
}

# Generate DNS name
$vmPublicIP = az vm show --resource-group $ResourceGroup --name $VMName --show-details --query publicIps -o tsv
$dnsName = "$VMName-$(Get-Random -Maximum 9999)"

Write-Info "Setting DNS name: $dnsName.$Location.cloudapp.azure.com"
az network public-ip update `
    --resource-group $ResourceGroup `
    --name "${VMName}PublicIP" `
    --dns-name $dnsName `
    --output none

# Output connection information
Write-Success "================================"
Write-Success "Deployment Complete!"
Write-Success "================================"
Write-Info "VM Name: $VMName"
Write-Info "Resource Group: $ResourceGroup"
Write-Info "Location: $Location"
Write-Info "Public IP: $vmPublicIP"
Write-Info "DNS Name: $dnsName.$Location.cloudapp.azure.com"
Write-Info "RDP Username: $AdminUsername"
Write-Info ""
Write-Info "To connect via RDP:"
Write-Info "mstsc /v:$vmPublicIP"
Write-Info ""
Write-Info "Next steps:"
Write-Info "1. RDP to the VM using the credentials provided"
Write-Info "2. Run the setup-atlas.ps1 script on the VM"
Write-Info "3. Configure SSL certificates for production use"
Write-Success "================================"

# Save deployment info
$deploymentInfo = @{
    VMName = $VMName
    ResourceGroup = $ResourceGroup
    Location = $Location
    PublicIP = $vmPublicIP
    DNSName = "$dnsName.$Location.cloudapp.azure.com"
    AdminUsername = $AdminUsername
    DeploymentDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

$deploymentInfo | ConvertTo-Json | Out-File "atlas-deployment-info.json"
Write-Info "Deployment information saved to atlas-deployment-info.json"