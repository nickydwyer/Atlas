# Generate SSL/TLS certificates for Neo4j on Windows
param(
    [string]$Domain = "atlas-vm.northeurope.cloudapp.azure.com",
    [string]$CertPath = "C:\atlas\Atlas\certificates",
    [string]$Location = "northeurope"
)

# Update domain if location is provided
if ($Domain -eq "atlas-vm.northeurope.cloudapp.azure.com" -and $Location -ne "northeurope") {
    $Domain = "atlas-vm.$Location.cloudapp.azure.com"
}

Write-Host "Generating SSL/TLS certificates for Neo4j..." -ForegroundColor Green
Write-Host "Domain: $Domain" -ForegroundColor Cyan

# Create certificate directories
$boltPath = "$CertPath\bolt"
$httpsPath = "$CertPath\https"

New-Item -ItemType Directory -Force -Path $boltPath | Out-Null
New-Item -ItemType Directory -Force -Path $httpsPath | Out-Null

# Function to generate certificate
function Generate-Certificate {
    param(
        [string]$Path,
        [string]$CommonName,
        [string]$Type
    )
    
    Write-Host "Generating $Type certificate..." -ForegroundColor Yellow
    
    # Generate private key
    & openssl genrsa -out "$Path\private.key" 4096
    
    # Generate certificate signing request
    $subj = "/C=US/ST=State/L=City/O=Atlas/OU=IT/CN=$CommonName"
    & openssl req -new -key "$Path\private.key" -out "$Path\cert.csr" -subj $subj
    
    # Generate self-signed certificate (valid for 365 days)
    & openssl x509 -req -days 365 -in "$Path\cert.csr" -signkey "$Path\private.key" -out "$Path\public.crt"
    
    # Clean up CSR
    Remove-Item "$Path\cert.csr"
    
    # Set appropriate permissions
    $acl = Get-Acl "$Path\private.key"
    $acl.SetAccessRuleProtection($true, $false)
    $adminRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators", "FullControl", "Allow")
    $systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule("SYSTEM", "FullControl", "Allow")
    $acl.SetAccessRule($adminRule)
    $acl.SetAccessRule($systemRule)
    Set-Acl "$Path\private.key" $acl
    
    Write-Host "$Type certificate generated successfully!" -ForegroundColor Green
}

# Check if OpenSSL is installed
if (!(Get-Command openssl -ErrorAction SilentlyContinue)) {
    Write-Host "OpenSSL not found. Installing via Chocolatey..." -ForegroundColor Yellow
    choco install openssl -y
    refreshenv
}

# Generate certificates for Bolt and HTTPS
Generate-Certificate -Path $boltPath -CommonName $Domain -Type "Bolt"
Generate-Certificate -Path $httpsPath -CommonName $Domain -Type "HTTPS"

# Create a certificate bundle for Java truststore (optional)
Write-Host "Creating certificate information file..." -ForegroundColor Yellow

$certInfo = @"
Certificate Generation Complete!

Certificate Locations:
- Bolt SSL: $boltPath
- HTTPS SSL: $httpsPath

Certificate Details:
- Common Name: $Domain
- Validity: 365 days
- Key Size: 4096 bits

To use these certificates:
1. Ensure the certificates directory is mounted in docker-compose.yml
2. Update the domain name in your Azure VM's DNS settings
3. Configure Atlas to use bolt+s:// and https:// URLs

For production use:
- Replace self-signed certificates with CA-signed certificates
- Use Azure Key Vault to store certificates
- Enable certificate rotation

Connection URLs:
- Neo4j Browser: https://$Domain:7473
- Bolt: bolt+s://$Domain:7688
"@

Set-Content "$CertPath\README.txt" $certInfo
Write-Host $certInfo -ForegroundColor Cyan

Write-Host "`nCertificate generation complete!" -ForegroundColor Green