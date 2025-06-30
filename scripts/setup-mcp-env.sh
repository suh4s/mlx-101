#!/bin/bash

# MCP Environment Setup Script
# This script helps set up environment variables for MCP servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if we're in the right directory
if [ ! -f ".env.example" ]; then
    print_error "Error: .env.example not found. Please run this script from the project root."
    exit 1
fi

print_header "MCP Environment Setup"

# Step 1: Check for .env file and create from .env.example if needed
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "No .env file found, but .env.example exists. Creating .env from template..."
        cp .env.example .env
        print_success ".env file created from .env.example"
    else
        print_error "Neither .env nor .env.example found. Cannot proceed."
        exit 1
    fi
else
    print_info ".env file found"
    # Check if .env.example is newer than .env
    if [ ".env.example" -nt ".env" ]; then
        print_warning ".env.example appears to be newer than .env"
        echo -n "Do you want to backup current .env and recreate from .env.example? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            mv .env .env.backup.$(date +%Y%m%d_%H%M%S)
            cp .env.example .env
            print_success ".env file recreated from template (backup saved)"
        fi
    fi
fi

# Step 2: Load environment variables from .env
if [ -f ".env" ]; then
    print_info "Loading environment variables from .env file..."
    # Load .env file safely by reading line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*$ ]]; then
            continue
        fi
        # Export the variable
        if [[ "$line" =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
            export "$line"
        fi
    done < .env
    print_success "Environment variables loaded"
else
    print_error "No .env file found after creation attempt"
    exit 1
fi

# Step 3: Check which variables need to be set
print_header "Environment Variables Status"

# Required variables and their descriptions
required_vars="GITHUB_PERSONAL_ACCESS_TOKEN:GitHub Personal Access Token
NOTION_API_KEY:Notion API Key
TAVILY_API_KEY:Tavily API Key
SUPABASE_ACCESS_TOKEN:Supabase Access Token"

# Optional variables and their descriptions
optional_vars="GITHUB_TOOLSETS:GitHub Toolsets
GITHUB_READ_ONLY:GitHub Read-Only Mode
DEFAULT_MINIMUM_TOKENS:Context7 Minimum Tokens"

missing_required=()
missing_optional=()

# Function to get variable description
get_var_description() {
    local var_name="$1"
    local var_list="$2"
    echo "$var_list" | while IFS=':' read -r name desc; do
        if [ "$name" = "$var_name" ]; then
            echo "$desc"
            break
        fi
    done
}

# Check required variables
echo "$required_vars" | while IFS=':' read -r var desc; do
    eval "var_value=\$$var"
    var_lower=$(echo "$var" | tr '[:upper:]' '[:lower:]')
    if [ -z "$var_value" ] || [ "$var_value" = "your_${var_lower}_here" ] || [[ "$var_value" == *"_here" ]]; then
        print_error "$desc ($var) - Not set"
        missing_required+=("$var")
    else
        # Show partial value for security
        if [ ${#var_value} -gt 10 ]; then
            display_value="${var_value:0:10}..."
        else
            display_value="$var_value"
        fi
        print_success "$desc ($var) - Set ($display_value)"
    fi
done

# Check optional variables
echo "$optional_vars" | while IFS=':' read -r var desc; do
    eval "var_value=\$$var"
    if [ -z "$var_value" ]; then
        print_warning "$desc ($var) - Not set (optional)"
        missing_optional+=("$var")
    else
        print_success "$desc ($var) - Set ($var_value)"
    fi
done

# Step 4: Interactive setup for missing required variables
# Collect missing variables into arrays first
missing_required=()
missing_optional=()

echo "$required_vars" | while IFS=':' read -r var desc; do
    eval "var_value=\$$var"
    var_lower=$(echo "$var" | tr '[:upper:]' '[:lower:]')
    if [ -z "$var_value" ] || [ "$var_value" = "your_${var_lower}_here" ] || [[ "$var_value" == *"_here" ]]; then
        echo "$var" >> /tmp/missing_required.txt
    fi
done

echo "$optional_vars" | while IFS=':' read -r var desc; do
    eval "var_value=\$$var"
    if [ -z "$var_value" ]; then
        echo "$var" >> /tmp/missing_optional.txt
    fi
done

# Read missing variables from temp files
if [ -f "/tmp/missing_required.txt" ]; then
    missing_required=($(cat /tmp/missing_required.txt))
    rm -f /tmp/missing_required.txt
fi

if [ -f "/tmp/missing_optional.txt" ]; then
    missing_optional=($(cat /tmp/missing_optional.txt))
    rm -f /tmp/missing_optional.txt
fi

if [ ${#missing_required[@]} -gt 0 ]; then
    print_header "Setting Up Missing Required Variables"
    print_info "The following required variables need to be set:"
    
    for var in "${missing_required[@]}"; do
        desc=$(get_var_description "$var" "$required_vars")
        echo "  - $desc ($var)"
    done
    
    echo -n "Do you want to set them now interactively? (y/N): "
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        for var in "${missing_required[@]}"; do
            echo ""
            desc=$(get_var_description "$var" "$required_vars")
            print_info "Setting $desc ($var)"
            
            # Provide instructions for each variable
            case $var in
                "GITHUB_PERSONAL_ACCESS_TOKEN")
                    echo "Instructions:"
                    echo "1. Go to https://github.com/settings/tokens"
                    echo "2. Click 'Generate new token (classic)'"
                    echo "3. Select appropriate scopes (repo, user, etc.)"
                    echo "4. Copy the generated token"
                    ;;
                "NOTION_API_KEY")
                    echo "Instructions:"
                    echo "1. Go to https://www.notion.so/my-integrations"
                    echo "2. Click 'New integration'"
                    echo "3. Copy the 'Internal Integration Token'"
                    ;;
                "TAVILY_API_KEY")
                    echo "Instructions:"
                    echo "1. Go to https://tavily.com/"
                    echo "2. Sign up/in and go to dashboard"
                    echo "3. Copy your API key"
                    ;;
                "SUPABASE_ACCESS_TOKEN")
                    echo "Instructions:"
                    echo "1. Go to https://app.supabase.com/"
                    echo "2. Select your project → Settings → API"
                    echo "3. Copy the 'service_role' key"
                    ;;
            esac
            
            echo -n "Enter $desc: "
            read -r value
            
            if [ -n "$value" ]; then
                # Escape special characters for sed
                escaped_value=$(printf '%s\n' "$value" | sed 's/[[\.*^$()+?{|]/\\&/g')
                
                # Update .env file
                if grep -q "^${var}=" .env; then
                    # Variable exists, update it
                    if [[ "$OSTYPE" == "darwin"* ]]; then
                        # macOS
                        sed -i '' "s/^${var}=.*/${var}=${escaped_value}/" .env
                    else
                        # Linux
                        sed -i "s/^${var}=.*/${var}=${escaped_value}/" .env
                    fi
                else
                    # Variable doesn't exist, add it
                    echo "${var}=${value}" >> .env
                fi
                
                print_success "Set ${var}"
            else
                print_warning "Skipped ${var}"
            fi
        done
    else
        print_info "Please edit .env file manually with your API keys"
        print_info "Refer to docs/MCP_ENVIRONMENT_SETUP.md for detailed instructions"
    fi
fi

# Step 5: Validate the setup
print_header "Validation"

# Source the updated .env file
while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*$ ]]; then
        continue
    fi
    # Export the variable
    if [[ "$line" =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
        export "$line"
    fi
done < .env

# Re-check required variables
all_set=true
echo "$required_vars" | while IFS=':' read -r var desc; do
    eval "var_value=\$$var"
    var_lower=$(echo "$var" | tr '[:upper:]' '[:lower:]')
    if [ -z "$var_value" ] || [ "$var_value" = "your_${var_lower}_here" ] || [[ "$var_value" == *"_here" ]]; then
        print_error "$desc ($var) - Still not set"
        echo "false" > /tmp/all_set.txt
    fi
done

if [ -f "/tmp/all_set.txt" ]; then
    all_set=false
    rm -f /tmp/all_set.txt
fi

if [ "$all_set" = true ]; then
    print_success "All required environment variables are set!"
    print_info "You can now use MCP servers in Cline and VSCode"
    print_info "Remember to restart VSCode for changes to take effect"
else
    print_warning "Some required variables are still missing"
    print_info "Please edit .env file manually or run this script again"
fi

# Step 6: Set permissions
print_header "Security"

# Set secure permissions on .env file
chmod 600 .env
print_success "Set secure permissions on .env file (600)"

# Check if .env is in .gitignore
if [ -f ".gitignore" ]; then
    if ! grep -q "^\.env$" .gitignore; then
        print_warning ".env is not in .gitignore"
        echo -n "Add .env to .gitignore? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            echo ".env" >> .gitignore
            print_success "Added .env to .gitignore"
        fi
    else
        print_success ".env is properly excluded from version control"
    fi
fi

print_header "Next Steps"
print_info "1. Restart VSCode: cmd+shift+p → 'Developer: Reload Window'"
print_info "2. Test MCP servers in VSCode Chat or Cline"
print_info "3. Run './scripts/validate-mcp-env.sh' to validate your setup"
print_info "4. See docs/MCP_ENVIRONMENT_SETUP.md for troubleshooting"

echo ""
print_success "MCP Environment setup complete!"
