#!/usr/bin/env bash

# MCP Environment Validation Script
# This script validates that all environment variables are properly set for MCP servers
# POSIX-compliant and compatible with older bash versions (e.g., macOS default)

# Stop on first error
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

# --- Associative Array Emulation for POSIX/Bash 3.2 Compatibility ---

# Simulates: token_patterns[$1]
get_token_pattern() {
    case "$1" in
        "GITHUB_PERSONAL_ACCESS_TOKEN") echo "^ghp_[a-zA-Z0-9]{36}$" ;;
        "NOTION_API_KEY") echo "^ntn_[a-zA-Z0-9]+$" ;;
        "TAVILY_API_KEY") echo "^tvly-dev-[a-zA-Z0-9]+$" ;;
        "SUPABASE_ACCESS_TOKEN") echo "^sbp_[a-zA-Z0-9]+$" ;;
        *) echo "" ;;
    esac
}

# Simulates: required_vars[$1]
get_required_var_name() {
    case "$1" in
        "GITHUB_PERSONAL_ACCESS_TOKEN") echo "GitHub Personal Access Token" ;;
        "NOTION_API_KEY") echo "Notion API Key" ;;
        "TAVILY_API_KEY") echo "Tavily API Key" ;;
        "SUPABASE_ACCESS_TOKEN") echo "Supabase Access Token" ;;
        *) echo "" ;;
    esac
}

# Simulates: optional_vars[$1]
get_optional_var_name() {
    case "$1" in
        "GITHUB_TOOLSETS") echo "GitHub Toolsets" ;;
        "GITHUB_READ_ONLY") echo "GitHub Read-Only Mode" ;;
        "DEFAULT_MINIMUM_TOKENS") echo "Context7 Minimum Tokens" ;;
        *) echo "" ;;
    esac
}

# --- Main Script ---

# Check if we're in the right directory
if [ ! -f ".env.example" ]; then
    print_error "Error: .env.example not found. Please run this script from the project root."
    exit 1
fi

print_header "MCP Environment Validation"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found"
    print_info "Run './scripts/setup-mcp-env.sh' to create it"
    exit 1
fi

# Load environment variables safely
if [ -f .env ]; then
    # Use 'set -a' to export all variables defined in the .env file
    set -a
    # shellcheck source=.env
    . .env
    set +a
fi

# Define required and optional variables as simple arrays of keys
required_vars_keys=(
    "GITHUB_PERSONAL_ACCESS_TOKEN"
    "NOTION_API_KEY"
    "TAVILY_API_KEY"
    "SUPABASE_ACCESS_TOKEN"
)

optional_vars_keys=(
    "GITHUB_TOOLSETS"
    "GITHUB_READ_ONLY"
    "DEFAULT_MINIMUM_TOKENS"
)

# Validation counters
valid_count=0
invalid_count=0
missing_count=0
warning_count=0

print_header "Required Environment Variables"

# Validate required variables
for var in "${required_vars_keys[@]}"; do
    # Use eval to get the value of the variable whose name is in 'var'
    eval "value=\"\$$var\""
    var_name=$(get_required_var_name "$var")
    
    if [ -z "$value" ] || echo "$value" | grep -q "_here$"; then
        print_error "${var_name} (${var}) - Missing or placeholder"
        missing_count=$((missing_count + 1))
    else
        pattern=$(get_token_pattern "$var")
        # Check if pattern exists for this variable
        if [ -n "$pattern" ]; then
            # Use grep for POSIX-compliant regex matching
            if echo "$value" | grep -Eq "$pattern"; then
                # Show partial value for security
                if [ ${#value} -gt 20 ]; then
                    display_value="${value%"${value#??????????}"}...${value##*????}"
                else
                    display_value="${value%"${value#??????????}"}..."
                fi
                print_success "${var_name} (${var}) - Valid format ($display_value)"
                valid_count=$((valid_count + 1))
            else
                print_error "${var_name} (${var}) - Invalid format"
                print_info "Expected pattern: $pattern"
                invalid_count=$((invalid_count + 1))
            fi
        else
            # No pattern to validate against, just check if not empty
            print_success "${var_name} (${var}) - Set"
            valid_count=$((valid_count + 1))
        fi
    fi
done

print_header "Optional Environment Variables"

# Validate optional variables
for var in "${optional_vars_keys[@]}"; do
    eval "value=\"\$$var\""
    var_name=$(get_optional_var_name "$var")
    
    if [ -z "$value" ]; then
        print_warning "${var_name} (${var}) - Not set (optional)"
        warning_count=$((warning_count + 1))
    else
        print_success "${var_name} (${var}) - Set ($value)"
    fi
done

# Validate file permissions
print_header "Security Validation"

# Check .env file permissions
if [ -f ".env" ]; then
    # Try different methods to get file permissions
    if command -v stat >/dev/null 2>&1; then
        # macOS stat format
        env_perms=$(stat -f "%A" .env 2>/dev/null | head -1)
        # If macOS format failed, try Linux format
        if [ -z "$env_perms" ] || [ "$env_perms" = "" ]; then
            env_perms=$(stat -c "%a" .env 2>/dev/null | head -1)
        fi
        # Clean up any extra output
        env_perms=$(echo "$env_perms" | tr -d '\n' | grep -o '^[0-9]*' || echo "unknown")
    else
        env_perms="unknown"
    fi
    
    if [ "$env_perms" = "600" ]; then
        print_success ".env file has secure permissions (600)"
    elif [ "$env_perms" = "unknown" ] || [ -z "$env_perms" ]; then
        print_warning ".env file permissions could not be determined"
    else
        print_error ".env file has insecure permissions ($env_perms)"
        print_info "Run 'chmod 600 .env' to fix"
        invalid_count=$((invalid_count + 1))
    fi
else
    print_warning ".env file not found for permission check."
fi

# Check if .env is in .gitignore
if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        print_success ".env is properly excluded from version control"
    else
        print_error ".env is not in .gitignore - secrets might be committed!"
        print_info "Add '.env' to .gitignore"
        invalid_count=$((invalid_count + 1))
    fi
else
    print_warning ".gitignore file not found"
    warning_count=$((warning_count + 1))
fi

# Check MCP configuration files
print_header "MCP Configuration Validation"

# Check if Cline MCP config exists and uses environment variables
if [ -f ".cline/mcp.json" ]; then
    if grep -q "\${" .cline/mcp.json; then
        print_success "Cline MCP config uses environment variables"
    else
        print_warning "Cline MCP config may contain hardcoded secrets"
        warning_count=$((warning_count + 1))
    fi
else
    print_warning "Cline MCP config not found (.cline/mcp.json)"
    warning_count=$((warning_count + 1))
fi

# Test environment variable expansion
print_header "Environment Variable Expansion Test"

expansion_failures=0
for var in "${required_vars_keys[@]}"; do
    eval "var_value=\"\$$var\""
    if [ -n "$var_value" ]; then
        # Simple test - check if variable expands to non-empty value
        eval "expanded=\"\$$var\""
        if [ "$expanded" = "$var_value" ]; then
            print_success "${var} expands correctly"
        else
            print_error "${var} expansion failed"
            expansion_failures=$((expansion_failures + 1))
        fi
    else
        print_warning "${var} is empty, skipping expansion test"
    fi
done

# Summary
print_header "Validation Summary"

total_required=${#required_vars_keys[@]}
echo "Required variables: $total_required"
echo -e "  ${GREEN}✅ Valid: $valid_count${NC}"
invalid_and_missing=$((invalid_count + missing_count))
if [ "$invalid_and_missing" -gt 0 ]; then
    echo -e "  ${RED}❌ Invalid/Missing: $invalid_and_missing${NC}"
else
    echo -e "  ${GREEN}❌ Invalid/Missing: 0${NC}"
fi
echo ""
echo "Optional variables: ${#optional_vars_keys[@]}"
if [ "$warning_count" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠️  Warnings: $warning_count${NC}"
else
    echo -e "  ${GREEN}⚠️  Warnings: 0${NC}"
fi
echo ""

# Determine exit status
if [ $missing_count -gt 0 ] || [ $invalid_count -gt 0 ] || [ $expansion_failures -gt 0 ]; then
    print_error "Validation failed!"
    echo ""
    print_info "Next steps:"
    if [ $missing_count -gt 0 ]; then
        print_info "1. Run './scripts/setup-mcp-env.sh' to set missing variables"
    fi
    if [ $invalid_count -gt 0 ]; then
        print_info "2. Check token formats and regenerate if necessary"
    fi
    print_info "3. See docs/MCP_ENVIRONMENT_SETUP.md for detailed instructions"
    exit 1
else
    print_success "All validations passed!"
    echo ""
    print_info "Your MCP environment is properly configured"
    print_info "You can now use MCP servers in Cline and VSCode"
    
    if [ $warning_count -gt 0 ]; then
        echo ""
        print_warning "Note: $warning_count warnings found (non-critical)"
    fi
    
    exit 0
fi
