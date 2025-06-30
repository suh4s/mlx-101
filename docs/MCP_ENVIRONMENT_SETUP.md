# MCP Environment Variables Setup Guide

This guide provides clear instructions for setting up environment variables required by MCP server configurations.

## Quick Setup

1. **Copy the template file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your actual API keys (see below for instructions)

3. **Run the setup script:**
   ```bash
   ./scripts/setup-mcp-env.sh
   ```

## Required Environment Variables

### 🔧 GitHub Configuration
```bash
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
GITHUB_TOOLSETS=                    # Optional: specify toolsets (e.g., "repos,issues")
GITHUB_READ_ONLY=                   # Optional: set to "1" for read-only mode
```

### 📝 Notion Configuration
```bash
NOTION_API_KEY=your_notion_api_key_here
```

### 🔍 Tavily Search Configuration
```bash
TAVILY_API_KEY=your_tavily_api_key_here
```

### 📚 Context7 Configuration
```bash
DEFAULT_MINIMUM_TOKENS=             # Optional: minimum token threshold
```

### 🗄️ Supabase Configuration
```bash
SUPABASE_ACCESS_TOKEN=your_supabase_access_token_here
```

## How to Obtain API Keys

### GitHub Personal Access Token

1. **Navigate to GitHub Settings:**
   - Go to https://github.com/settings/tokens
   - Or: Profile → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Create New Token:**
   - Click "Generate new token (classic)"
   - Give it a descriptive name (e.g., "MCP Server Access")
   - Set expiration as needed

3. **Select Scopes:**
   - For full functionality: `repo`, `user`, `admin:org`
   - For read-only: `repo:status`, `public_repo`
   - For issues/PRs: `repo`, `user:email`

4. **Copy Token:**
   - Copy the generated token immediately
   - Add to `.env` file: `GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxx`

### Notion Integration Token

1. **Create Integration:**
   - Go to https://www.notion.so/my-integrations
   - Click "New integration"

2. **Configure Integration:**
   - Name: "MCP Server Integration"
   - Associated workspace: Select your workspace
   - Type: Internal

3. **Get Token:**
   - Copy the "Internal Integration Token"
   - Add to `.env` file: `NOTION_API_KEY=secret_xxxxxxxxxxxx`

4. **Share Pages:**
   - Go to the Notion pages you want to access
   - Click "Share" → "Invite" → Select your integration

### Tavily API Key

1. **Sign Up:**
   - Go to https://tavily.com/
   - Create account or sign in

2. **Get API Key:**
   - Navigate to Dashboard/API section
   - Copy your API key
   - Add to `.env` file: `TAVILY_API_KEY=tvly-xxxxxxxxxx`

### Supabase Access Token

1. **Open Project:**
   - Go to https://app.supabase.com/
   - Select your project

2. **Navigate to API Settings:**
   - Go to Settings → API
   - Find "Project API keys"

3. **Copy Service Role Key:**
   - Copy the `service_role` key (not the `anon` key)
   - Add to `.env` file: `SUPABASE_ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## Environment Validation

### Check if Variables are Set
```bash
# Run the validation script
./scripts/validate-mcp-env.sh

# Or check manually
source .env
echo "GitHub Token: ${GITHUB_PERSONAL_ACCESS_TOKEN:0:10}..."
echo "Notion Key: ${NOTION_API_KEY:0:10}..."
echo "Tavily Key: ${TAVILY_API_KEY:0:10}..."
echo "Supabase Token: ${SUPABASE_ACCESS_TOKEN:0:10}..."
```

### Test MCP Servers
```bash
# Test if environment variables are properly loaded
npm run test-mcp-env
```

## Security Best Practices

### ✅ Do's
- Keep `.env` file in project root
- Use the provided `.env.example` as template
- Regularly rotate API keys
- Use read-only tokens when possible
- Store backup of tokens in secure password manager

### ❌ Don'ts
- Never commit `.env` file to version control
- Don't share tokens in chat/email
- Don't use production tokens for development
- Don't store tokens in code comments
- Don't use overly broad token permissions

## Troubleshooting

### Environment Variables Not Loading
1. **Check file location:** `.env` must be in project root
2. **Check file format:** No spaces around `=` sign
3. **Check permissions:** File should be readable
4. **Restart applications:** VSCode, Cline, terminal sessions

### API Keys Not Working
1. **Verify token format:** Each service has different token formats
2. **Check permissions:** Ensure tokens have required scopes
3. **Test tokens directly:** Use service's API testing tools
4. **Check expiration:** Some tokens expire and need renewal

### MCP Servers Not Connecting
1. **Validate environment:** Run `./scripts/validate-mcp-env.sh`
2. **Check network:** Ensure internet connectivity
3. **Review logs:** Check VSCode Developer Console
4. **Restart services:** Restart VSCode and Cline

## Advanced Configuration

### Multiple Environments
```bash
# Development
.env.development

# Production  
.env.production

# Load specific environment
export ENV=development
./scripts/load-env.sh $ENV
```

### Conditional Loading
```bash
# Load only if file exists
[ -f .env ] && source .env

# Load with fallback
source .env.example
[ -f .env ] && source .env
```

## Support

If you encounter issues:
1. Check this guide first
2. Validate your environment setup
3. Review the main MCP configuration documentation
4. Check service-specific documentation for API key issues

---

**Next Steps:** After setting up your environment variables, refer to the main [MCP Configuration Summary](./MCP_CONFIGURATION_SUMMARY.md) for usage instructions.
