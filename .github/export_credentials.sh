# Export opencode v2 credentials from local machine to GitHub secrets for use in workflows.
set -o pipefail
opencode auth export |
  gh secret set OPENCODE_CREDENTIALS_JSON --repo rajnutakki/EDGAR
