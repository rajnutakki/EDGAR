# EDGAR Output Run Analyzer Agent Setup

This folder contains a custom **Model Context Protocol (MCP)** server and a platform-agnostic subagent system designed to allow researchers to query, summarize, and retrieve NumPy model codes from EDGAR evolutionary runs using natural language.

---

## Folder Architecture

- **`tools/mcp_server.py`**: A lightweight Python FastMCP server defining the programmatic analysis tools. It interfaces directly with EDGAR's internal `Population`, `Program`, `read_status`, and `read_metrics` modules.
- **`edgar_analyzer_instructions.md`**: **The Single Source of Truth** containing the core persona and scientific analysis instructions for the agent.
- **`sync_agents.py`**: A synchronization script that loads the master instructions and generates CLI-specific subagent configuration files and MCP server definitions.

---

## Setup Instructions

### 1. Synchronization Utility
Run the synchronization script from the repository root:
```bash
uv run python edgar_experimental/agents/output/sync_agents.py
```
This automatically builds and updates:
1. **`~/.config/opencode/opencode.json`** & **`~/.config/opencode/agents/edgar-analyzer.md`** (OpenCode Global - accessible across any project)
2. **`.opencode/agents/edgar-analyzer.md`** (OpenCode Local)
3. **`.claude/agents/edgar-analyzer.md`** & **`.mcp.json`** (Claude Code CLI)
4. **`.gemini/agents/edgar-analyzer.md`** (Gemini CLI)
5. **`.github/agents/edgar-analyzer.agent.md`** (GitHub Copilot CLI)

---

### 2. OpenCode Setup (Global Across Projects)
Because `sync_agents.py` registers the MCP server in `~/.config/opencode/opencode.json` with an absolute path and `--directory` flag pointing to this repository, the `edgar-analyzer` agent is available in **any project or folder**:

Start OpenCode in any terminal / directory:
```bash
opencode
```
And prompt or delegate to `edgar-analyzer`:
> **You:** "Ask edgar-analyzer to list runs and show the top model from 06-15/17-18-43."

Or switch directly to the agent in OpenCode:
```
/agent edgar-analyzer
```

---

### 3. Gemini CLI Subagent Setup
The Gemini CLI subagent utilizes an **inline** MCP server definition in its generated frontmatter.

No registration commands are required! Simply start your Gemini CLI session in the repository:
```bash
gemini
```
And ask questions using the `@` prefix:
```bash
@edgar-analyzer "list the available runs"
```

---

### 4. Claude Code Setup
The MCP server definition is provided in `.mcp.json` in the repo root, automatically configured by `sync_agents.py`.

Start Claude Code in your terminal in the repository:
```bash
claude
```
The custom `.claude/agents/edgar-analyzer.md` subagent will automatically be loaded.
You can manually activate it using the `/agents` command, or delegate tasks:
> **You:** "Ask the edgar-analyzer subagent to summarize the specs of the run `06-15/17-18-43` and show me the top model."

---

## Maintenance & Updating Instructions

If you need to change the system prompt, persona, or logical rules of the analyzer subagent:
1. Edit **`edgar_experimental/agents/output/edgar_analyzer_instructions.md`**.
2. Run the sync command:
   ```bash
   uv run python edgar_experimental/agents/output/sync_agents.py
   ```
All agent definitions and MCP server configurations across all supported platforms will instantly update in lockstep.