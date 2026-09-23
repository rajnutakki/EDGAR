#!/usr/bin/env python3
"""sync_agents.py - Sync agent instructions from external master markdown file
to Gemini CLI, Claude Code, GitHub Copilot CLI, and OpenCode (local & global).
"""

import json
import sys
from pathlib import Path

# Gemini frontmatter
GEMINI_FRONTMATTER = """---
name: edgar-analyzer
description: Explores EDGAR run directories, retrieves run specifications, lists available runs, and inspects numpy/JAX model codes.
tools:
  - mcp_edgar_analyzer_list_runs
  - mcp_edgar_analyzer_get_run_specs
  - mcp_edgar_analyzer_get_top_models
  - mcp_edgar_analyzer_inspect_model
  - mcp_edgar_analyzer_filter_models_by_parameters
  - mcp_edgar_analyzer_compare_model_syntax_trees
mcp_servers:
  edgar_analyzer:
    command: "uv"
    args: [
      "run",
      "--with",
      "mcp",
      "python",
      "edgar_experimental/agents/output/tools/mcp_server.py"
    ]
model: gemini-2.5-flash
temperature: 0.1
max_turns: 15
---
"""

# Claude frontmatter
CLAUDE_FRONTMATTER = """---
name: "edgar-analyzer"
description: "Analyzes EDGAR run outputs — lists runs, compares model code, and inspects numpy/JAX models and parameter estimators via the edgar_analyzer MCP server. Use when the user wants to study or compare the results of an EDGAR run."
color: "blue"
---
"""

# Copilot CLI frontmatter
COPILOT_FRONTMATTER = """---
name: edgar-analyzer
description: Analyzes EDGAR run outputs — lists runs, compares model code, and inspects numpy/JAX models and parameter estimators via the edgar_analyzer MCP server. Use when the user wants to study or compare the results of an EDGAR run.
tools:
  - edgar_analyzer/*
---
"""

# OpenCode frontmatter
OPENCODE_FRONTMATTER = """---
name: edgar-analyzer
description: Analyzes EDGAR run outputs — lists runs, compares model code, and inspects numpy/JAX models and parameter estimators via the edgar_analyzer MCP server. Use when the user wants to study or compare the results of an EDGAR run.
mode: all
---
"""


def sync():
    # Resolve paths relative to repository root
    repo_root = Path(__file__).resolve().parents[3]
    instructions_path = (
        repo_root / "edgar_experimental/agents/output/edgar_analyzer_instructions.md"
    )
    mcp_server_path = repo_root / "edgar_experimental/agents/output/tools/mcp_server.py"

    gemini_path = repo_root / ".gemini/agents/edgar-analyzer.md"
    claude_path = repo_root / ".claude/agents/edgar-analyzer.md"
    copilot_path = repo_root / ".github/agents/edgar-analyzer.agent.md"
    opencode_local_path = repo_root / ".opencode/agents/edgar-analyzer.md"
    opencode_global_path = Path.home() / ".config/opencode/agents/edgar-analyzer.md"
    opencode_global_json = Path.home() / ".config/opencode/opencode.json"

    if not instructions_path.exists():
        print(f"Error: Could not find master instructions at {instructions_path}")
        sys.exit(1)

    # Read master instructions
    instructions = instructions_path.read_text().strip()

    # 1. Generate Gemini subagent file
    gemini_path.parent.mkdir(parents=True, exist_ok=True)
    gemini_path.write_text(GEMINI_FRONTMATTER + "\n" + instructions + "\n")
    print(f"Generated Gemini Subagent configuration at: {gemini_path}")

    # 2. Generate Claude subagent file
    # For Claude, replace Gemini/custom MCP tool prefix with Claude's double underscore format
    claude_instructions = instructions.replace(
        "mcp_edgar_analyzer_", "mcp__edgar_analyzer__"
    )
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.write_text(CLAUDE_FRONTMATTER + "\n" + claude_instructions + "\n")
    print(f"Generated Claude Subagent configuration at: {claude_path}")

    # 3. Generate Copilot CLI agent file
    copilot_instructions = instructions.replace(
        "mcp_edgar_analyzer_", "edgar_analyzer_"
    )
    copilot_path.parent.mkdir(parents=True, exist_ok=True)
    copilot_path.write_text(COPILOT_FRONTMATTER + "\n" + copilot_instructions + "\n")
    print(f"Generated Copilot CLI Agent configuration at: {copilot_path}")

    # 4. Generate OpenCode agent files (local repo and global config)
    opencode_instructions = instructions.replace(
        "mcp_edgar_analyzer_", "edgar_analyzer_"
    )
    opencode_local_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_local_path.write_text(
        OPENCODE_FRONTMATTER + "\n" + opencode_instructions + "\n"
    )
    print(f"Generated OpenCode local subagent at: {opencode_local_path}")

    if opencode_global_path.parent.exists() or Path.home().joinpath(".config").exists():
        opencode_global_path.parent.mkdir(parents=True, exist_ok=True)
        opencode_global_path.write_text(
            OPENCODE_FRONTMATTER + "\n" + opencode_instructions + "\n"
        )
        print(f"Generated OpenCode global subagent at: {opencode_global_path}")

    # 5. Ensure local .mcp.json is correctly configured (Claude Code)
    mcp_path = repo_root / ".mcp.json"
    expected_mcp_config = {
        "command": "uv",
        "args": [
            "run",
            "--with",
            "mcp",
            "python",
            "edgar_experimental/agents/output/tools/mcp_server.py",
        ],
    }

    mcp_data = {}
    if mcp_path.exists():
        try:
            mcp_data = json.loads(mcp_path.read_text())
            if not isinstance(mcp_data, dict):
                mcp_data = {}
        except Exception as e:
            print(
                f"Warning: Failed to parse existing .mcp.json ({e}). Re-initializing."
            )
            mcp_data = {}

    if "mcpServers" not in mcp_data or not isinstance(mcp_data["mcpServers"], dict):
        mcp_data["mcpServers"] = {}

    current_server_config = mcp_data["mcpServers"].get("edgar_analyzer")
    if current_server_config != expected_mcp_config:
        mcp_data["mcpServers"]["edgar_analyzer"] = expected_mcp_config
        mcp_path.write_text(json.dumps(mcp_data, indent=2) + "\n")
        print(f"Updated .mcp.json configuration at: {mcp_path}")
    else:
        print(".mcp.json is already configured correctly.")

    # 6. Ensure global OpenCode configuration has the edgar_analyzer MCP server
    expected_opencode_mcp = {
        "type": "local",
        "command": [
            "uv",
            "run",
            "--directory",
            str(repo_root),
            "--with",
            "mcp",
            "python",
            str(mcp_server_path),
        ],
    }

    opencode_json_data = {}
    if opencode_global_json.exists():
        try:
            opencode_json_data = json.loads(opencode_global_json.read_text())
            if not isinstance(opencode_json_data, dict):
                opencode_json_data = {}
        except Exception as e:
            print(
                f"Warning: Failed to parse existing ~/.config/opencode/opencode.json ({e}). Re-initializing."
            )
            opencode_json_data = {}

    if "mcp" not in opencode_json_data or not isinstance(
        opencode_json_data["mcp"], dict
    ):
        opencode_json_data["mcp"] = {}

    if opencode_json_data["mcp"].get("edgar_analyzer") != expected_opencode_mcp:
        opencode_json_data["mcp"]["edgar_analyzer"] = expected_opencode_mcp
        opencode_global_json.parent.mkdir(parents=True, exist_ok=True)
        opencode_global_json.write_text(json.dumps(opencode_json_data, indent=2) + "\n")
        print(f"Updated global OpenCode MCP configuration at: {opencode_global_json}")
    else:
        print("~/.config/opencode/opencode.json MCP server is already up to date.")

    print("\nAll subagents and MCP configurations successfully synced!")


if __name__ == "__main__":
    sync()
