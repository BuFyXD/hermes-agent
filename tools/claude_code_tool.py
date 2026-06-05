#!/usr/bin/env python3
"""
Claude Code Sub-Agent Tool

Spawns Claude Code in non-interactive print mode (-p) as a synchronous
sub-agent. The parent agent blocks until Claude Code completes or the
timeout is reached. Results are parsed from Claude Code's structured
JSON output.

For multi-turn interactive sessions, use the MCP bridge (mcp_claude_code_*
tools) which supports session persistence, polling, and permission handling.

Binary resolution (priority order):
  1. ``claude_code.binary_path`` in config.yaml
  2. Profile's ``scripts/cc`` wrapper (auto-detected via HERMES_HOME)
  3. ``claude`` in PATH
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CLAUDE_CODE_SCHEMA = {
    "name": "claude_code",
    "description": (
        "Delegate a coding task to Claude Code (DeepSeek backend) and wait "
        "for the result. Claude Code can read files, write code, run shell "
        "commands, and search the web — use it for complex multi-step coding "
        "tasks.\n\n"
        "This is a SYNCHRONOUS call — the parent agent pauses until Claude "
        "Code finishes or the timeout is reached. For very long tasks or "
        "multi-turn interactive sessions, the MCP-based claude_code tools "
        "(mcp_claude_code_claude_code, etc.) support polling and session "
        "persistence.\n\n"
        "Examples:\n"
        '  - claude_code(prompt="Refactor src/auth.py to use JWT tokens", '
        'allowed_tools=["Read","Edit","Bash"], max_turns=15)\n'
        '  - claude_code(prompt="Review the git diff for security issues", '
        'max_turns=5)\n'
        '  - claude_code(prompt="Write unit tests for utils.py", '
        'cwd="/path/to/project")'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The task for Claude Code. Be specific — include file "
                    "paths, requirements, constraints, and acceptance criteria. "
                    "The more precise the prompt, the better the result."
                ),
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Working directory for Claude Code. Default: current "
                    "directory."
                ),
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tools Claude Code may use. Common values: "
                    "Read, Edit, Write, Bash, WebSearch, WebFetch. "
                    "Use Bash(git *), Bash(npm *) for fine-grained control. "
                    "Empty = all default tools available."
                ),
            },
            "disallowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tools explicitly denied. E.g., ['Bash'] to prevent "
                    "command execution while allowing Read and Edit."
                ),
            },
            "max_turns": {
                "type": "integer",
                "description": (
                    "Maximum tool-use turns. Default: from config (50). "
                    "Higher = more thorough but slower and more expensive."
                ),
            },
            "effort": {
                "type": "string",
                "enum": ["low", "medium", "high", "max"],
                "description": (
                    "Reasoning effort level. 'max' for complex multi-step "
                    "tasks, 'low' for simple lookups. Default: from config."
                ),
            },
            "append_system_prompt": {
                "type": "string",
                "description": (
                    "Text appended to Claude Code's default system prompt. "
                    "Use for project-specific instructions or constraints."
                ),
            },
        },
        "required": ["prompt"],
    },
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_claude_code_config() -> dict:
    """Load the ``claude_code`` section from Hermes config."""
    try:
        from hermes_cli.config import load_config

        full = load_config() or {}
        return full.get("claude_code") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def _resolve_binary(cfg: dict) -> str | None:
    """Resolve the Claude Code binary path.

    Priority:
      1. ``claude_code.binary_path`` in config.yaml (explicit override)
      2. Profile ``scripts/cc`` wrapper (sources bashrc → sets env vars)
      3. ``claude`` in PATH (fallback)
    """
    # 1. Explicit config override
    explicit = cfg.get("binary_path")
    if explicit:
        explicit = os.path.expanduser(explicit)
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit

    # 2. Profile cc wrapper — look under HERMES_HOME/profiles/*/scripts/cc
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        profiles_dir = Path(hermes_home) / "profiles"
        if profiles_dir.is_dir():
            for profile_dir in sorted(profiles_dir.iterdir()):
                cc_path = profile_dir / "scripts" / "cc"
                if cc_path.is_file() and os.access(str(cc_path), os.X_OK):
                    return str(cc_path)

    # 3. PATH fallback
    claude_bin = shutil.which("claude")
    if claude_bin:
        return claude_bin

    return None


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


def _check_claude_code_requirements() -> bool:
    """Return True if a usable Claude Code binary is available."""
    cfg = _load_claude_code_config()
    return _resolve_binary(cfg) is not None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _claude_code_handler(
    prompt: str = "",
    cwd: str | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    max_turns: int | None = None,
    effort: str | None = None,
    append_system_prompt: str | None = None,
) -> str:
    """Execute Claude Code in print mode and return the parsed result."""
    cfg = _load_claude_code_config()

    # --- resolve binary ---------------------------------------------------
    binary = _resolve_binary(cfg)
    if not binary:
        return tool_error(
            "Claude Code binary not found. Install via: "
            "npm install -g @anthropic-ai/claude-code, or set "
            "claude_code.binary_path in config.yaml."
        )

    # --- validate prompt --------------------------------------------------
    if not prompt or not prompt.strip():
        return tool_error("prompt is required")

    # --- resolve working directory ----------------------------------------
    resolved_cwd = None
    if cwd:
        resolved_cwd = os.path.expanduser(cwd)
    if not resolved_cwd:
        resolved_cwd = cfg.get("default_cwd") or os.getcwd()
    if not os.path.isdir(resolved_cwd):
        return tool_error(f"Working directory does not exist: {resolved_cwd}")

    # --- build argument list ----------------------------------------------
    argv = [binary, "-p", prompt, "--output-format", "json"]

    # Tool restrictions
    if allowed_tools:
        argv.extend(["--allowedTools", ",".join(allowed_tools)])
    if disallowed_tools:
        argv.extend(["--disallowedTools", ",".join(disallowed_tools)])

    # Turn limit
    turns = max_turns if max_turns is not None else cfg.get("default_max_turns")
    if turns:
        argv.extend(["--max-turns", str(turns)])

    # Effort level
    eff = effort or cfg.get("default_effort")
    if eff:
        argv.extend(["--effort", eff])

    # Extra system prompt
    if append_system_prompt:
        argv.extend(["--append-system-prompt", append_system_prompt])

    # Auto-approve all tool use (the parent agent is the security boundary)
    argv.append("--dangerously-skip-permissions")

    # --- execute ----------------------------------------------------------
    timeout_seconds = cfg.get("timeout", 300)
    logger.info(
        "claude_code: executing in %s (timeout=%ds, max_turns=%s)",
        resolved_cwd,
        timeout_seconds,
        turns,
    )

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=resolved_cwd,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return tool_error(
            f"Claude Code timed out after {timeout_seconds}s. "
            "Consider using the MCP-based claude_code tools "
            "(mcp_claude_code_claude_code) for long-running tasks, "
            "or increase claude_code.timeout in config.yaml."
        )
    except FileNotFoundError:
        return tool_error(
            f"Binary not found: {binary}. "
            "Check claude_code.binary_path in config.yaml."
        )
    except OSError as exc:
        return tool_error(f"Failed to execute Claude Code: {exc}")

    # --- parse output -----------------------------------------------------
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    # Try to parse structured JSON from Claude Code
    parsed = None
    try:
        parsed = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        pass

    # Extract the text result from Claude Code's JSON response
    # Format: {"type":"result","subtype":"success","result":"...","num_turns":N,...}
    if isinstance(parsed, dict) and parsed.get("type") == "result":
        response_text = parsed.get("result", "")
        subtype = parsed.get("subtype", "unknown")
        num_turns = parsed.get("num_turns", 0)
        cost = parsed.get("total_cost_usd", 0)
        session_id = parsed.get("session_id", "")
        is_error = parsed.get("is_error", False)

        if is_error or subtype not in ("success",):
            return tool_result(
                success=False,
                output=response_text,
                subtype=subtype,
                num_turns=num_turns,
                cost_usd=cost,
                session_id=session_id,
                error=response_text if is_error else None,
            )

        return tool_result(
            success=True,
            output=response_text,
            subtype=subtype,
            num_turns=num_turns,
            cost_usd=cost,
            session_id=session_id,
        )

    # Non-JSON or unexpected format — return raw output
    if result.returncode != 0:
        error_msg = stderr or stdout or "Unknown error"
        return tool_error(
            f"Claude Code exited with code {result.returncode}: {error_msg[:500]}"
        )

    # Success but not JSON — return text directly
    return tool_result(success=True, output=stdout, returncode=0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="claude_code",
    toolset="claude-code",
    schema=CLAUDE_CODE_SCHEMA,
    handler=lambda args, **kw: _claude_code_handler(
        prompt=args.get("prompt", ""),
        cwd=args.get("cwd"),
        allowed_tools=args.get("allowed_tools"),
        disallowed_tools=args.get("disallowed_tools"),
        max_turns=args.get("max_turns"),
        effort=args.get("effort"),
        append_system_prompt=args.get("append_system_prompt"),
    ),
    check_fn=_check_claude_code_requirements,
    emoji="🤖",
)
