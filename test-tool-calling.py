#!/usr/bin/env python3
"""Test Ollama models for tool calling WITH tool definitions (like OpenClaw does)."""

import json
import subprocess
import sys

# This is a simplified version of what OpenClaw sends - tool definitions + message
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "sessions_spawn",
            "description": "Spawn a subagent session",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task for subagent"},
                    "label": {"type": "string", "description": "Label for subagent"},
                },
                "required": ["task"],
            },
        },
    }
]

PROMPT = """Use sessions_spawn to create a subagent with task='say hello' and label='test'."""


def test_model(model: str) -> dict:
    """Test a single model with tool definitions."""
    result = {
        "model": model,
        "success": False,
        "output": "",
        "error": None,
        "has_tool_call": False,
    }

    try:
        proc = subprocess.run(
            [
                "curl",
                "-s",
                "http://localhost:11434/api/chat",
                "-d",
                json.dumps(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": PROMPT}],
                        "tools": TOOL_DEFINITIONS,
                        "stream": False,
                        "options": {"num_predict": 500},
                    }
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            result["error"] = f"curl failed: {proc.stderr}"
            return result

        data = json.loads(proc.stdout)
        message = data.get("message", {})

        # Check for structured tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            result["success"] = True
            result["has_tool_call"] = True
            result["output"] = f"tool_calls: {json.dumps(tool_calls)}"

        # Also check if it's in content (text format)
        content = message.get("content", "")
        if "sessions_spawn" in content and "tool_call" not in result["output"].lower():
            result["output"] = f"text (not tool_calls): {content[:200]}"

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except json.JSONDecodeError as e:
        result["error"] = f"json error: {e}"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    # Get models with sizes
    proc = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/tags"], capture_output=True, text=True
    )
    data = json.loads(proc.stdout)
    models_with_size = [(m["name"], m.get("size", 0)) for m in data.get("models", [])]

    # Sort by size (smallest first) - test top 15
    models_with_size.sort(key=lambda x: x[1])
    models = [m[0] for m in models_with_size[:15]]

    print(f"Testing {len(models)} models for tool calling WITH tool definitions...")
    print("-" * 60)

    success_count = 0
    for model in models:
        result = test_model(model)
        status = "PASS" if result["success"] else "FAIL"
        print(f"{status:4} {model}")

        if result["success"]:
            success_count += 1
            print(f"      → {result['output'][:100]}")

        if result["error"]:
            print(f"      → error: {result['error']}")

    print("-" * 60)
    print(f"Success: {success_count}/{len(models)}")


if __name__ == "__main__":
    main()
