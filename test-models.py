#!/usr/bin/env python3
"""Test Ollama models for code generation capability."""

import json
import subprocess
import sys

PROMPT = """Write a simple Python hello world program. Output ONLY the code, no explanation."""


def test_model(model: str) -> dict:
    """Test a single model."""
    result = {
        "model": model,
        "success": False,
        "output": "",
        "error": None,
        "has_print": False,
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
                        "stream": False,
                        "options": {"num_predict": 200},
                    }
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if proc.returncode != 0:
            result["error"] = f"curl failed: {proc.stderr}"
            return result

        data = json.loads(proc.stdout)
        output = data.get("message", {}).get("content", "").strip()
        result["output"] = output

        # Check if it generated valid-looking Python code
        if "print(" in output and "hello" in output.lower():
            result["success"] = True
            result["has_print"] = True
        elif "def " in output and "hello" in output.lower():
            result["success"] = True

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

    # Sort by size (smallest first)
    models_with_size.sort(key=lambda x: x[1])
    models = [m[0] for m in models_with_size]

    print(f"Testing {len(models)} models (smallest to largest)...")
    print("-" * 60)

    success_count = 0
    for model in models:
        result = test_model(model)
        status = "PASS" if result["success"] else "FAIL"
        print(f"{status:4} {model}")

        if result["success"]:
            success_count += 1
            # Show snippet
            snippet = result["output"][:80].replace("\n", " ")
            print(f"      → {snippet}...")

        if result["error"]:
            print(f"      → error: {result['error']}")

    print("-" * 60)
    print(f"Success: {success_count}/{len(models)}")


if __name__ == "__main__":
    import sys

    sys.stdout = sys.stderr
    main()
