"""Subprocess-based Python interpreter for ARC-AGI code execution.

Simplified port from rlvr's SubprocessInterpreter. Removed DSPy LM support
and tool registration/IPC — only keeps the core subprocess lifecycle,
JSON communication, and execution with SUBMIT/RESET_HISTORY signals.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


class CodeInterpreterError(Exception):
    """Error from the code interpreter."""
    pass


class HistoryReset:
    """Signal from interpreter that the agent wants to reset REPL history."""

    def __init__(self, summary: str):
        self.summary = summary


class FinalOutput:
    """Returned by interpreter.execute() when SUBMIT() is called.

    This signals that the code execution loop should terminate and return
    the contained output to the caller.
    """

    def __init__(self, output: Any):
        self.output = output

    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


# ---------------------------------------------------------------------------
# REPL loop script (runs inside the subprocess)
# ---------------------------------------------------------------------------

_REPL_SCRIPT = r'''
import json
import sys
import io
import traceback

# ---- Signal exceptions ----

class _FinalOutputSignal(Exception):
    def __init__(self, data):
        self.data = data

def _make_json_serializable(obj):
    """Recursively convert numpy arrays and other non-JSON types to serializable forms."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj

class _ResetHistorySignal(Exception):
    def __init__(self, summary):
        self.summary = summary

# ---- Globals for the sandbox ----

def SUBMIT(**kwargs):
    """Submit final output and end the REPL session."""
    raise _FinalOutputSignal(kwargs)

def RESET_HISTORY(summary: str):
    """Compact REPL history into a summary to reduce noise (variables persist)."""
    raise _ResetHistorySignal(summary)

# ---- REPL loop ----

namespace = {"SUBMIT": SUBMIT, "RESET_HISTORY": RESET_HISTORY, "__builtins__": __builtins__}

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def recv():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

# Process initial config
config = recv()

# Execute sandbox code if provided
_sandbox_error = None
if config.get("sandbox_code"):
    try:
        exec(config["sandbox_code"], namespace)
    except Exception as e:
        _sandbox_error = f"{type(e).__name__}: {e}"

# Signal ready (include sandbox error if any, so caller can log it)
send({"ready": True, **({"sandbox_error": _sandbox_error} if _sandbox_error else {})})

# Main REPL loop
while True:
    try:
        msg = recv()
    except (json.JSONDecodeError, EOFError):
        break

    if msg.get("shutdown"):
        break

    code = msg.get("code", "")

    # Capture stdout
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        exec(code, namespace)
        sys.stdout = old_stdout
        output = captured.getvalue()
        send({"output": output})

    except _FinalOutputSignal as e:
        sys.stdout = old_stdout
        captured_output = captured.getvalue()
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = _make_json_serializable(e.data)
        send({"error": "FinalOutput", "errorType": "FinalOutput", "errorArgs": [serializable_data], "output": captured_output})

    except _ResetHistorySignal as e:
        sys.stdout = old_stdout
        send({"type": "reset_history", "summary": e.summary})

    except SyntaxError as e:
        sys.stdout = old_stdout
        send({"error": str(e), "errorType": "SyntaxError"})

    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        send({"error": tb, "errorType": type(e).__name__})
'''


# ---------------------------------------------------------------------------
# SubprocessInterpreter
# ---------------------------------------------------------------------------


class SubprocessInterpreter:
    """CodeInterpreter that runs a persistent CPython subprocess.

    State persists across execute() calls. Communication is JSON-RPC
    over stdin/stdout.
    """

    def __init__(
        self,
        sandbox_code: str | None = None,
        timeout: float = 30.0,
    ):
        self._sandbox_code = sandbox_code
        self._timeout = timeout
        self._process: subprocess.Popen | None = None
        self._started = False

    def start(self) -> None:
        """Start the subprocess and configure it."""
        if self._started and self._process and self._process.poll() is None:
            return

        env = os.environ.copy()

        self._process = subprocess.Popen(
            [sys.executable, "-c", _REPL_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

        config: dict[str, Any] = {}
        if self._sandbox_code:
            config["sandbox_code"] = self._sandbox_code

        self._send(config)

        response = self._recv()
        if not response.get("ready"):
            raise CodeInterpreterError(f"Subprocess failed to start: {response}")
        if response.get("sandbox_error"):
            raise CodeInterpreterError(f"Sandbox initialization failed: {response['sandbox_error']}")

        self._started = True

    def shutdown(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._send({"shutdown": True})
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._started = False

    def execute(self, code: str) -> Any:
        """Execute Python code in the subprocess.

        Returns:
            - (FinalOutput, captured_output) if SUBMIT() was called
            - HistoryReset if RESET_HISTORY() was called
            - str with captured stdout otherwise

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On syntax errors
        """
        if not self._started:
            self.start()

        self._send({"code": code})

        result = self._recv()

        # Handle RESET_HISTORY signal
        if result.get("type") == "reset_history":
            return HistoryReset(result.get("summary", ""))

        # Handle errors
        if "error" in result:
            error_msg = result["error"]
            error_type = result.get("errorType", "RuntimeError")
            error_args = result.get("errorArgs", [])

            if error_type == "FinalOutput":
                final_data = error_args[0] if error_args else None
                captured_output = result.get("output", "")
                return FinalOutput(final_data), captured_output
            elif error_type == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax: {error_msg}")
            else:
                raise CodeInterpreterError(f"{error_type}: {error_msg}")

        # Normal output
        return result.get("output", "")

    # ---- Internal helpers ----

    def _send(self, msg: dict) -> None:
        """Send JSON message to subprocess stdin."""
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    pass
            exit_code = self._process.returncode if self._process else None
            raise CodeInterpreterError(
                f"Subprocess is not running (exit code: {exit_code})" + (f". Stderr: {stderr}" if stderr else "")
            )
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(msg) + "\n")
        self._process.stdin.flush()

    def _recv(self, timeout: float | None = None) -> dict:
        """Read JSON message from subprocess stdout."""
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read()
            raise CodeInterpreterError(f"Subprocess is not running. Stderr: {stderr}")

        assert self._process.stdout is not None

        timeout_secs = timeout if timeout is not None else self._timeout
        if timeout_secs:
            sel = selectors.DefaultSelector()
            sel.register(self._process.stdout, selectors.EVENT_READ)
            try:
                ready = sel.select(timeout=timeout_secs)
                if not ready:
                    self._process.kill()
                    self._process.wait()
                    raise CodeInterpreterError(f"Code execution timed out after {timeout_secs} seconds")
            finally:
                sel.close()

        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise CodeInterpreterError(f"No output from subprocess. Stderr: {stderr}")

        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise CodeInterpreterError(f"Invalid JSON from subprocess: {line.strip()!r}") from e

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
