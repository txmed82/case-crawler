"""Shared helper for invoking external subprocess backends.

The imaging, time-series, and clinical-text generators all support an
"external command" backend: spawn a subprocess, pipe a JSON payload to
stdin, read JSON from stdout. The three implementations were near
identical — same try/except shape, same timeout/CalledProcessError/OSError
fan-out, same RuntimeError messages with a label swapped in. This module
collapses them into one helper so a fix to e.g. error formatting only has
to land in one place.
"""

from __future__ import annotations

import subprocess


def run_external_command(
    command: list[str],
    payload: str,
    *,
    backend_label: str,
    timeout_seconds: float,
) -> str:
    """Run ``command`` with ``payload`` on stdin and return stdout.

    Wraps the standard ``subprocess.run(..., check=True)`` call with the
    failure modes the generators care about:

    - ``TimeoutExpired`` -> ``RuntimeError`` with the timeout in seconds
    - ``CalledProcessError`` -> ``RuntimeError`` with returncode and
      captured stdout/stderr (so the caller can surface what the backend
      printed before failing)
    - ``OSError`` (e.g. binary not on ``$PATH``) -> ``RuntimeError``

    All three fold into a single ``RuntimeError`` so callers only have
    to catch one type.
    """

    try:
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"External {backend_label} backend timed out after "
            f"{timeout_seconds:.0f}s: {command!r}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"External {backend_label} backend failed with exit code "
            f"{exc.returncode}: {command!r}. "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"External {backend_label} backend could not be executed: {command!r}."
        ) from exc
    return result.stdout
