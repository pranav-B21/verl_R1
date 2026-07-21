# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Best-effort repair of malformed ``<tool_call>`` JSON emitted by the policy.

Motivation (measured on 3,058 real tool calls from 3 saved Amazon decodes):
2.13% (65) of ``<tool_call>`` bodies fail ``json.loads``, and 97% of those (63/65)
are then discarded by the rollout loop, so the search tool never fires and the turn
retrieves nothing at all.

The dominant cause is quoting collision on apostrophe-leading album titles. Asked to
name items inside a JSON string, the model wraps titles in single quotes, but for
titles like ``'70s Gold`` or ``'68 Comeback Special`` the leading apostrophe collides
with its own convention and it emits a literal ``"`` inside an already-quoted string::

    {"name": "search", "arguments": {"query_list": ["... played '60's Classics', "'70s Gold', ..."]}}
                                                                                 ^ terminates the string early

Observed breakdown: 48 ``Expecting ',' delimiter`` (stray quote), 12 ``Invalid \\escape``,
5 ``Expecting value``.

This module repairs the quoting/escaping cases only. It recovers 55.4% of the real
failures (36/65, i.e. 2.13% -> 0.95% residual) with **zero** semantic change across a
947-call regression corpus that already parsed. The residual 29 are garbled in ways no
safe generic repair can address (a stray ``?`` before ``]``, doubled ``]]``, mismatched
brackets) — recovering those would mean guessing at the model's intent, so they are
deliberately left to the caller's existing fallback.

Contract: never raises, and never touches input that already parses.
"""

import json
import re
from typing import Any, Optional

__all__ = ["repair_json_text", "repair_tool_call_content"]

# Characters that may legally follow a backslash inside a JSON string.
_VALID_ESCAPES = frozenset('"\\/bfnrtu')

# Characters that may legally follow a string's closing quote (after whitespace).
_STRUCTURAL = frozenset(',:]}')

_TOOL_CALL_BLOCK = re.compile(r"(<tool_call>)(.*?)(</tool_call>)", re.S)


def _escape_stray_quotes(s: str) -> str:
    """Escape quotes/backslashes that appear literally inside a JSON string.

    A ``"`` seen while inside a string is a genuine terminator only when the next
    non-whitespace character is structural (``,`` ``:`` ``]`` ``}``) or end-of-input;
    otherwise the model meant a literal quote, so escape it. A backslash that begins
    no valid JSON escape is likewise escaped rather than dropped.

    This is intentionally conservative: it can only *add* escapes inside strings, so
    text that already parses is returned byte-identical.
    """
    out = []
    i, n = 0, len(s)
    in_string = False
    while i < n:
        c = s[i]
        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
            i += 1
            continue

        # --- inside a string literal ---
        if c == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in _VALID_ESCAPES:
                out.append(c)
                out.append(nxt)
                i += 2
            else:
                out.append("\\\\")  # lone backslash -> escape it
                i += 1
            continue

        if c == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] in _STRUCTURAL:
                out.append(c)  # legitimate terminator
                in_string = False
            else:
                out.append('\\"')  # stray literal quote -> escape
            i += 1
            continue

        out.append(c)
        i += 1
    return "".join(out)


def repair_json_text(text: str) -> Optional[Any]:
    """Parse ``text`` as JSON, repairing model quoting errors if needed.

    Returns the parsed object, or None if it is unparseable even after repair.
    Never raises.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.loads(_escape_stray_quotes(text))
    except Exception:
        return None


def repair_tool_call_content(content: str) -> Optional[str]:
    """Rewrite every repairable ``<tool_call>`` body in ``content``.

    Returns the repaired content, or None if nothing could be repaired (in which
    case the caller should keep its existing failure behaviour). Blocks that already
    parse are left untouched.
    """
    if not content:
        return None

    repaired_any = False

    def _sub(m: re.Match) -> str:
        nonlocal repaired_any
        open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
        try:
            json.loads(body.strip())
            return m.group(0)  # already valid -> leave exactly as-is
        except (json.JSONDecodeError, TypeError):
            pass
        stripped = body.strip()
        fixed = _escape_stray_quotes(stripped)
        try:
            json.loads(fixed)
        except Exception:
            return m.group(0)  # unrepairable -> leave for the caller's fallback
        repaired_any = True
        # Preserve the body's original leading/trailing whitespace. sglang's Qwen
        # detector uses "\n</tool_call>" as its end-of-tool-call token, so the block
        # must keep the newline between the JSON and the closing tag or the repaired
        # content will not re-parse (the detector's ``<tool_call>(.*?)\n</tool_call>``
        # pattern would no longer match). Only the JSON interior is rewritten.
        lead = body[: len(body) - len(body.lstrip())]
        trail = body[len(body.rstrip()):]
        return f"{open_tag}{lead}{fixed}{trail}{close_tag}"

    new_content = _TOOL_CALL_BLOCK.sub(_sub, content)
    return new_content if repaired_any else None
