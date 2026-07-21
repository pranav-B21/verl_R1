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

"""CPU regression tests for the malformed-tool-call JSON repair.

These lock in the two contracts the rollout loop relies on:

1. Repairing a malformed ``<tool_call>`` body preserves the surrounding
   whitespace (notably the ``\\n`` before ``</tool_call>``) so sglang's Qwen
   detector, whose end-of-tool-call token is ``\\n</tool_call>``, still matches
   the repaired block. Without this, a repaired call re-parses to zero tool
   calls even though the JSON is valid.
2. Content that already parses is returned byte-identical (no-op), so the
   repair can never regress a call that parses today.
"""

import json

from verl.utils.tool_call_repair import repair_json_text, repair_tool_call_content

# The exact stray-quote collision documented in tool_call_repair.py: the model
# emits a literal `"` mid-string (before '70s), terminating the string early.
MALFORMED_BODY = (
    '{"name": "search", "arguments": {"query_list": '
    '["a user played \'60\'s Classics\', "\'70s Gold\', \'Abbey Road\'"]}}'
)
MALFORMED_CONTENT = f"<tool_call>\n{MALFORMED_BODY}\n</tool_call>"

VALID_BODY = '{"name": "search", "arguments": {"query_list": ["jazz like Kind of Blue"]}}'
VALID_CONTENT = f"<tool_call>\n{VALID_BODY}\n</tool_call>"


def test_malformed_body_is_actually_malformed():
    # Guard against the fixture silently becoming valid JSON.
    try:
        json.loads(MALFORMED_BODY)
        raise AssertionError("fixture is no longer malformed JSON")
    except json.JSONDecodeError:
        pass


def test_repair_recovers_malformed_body():
    repaired = repair_tool_call_content(MALFORMED_CONTENT)
    assert repaired is not None, "repair should recover the stray-quote collision"
    # The repaired block's JSON interior must now parse.
    body = repaired.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    parsed = json.loads(body)
    assert parsed["name"] == "search"
    assert parsed["arguments"]["query_list"]  # non-empty query list survived


def test_repair_preserves_newline_before_close_tag():
    # sglang's Qwen detector matches `<tool_call>(.*?)\n</tool_call>`; the newline
    # before the closing tag must survive or the repaired block re-parses to [].
    repaired = repair_tool_call_content(MALFORMED_CONTENT)
    assert repaired is not None
    assert "\n</tool_call>" in repaired, "newline before </tool_call> was dropped"
    assert repaired.startswith("<tool_call>\n"), "newline after <tool_call> was dropped"


def test_valid_content_is_noop():
    # Already-valid content must return None (nothing to repair) and never mutate.
    assert repair_tool_call_content(VALID_CONTENT) is None


def test_valid_json_text_is_byte_identical():
    obj = repair_json_text(VALID_BODY)
    assert obj == json.loads(VALID_BODY)


def test_empty_input_is_noop():
    assert repair_tool_call_content("") is None
    assert repair_tool_call_content(None) is None
