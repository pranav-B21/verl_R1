from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionToolCall
from verl.workers.rollout.schemas import AsyncRolloutRequest, Message


class DummyProcessor:
    def __init__(self, expect_tool_calls: bool):
        self.expect_tool_calls = expect_tool_calls
        self.captured_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.captured_messages = messages
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            has_tool_calls = "tool_calls" in msg
            if self.expect_tool_calls:
                assert has_tool_calls, "Expected tool_calls to be preserved when present"
            else:
                assert not has_tool_calls, "Empty tool_calls should be pruned before reaching the template"
        return "dummy_prompt"


def _call_handle_apply_chat_template(processor, messages):
    return AsyncRolloutRequest._handle_apply_chat_template(
        processor,
        messages,
        multi_modal_data={"image": [], "video": []},
        tools=None,
        add_generation_prompt=False,
        tokenize=False,
    )


def test_handle_apply_chat_template_prunes_empty_tool_calls():
    processor = DummyProcessor(expect_tool_calls=False)
    messages = [Message(role="assistant", content="Hello", tool_calls=[])]

    prompt = _call_handle_apply_chat_template(processor, messages)

    assert prompt == "dummy_prompt"
    assert processor.captured_messages[0].get("tool_calls") is None


def test_handle_apply_chat_template_preserves_real_tool_calls():
    processor = DummyProcessor(expect_tool_calls=True)
    tool_call = OpenAIFunctionToolCall(
        id="tool_1",
        function=OpenAIFunctionCallSchema(name="foo", arguments={"bar": "baz"}),
    )
    messages = [Message(role="assistant", content="Calling tool", tool_calls=[tool_call])]

    prompt = _call_handle_apply_chat_template(processor, messages)

    assert prompt == "dummy_prompt"
    tool_calls = processor.captured_messages[0]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "foo"
