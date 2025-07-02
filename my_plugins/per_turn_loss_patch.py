from contextlib import contextmanager
from datasets import Value, Sequence
from swift.llm.dataset.preprocessor.core import RowPreprocessor
from swift.llm.template.base import Template, ContextType
from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map, Messages
from typing import Dict, Any

# We need to patch RowPreprocessor so that it doesn't remove keys you want to keep
orig_check = RowPreprocessor._check_messages


def _check_messages_keep_loss_weight(row: Dict[str, Any]) -> None:
    if "messages" not in row:
        return
    messages = row["messages"]
    assert len(messages) > 0, f"messages: {messages}"
    # fix swift/SlimOrca
    for message in messages:
        to_keep = {"role", "content", "loss_weight"}
        keys = set(message.keys()) - to_keep
        for key in keys:
            message.pop(key)

    for message in messages:
        role, content = message["role"], message["content"]
        assert role in {
            "system",
            "user",
            "tool_call",
            "tool_response",
            "tool",
            "assistant",
        }, f"message: {message}"
        assert content is not None, f"message: {message}"


RowPreprocessor._check_messages = staticmethod(_check_messages_keep_loss_weight)


# Caching datset using ArrowWriter should preserve loss_weight
@contextmanager
def _patch_arrow_writer_with_loss_weight():
    from datasets.arrow_writer import ArrowWriter

    def _new_init(self, schema=None, features=None, *args, **kwargs):
        if features is not None:
            features["messages"] = [
                {
                    "role": Value(dtype="string"),
                    "content": Value(dtype="string"),
                    "loss_weight": Value(dtype="float32"),
                }
            ]
            features["images"] = [{"bytes": Value(dtype="binary"), "path": Value(dtype="string")}]
            features["objects"] = {
                "ref": Sequence(feature=Value(dtype="string"), length=-1),
                "bbox": Sequence(
                    feature=Sequence(feature=Value(dtype="float64"), length=-1),
                    length=-1,
                ),
                "bbox_type": Value(dtype="string"),
                "image_id": Sequence(feature=Value(dtype="int64"), length=-1),
            }
        ArrowWriter.__origin_init__(self, schema, features, *args, **kwargs)  # type:ignore

    ArrowWriter.__origin_init__ = ArrowWriter.__init__  # type:ignore
    ArrowWriter.__init__ = _new_init
    try:
        yield
    finally:
        ArrowWriter.__init__ = ArrowWriter.__origin_init__  # type:ignore
        del ArrowWriter.__origin_init__  # type:ignore


# replace the original patcher
RowPreprocessor._patch_arrow_writer = staticmethod(_patch_arrow_writer_with_loss_weight)


# Patch Template at the time of merging turns (take care of loss_weight merging logic)
def _swift_prepare_messages_loss_weight_patch(self, messages):
    if len(messages) < 2:
        return
    i = 1
    while i < len(messages):
        pre_message, message = messages[i - 1], messages[i]
        pre_role, pre_content = pre_message["role"], pre_message["content"]
        role, content = message["role"], message["content"]
        if pre_role == "assistant" and role == "tool":
            i_start = i
            while i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                i += 1
            pre_message["content"], tool_content = self.agent_template._format_tool_responses(
                pre_content, messages[i_start : i + 1]
            )
            messages[i_start : i + 1] = [{"role": "tool", "content": tool_content, "loss_weight": 0.0}]
            i = i_start + 1
        elif pre_role == "assistant" and role == "assistant" or pre_role == "user" and role == "user":
            # Consecutive messages from the assistant/user role need to be merged to prevent errors.
            pre_message["content"] = pre_content + content
            messages.pop(i)
        else:
            i += 1


def _preprocess_function_call_loss_weight_patch(self, inputs) -> None:
    agent_template = self.agent_template
    agent_template.template_meta = self.template_meta  # for hermes
    if inputs.tools:
        if isinstance(inputs.tools, str):
            inputs.tools = agent_template._parse_json(inputs.tools)
            if not isinstance(inputs.tools, (list, tuple)):
                inputs.tools = [inputs.tools]
        elif isinstance(inputs.tools, (list, tuple)):
            inputs.tools = [agent_template._parse_json(tool) for tool in inputs.tools]
        else:
            raise ValueError(f"inputs.tools: {inputs.tools}")
        for i, tool in enumerate(inputs.tools):
            inputs.tools[i] = agent_template.wrap_tool(tool)
    i = 0
    messages = inputs.messages
    while i < len(messages):
        if messages[i]["role"] == "tool_call":
            i_start = i
            try:
                loss_weight = messages[i]["loss_weight"]
            except Exception:
                print("'loss weight' key is missing in tool call, using 'loss_scale' = 0.0")
                loss_weight = 0.0
            while i + 1 < len(messages) and messages[i + 1]["role"] == "tool_call":
                i += 1
            tool_content = self.agent_template._format_tool_calls(messages[i_start : i + 1])
            messages[i_start : i + 1] = [
                {
                    "role": "assistant",
                    "content": tool_content,
                    "loss_weight": loss_weight,
                }
            ]
            i = i_start + 1
        else:
            i += 1


Template._swift_prepare_messages = _swift_prepare_messages_loss_weight_patch
Template._preprocess_function_call = _preprocess_function_call_loss_weight_patch


# finally define the loss scale function and register it:
class PerTurnLossScale(LossScale):
    """Weight comes from each assistant message's `my_weight` field."""

    def __call__(
        self,
        context_list: list[str],
        context_types: list[ContextType],
        messages: Messages,
        **kwargs,
    ) -> tuple[list[str], list[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        n_round = len(messages) // 2
        for context, context_type in zip(context_list, context_types):
            is_last_round = i + 1 == n_round
            loss_weight = 0
            if context_type == ContextType.SUFFIX:
                loss_scale = 1
            elif context_type == ContextType.RESPONSE:
                assert context == messages[2 * i + 1]["content"]
                try:
                    loss_weight = float(messages[2 * i + 1]["loss_weight"])
                except Exception as e:  # noqa
                    # raise ValueError(f"loss_weight is not there in message keys, \n {message[2*i + 1]=} \n {e=}")
                    loss_weight = 1
                i += 1
            new_context, loss_scale = self.get_loss_scale(
                context,
                context_type,
                is_last_round,
                weight=float(loss_weight),  # type:ignore
            )
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale

    def get_loss_scale(self, context, context_type, is_last_round, *, weight=1.0, **_):
        return [context], [float(weight)]


loss_scale_map["per_turn_weight"] = PerTurnLossScale
