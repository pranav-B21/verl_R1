# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# Logging configuration matches other preprocess utilities for consistent output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless recommendation assistant."

DEFAULT_REASONING_INSTRUCTION = """Begin by briefly analyzing the current user's viewing history to infer \
his preference. If necessary, you can then generate a query and call an existing search engine \
by specifying "<tool_call> query </tool_call>". The query can search: (1) other users' interaction histories and (2) \
music metadata (Price, SalesRank, Brand, Categories). \
For example, you may query to identify users who engaged with musics similar to those viewed by the current user, \
retrieve their interaction patterns, and use these insights to predict additional musics the current user may appreciate.
Also, you may query metadata to enrich context for specific musics. \
Only query the search engine when necessary, and keep all reasoning concise.
Remember to provide the name of the user's most preferred music (a single item) enclosed within <answer> and </answer> at last, using two double quotes. For example: <answer> \"Revenge\" </answer>."""

DEFAULT_PROMPT_PREFIX = """Resolve the given task. You must conduct reasoning inside <think> and </think> first every \
time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine \
by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. \
You can contiue this reasoning-search process. If you find no further external knowledge needed, you must directly provide \
the name of the user's most preferred music inside <answer> and </answer> and use two double quotes to enclose the name of the music, without any other illustrations. Please provide only a single music that best matches the user's preferences. For example, <answer> \"Revenge\" </answer>. Task: {question}
"""


def _load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}, but found {type(data)}")
    return data


def _maybe_sample_records(
    records: Sequence[Dict[str, Any]], sample_size: Optional[int], seed: Optional[int]
) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), sample_size)
    return [records[i] for i in sorted(indices)]


def _build_question(entry: Dict[str, Any], reasoning_instruction: str, prompt_prefix: str) -> str:
    instruction = entry.get("instruction", "").strip()
    user_input = entry.get("input", "").strip()
    question_body = "\n".join(filter(None, [instruction, user_input, reasoning_instruction.strip()]))
    return prompt_prefix.format(question=question_body)


def _build_record(
    entry: Dict[str, Any],
    idx: int,
    split: str,
    data_source: str,
    ability: str,
    reasoning_instruction: str,
    prompt_prefix: str,
    system_prompt: str,
) -> Dict[str, Any]:
    question = _build_question(entry, reasoning_instruction, prompt_prefix)
    golden_answer = entry.get("output", "")
    if isinstance(golden_answer, str):
        golden_answer = golden_answer.strip()

    reward_model_data = {
        "style": "rule",
        "ground_truth": {
            "target": golden_answer,
        },
    }
    data_source_tagged = f"{data_source}_{split}"
    prompt = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": reward_model_data["ground_truth"],
                "question": question,
                "data_source": data_source_tagged,
            }
        }
    }
    extra_info = {
        "index": idx,
        "need_tools_kwargs": True,
        "question": question,
        "split": split,
        "tools_kwargs": tools_kwargs,
    }

    return {
        "data_source": data_source_tagged,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": entry.get("metadata"),
    }


def _convert_split(
    records: Sequence[Dict[str, Any]],
    split: str,
    data_source: str,
    ability: str,
    reasoning_instruction: str,
    prompt_prefix: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    processed = [
        _build_record(
            entry,
            idx,
            split,
            data_source=data_source,
            ability=ability,
            reasoning_instruction=reasoning_instruction,
            prompt_prefix=prompt_prefix,
            system_prompt=system_prompt,
        )
        for idx, entry in enumerate(records)
    ]
    if not processed:
        raise ValueError(f"No records were built for split '{split}'.")
    return processed


def _write_parquet(records: List[Dict[str, Any]], path: str) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading training records from %s", args.train_json)
    train_records = _load_json_records(args.train_json)
    logger.info("Loaded %d training records", len(train_records))

    logger.info("Loading test records from %s", args.test_json)
    test_records = _load_json_records(args.test_json)
    logger.info("Loaded %d test records before sampling", len(test_records))
    test_records = _maybe_sample_records(test_records, args.test_sample_size, args.test_sample_seed)
    logger.info("Retained %d test records after sampling", len(test_records))

    logger.info("Processing training split...")
    train_processed = _convert_split(
        train_records,
        split="train",
        data_source=args.data_source,
        ability=args.ability,
        reasoning_instruction=args.reasoning_instruction,
        prompt_prefix=args.prompt_prefix,
        system_prompt=args.system_prompt,
    )
    train_output_path = os.path.join(args.output_dir, "train.parquet")
    _write_parquet(train_processed, train_output_path)
    logger.info("Saved training split to %s", train_output_path)

    logger.info("Processing test split...")
    test_processed = _convert_split(
        test_records,
        split="test",
        data_source=args.data_source,
        ability=args.ability,
        reasoning_instruction=args.reasoning_instruction,
        prompt_prefix=args.prompt_prefix,
        system_prompt=args.system_prompt,
    )
    test_output_path = os.path.join(args.output_dir, "test.parquet")
    _write_parquet(test_processed, test_output_path)
    logger.info("Saved test split to %s", test_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Amazon recommendation data into Search-R1 style Parquet files."
    )
    parser.add_argument(
        "--train_json",
        default="./amazon_data/CDs_and_Vinyl_train_sampled_orig.json",
        help="Path to the JSON file that contains the training split.",
    )
    parser.add_argument(
        "--test_json",
        default="./amazon_data/CDs_and_Vinyl_test.json",
        help="Path to the JSON file that contains the test split.",
    )
    parser.add_argument(
        "--output_dir",
        default="./amazon_data",
        help="Directory where the processed Parquet files will be written.",
    )
    parser.add_argument(
        "--test_sample_size",
        type=int,
        default=1000,
        help="Optional number of examples to keep in the test split. "
        "If the value is non-positive or larger than the available examples, no sampling occurs.",
    )
    parser.add_argument(
        "--test_sample_seed",
        type=int,
        default=43,
        help="Random seed for reproducible test sampling.",
    )
    parser.add_argument("--data_source", default="amazon", help="Value stored in the data_source column.")
    parser.add_argument("--ability", default="fact-reasoning", help="Ability tag for the processed rows.")
    parser.add_argument(
        "--reasoning_instruction",
        default=DEFAULT_REASONING_INSTRUCTION,
        help="Instruction appended to each user turn to steer tool usage.",
    )
    parser.add_argument(
        "--prompt_prefix",
        default=DEFAULT_PROMPT_PREFIX,
        help="Template used to wrap the final user prompt. Must include '{question}'.",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System message prepended to each prompt.",
    )

    main(parser.parse_args())
