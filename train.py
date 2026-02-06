from src.utils.windows_patch import apply_patches
apply_patches()

import argparse
import json
import os

from openai import AsyncOpenAI
import yaml

import agentlightning as agl

from src.agents.entity_filter import entity_filter_agent
from src.client.openai_httpx import build_async_httpx_client
from settings import OPTIMIZER_CONFIG, ROLLOUT_CONFIG

OPTIMIZER_BASE_URL = OPTIMIZER_CONFIG.base_url
OPTIMIZER_API_KEY = OPTIMIZER_CONFIG.api_key
OPTIMIZER_MODEL = OPTIMIZER_CONFIG.model_name

ROLLOUT_BASE_URL = ROLLOUT_CONFIG.base_url
ROLLOUT_API_KEY = ROLLOUT_CONFIG.api_key
ROLLOUT_MODEL = ROLLOUT_CONFIG.model_name

def _normalize_sample(item):
    if "input" in item and isinstance(item["input"], dict):
        inp = item["input"]
        question = inp.get("question") or item.get("question")
        entities = inp.get("entities") or item.get("entities")
        normalized = {}
        if question is not None:
            normalized["question"] = question
        if entities is not None:
            normalized["entities"] = entities
        if "human_score" in item:
            normalized["human_score"] = item["human_score"]
        if "gold" in item and item["gold"] is not None:
            normalized["gold"] = item["gold"]
        elif "output" in item and item["output"] is not None:
            normalized["gold"] = item["output"]
        if "gold_struct" in item and item["gold_struct"] is not None:
            normalized["gold_struct"] = item["gold_struct"]
        elif "format_output" in item and item["format_output"] is not None:
            normalized["gold_struct"] = item["format_output"]
        return normalized

    if "gold" not in item and "output" in item:
        item["gold"] = item["output"]
    if "gold_struct" not in item and "format_output" in item:
        item["gold_struct"] = item["format_output"]
    return item


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [_normalize_sample(json.loads(line)) for line in file if line.strip()]


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataset(dataset, goal, eval_mode, model, base_url, api_key):
    for item in dataset:
        item["goal"] = goal
        item["eval_mode"] = eval_mode
        item["model"] = model
        item["model_base_url"] = base_url
        item["model_api_key"] = api_key
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="entity_filter")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config_path = f"src/configs/nodes/{args.node}.yaml"
    train_path = f"src/datasets/{args.node}/train.jsonl"
    val_path = f"src/datasets/{args.node}/val.jsonl"

    config = load_config(config_path)
   
    rollout_model_name = args.model or ROLLOUT_MODEL
    
    train_ds = build_dataset(
        load_jsonl(train_path),
        config["goal"],
        config.get("eval_mode", "llm"),
        rollout_model_name,
        ROLLOUT_BASE_URL,
        ROLLOUT_API_KEY,
    )
    val_ds = build_dataset(
        load_jsonl(val_path),
        config["goal"],
        config.get("eval_mode", "llm"),
        rollout_model_name,
        ROLLOUT_BASE_URL,
        ROLLOUT_API_KEY,
    )

    algo = agl.APO(
        AsyncOpenAI(
            api_key=OPTIMIZER_API_KEY,
            base_url=OPTIMIZER_BASE_URL,
            http_client=build_async_httpx_client(),
        ),
        gradient_model=OPTIMIZER_MODEL,
        apply_edit_model=OPTIMIZER_MODEL,
    )
    trainer = agl.Trainer(
        algorithm=algo,
        strategy="shm",
        initial_resources={
            "prompt_template": agl.PromptTemplate(
                template=config["prompt_template"],
                engine="f-string",
            )
        },
        adapter=agl.TraceToMessages(),
    )

    trainer.fit(
        agent=entity_filter_agent,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    best_prompt = algo.get_best_prompt()
    print("Best Prompt Template:")
    print(best_prompt.template)


if __name__ == "__main__":
    main()
