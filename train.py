import argparse
import json

from openai import AsyncOpenAI
import yaml

import agentlightning as agl

from agents.entity_filter import entity_filter_agent
from config import get_openai_config


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataset(dataset, goal, eval_mode, model):
    for item in dataset:
        item["goal"] = goal
        item["eval_mode"] = eval_mode
        item["model"] = model
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="entity_filter")
    parser.add_argument("--model")
    args = parser.parse_args()

    config_path = f"configs/nodes/{args.node}.yaml"
    train_path = f"datasets/{args.node}/train.jsonl"
    val_path = f"datasets/{args.node}/val.jsonl"

    config = load_config(config_path)
    openai_config = get_openai_config()
    model_name = args.model or openai_config["model_name"]

    train_ds = build_dataset(
        load_jsonl(train_path),
        config["goal"],
        config.get("eval_mode", "llm"),
        model_name,
    )
    val_ds = build_dataset(
        load_jsonl(val_path),
        config["goal"],
        config.get("eval_mode", "llm"),
        model_name,
    )

    algo = agl.APO(
        AsyncOpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
        ),
        gradient_model=model_name,
        apply_edit_model=model_name,
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
    print(best_prompt.template)


if __name__ == "__main__":
    main()
