from src.utils.windows_patch import apply_patches
apply_patches()

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

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


def log(message: str):
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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


def save_config(path, config):
    """Save configuration back to YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True, default_flow_style=False, sort_keys=False)


def build_dataset(dataset, goal, eval_mode, model, base_url, api_key):
    for item in dataset:
        item["goal"] = goal
        item["eval_mode"] = eval_mode
        item["model"] = model
        item["model_base_url"] = base_url
        item["model_api_key"] = api_key
    return dataset


class ProgressMonitor:
    """Background thread to periodically print alive status."""
    
    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.start_time = None
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run(self):
        while self.running:
            time.sleep(self.interval)
            if self.running:
                elapsed = time.time() - self.start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                log(f"⏳ Training in progress... (elapsed: {minutes}m {seconds}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="entity_filter")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=3, 
                        help="Number of optimization rounds (beam_rounds)")
    parser.add_argument("--monitor-interval", type=int, default=60, 
                        help="Seconds between progress heartbeat messages")
    args = parser.parse_args()

    config_path = f"src/configs/nodes/{args.node}.yaml"
    train_path = f"src/datasets/{args.node}/train.jsonl"
    val_path = f"src/datasets/{args.node}/val.jsonl"

    log("=" * 60)
    log("Agent Trainer - Starting")
    log("=" * 60)
    
    log(f"Loading config from: {config_path}")
    config = load_config(config_path)
    original_prompt = config.get("prompt_template", "")
   
    rollout_model_name = args.model or ROLLOUT_MODEL
    log(f"Rollout model: {rollout_model_name}")
    log(f"Optimizer model: {OPTIMIZER_MODEL}")
    
    log(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    log(f"Loaded {len(train_data)} training samples")
    
    log(f"Loading validation data from: {val_path}")
    val_data = load_jsonl(val_path)
    log(f"Loaded {len(val_data)} validation samples")
    
    train_ds = build_dataset(
        train_data,
        config["goal"],
        config.get("eval_mode", "llm"),
        rollout_model_name,
        ROLLOUT_BASE_URL,
        ROLLOUT_API_KEY,
    )
    val_ds = build_dataset(
        val_data,
        config["goal"],
        config.get("eval_mode", "llm"),
        rollout_model_name,
        ROLLOUT_BASE_URL,
        ROLLOUT_API_KEY,
    )

    log(f"Initializing APO algorithm with {args.rounds} rounds...")
    algo = agl.APO(
        AsyncOpenAI(
            api_key=OPTIMIZER_API_KEY,
            base_url=OPTIMIZER_BASE_URL,
            http_client=build_async_httpx_client(),
        ),
        gradient_model=OPTIMIZER_MODEL,
        apply_edit_model=OPTIMIZER_MODEL,
        beam_rounds=args.rounds,
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

    log("Starting training...")
    log("=" * 60)
    
    # Start progress monitor
    monitor = ProgressMonitor(interval_seconds=args.monitor_interval)
    monitor.start()
    
    try:
        trainer.fit(
            agent=entity_filter_agent,
            train_dataset=train_ds,
            val_dataset=val_ds,
        )
    except KeyboardInterrupt:
        log("⚠️ Training interrupted by user (Ctrl+C)")
    except Exception as e:
        log(f"❌ Training error: {e}")
        raise
    finally:
        monitor.stop()
        log("=" * 60)
        log("Training session ended.")
        
        # Try to get and save the best prompt
        try:
            best_prompt = algo.get_best_prompt()
            if best_prompt and best_prompt.template:
                log("✅ Best prompt retrieved successfully!")
                log("-" * 40)
                print(best_prompt.template)
                log("-" * 40)
                
                # Save to config file
                config["prompt_template"] = best_prompt.template
                save_config(config_path, config)
                log(f"💾 Best prompt saved to: {config_path}")
                
                # Also save a backup with timestamp
                backup_path = f"src/configs/nodes/{args.node}_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                save_config(backup_path, config)
                log(f"💾 Backup saved to: {backup_path}")
            else:
                log("⚠️ No best prompt available.")
        except Exception as e:
            log(f"⚠️ Could not retrieve best prompt: {e}")
            log("The original prompt in the config file remains unchanged.")


if __name__ == "__main__":
    main()
