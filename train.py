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


class PromptMonitor:
    """Background thread that monitors APO for new best prompts and saves them immediately."""
    
    def __init__(self, algo: agl.APO, config_path: str, config: dict, check_interval: int = 30):
        self.algo = algo
        self.config_path = config_path
        self.config = config
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.start_time = None
        self.last_saved_template = config.get("prompt_template", "")
        self.best_score = 0.0
        self.save_count = 0
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log(f"📡 Prompt monitor started (checking every {self.check_interval}s)")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        log("📡 Prompt monitor stopped")
    
    def _run(self):
        while self.running:
            time.sleep(self.check_interval)
            if not self.running:
                break
            
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            try:
                # Try to get the current best prompt from APO
                best_prompt = self.algo.get_best_prompt()
                if best_prompt and best_prompt.template:
                    current_template = best_prompt.template
                    
                    # Check if this is a new/different prompt
                    if current_template != self.last_saved_template:
                        self.save_count += 1
                        self.last_saved_template = current_template
                        
                        # Update config and save
                        self.config["prompt_template"] = current_template
                        save_config(self.config_path, self.config)
                        
                        # Also save a versioned backup
                        backup_path = f"{self.config_path.replace('.yaml', '')}_v{self.save_count}.yaml"
                        save_config(backup_path, self.config)
                        
                        log(f"🎉 NEW BEST PROMPT DETECTED! Saved as v{self.save_count}")
                        log(f"   📁 Main: {self.config_path}")
                        log(f"   📁 Backup: {backup_path}")
                        log(f"   ⏱️  Elapsed: {minutes}m {seconds}s")
                        
                        # Print first 200 chars of new prompt
                        preview = current_template[:200].replace('\n', ' ')
                        log(f"   📝 Preview: {preview}...")
                    else:
                        log(f"⏳ [{minutes}m {seconds}s] Monitoring... (no new prompt yet, saves: {self.save_count})")
                else:
                    log(f"⏳ [{minutes}m {seconds}s] Monitoring... (no best prompt available yet)")
                    
            except ValueError as e:
                # get_best_prompt() raises ValueError if no best prompt yet
                log(f"⏳ [{minutes}m {seconds}s] Monitoring... (waiting for first evaluation)")
            except Exception as e:
                log(f"⚠️ [{minutes}m {seconds}s] Monitor error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="entity_filter")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=1, 
                        help="Number of optimization rounds (beam_rounds)")
    parser.add_argument("--monitor-interval", type=int, default=30, 
                        help="Seconds between prompt checks")
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

    log(f"Starting training with {args.rounds} rounds...")
    log(f"Prompt will be auto-saved every time a better version is found!")
    log("=" * 60)
    
    # Start prompt monitor (checks for new best prompts and saves them)
    monitor = PromptMonitor(
        algo=algo,
        config_path=config_path,
        config=config,
        check_interval=args.monitor_interval
    )
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
        log(f"Total prompt versions saved: {monitor.save_count}")
        
        # Final save attempt
        try:
            best_prompt = algo.get_best_prompt()
            if best_prompt and best_prompt.template:
                if best_prompt.template != monitor.last_saved_template:
                    config["prompt_template"] = best_prompt.template
                    save_config(config_path, config)
                    log(f"💾 Final best prompt saved to: {config_path}")
                
                log("=" * 60)
                log("FINAL BEST PROMPT:")
                log("-" * 40)
                print(best_prompt.template)
                log("-" * 40)
        except Exception as e:
            log(f"⚠️ Could not retrieve final best prompt: {e}")


if __name__ == "__main__":
    main()
