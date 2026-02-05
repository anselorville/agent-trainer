# Prompt Optimizer (Agent Lightning)

Minimal training scaffold for optimizing a prompt-based entity role classifier
in a finance retrieval workflow using Agent Lightning APO.

## Setup

1. Create a Python virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Configure OpenAI client in a `.env` file:

```
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_NAME=gpt-5-mini
```

## Run

```
python train.py --node entity_filter
```

## Data

- Add samples to `datasets/entity_filter/train.jsonl` and
  `datasets/entity_filter/val.jsonl`.
- Each line is a JSON object with `question` and `disambiguated_entities`.
- Optional: add `human_score` for human-evaluated samples.

## Notes

- APO currently optimizes a single prompt template per run.
- For multiple nodes, run training once per node config.
