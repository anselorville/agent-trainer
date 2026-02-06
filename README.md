# Agent Trainer (Agent Lightning APO)

An agent optimizer scaffold based on **Agent Lightning**, specifically designed for optimizing a prompt-based entity role classifier in finance retrieval workflows using the Automatic Prompt Optimization (APO) algorithm.

## Features

- **Prompt Optimization**: Automatically refines prompt templates based on training data.
- **Finance Domain Focused**: Pre-configured for entity role classification (subject, publisher, author, etc.).
- **Windows Compatible**: Includes monkeypatches for `agentlightning`'s Unix-specific dependencies (`fcntl`, `pwd`, `AF_UNIX`).

## Project Structure

```text
.
├── src/
│   ├── agents/          # Agent implementations (e.g., entity_filter_agent)
│   ├── client/          # HTTPX and LLM client configurations
│   ├── configs/         # YAML nodes and arbiter settings
│   ├── datasets/        # Training and validation JSONL files
│   └── evaluators/      # LLM Judge and Human Feedback logic
├── train.py             # Main entry point for training
├── settings.py          # Global configuration management
├── .env                 # Environment variables (API keys, Base URLs)
└── requirements.txt     # Project dependencies
```

## Setup

1. **Environment**: Ensure you are using the local virtual environment.
   ```powershell
   # Activate on Windows
   .\.venv\Scripts\Activate.ps1
   ```

2. **Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configuration**: Create a `.env` file (copy from `.env.example`).
   ```env
   LLM_API_KEY=your_api_key
   LLM_BASE_URL=https://api.openai.com/v1
   LLM_MODEL_NAME=glm-4.7
   ```

## Usage

Run training for a specific node (e.g., `entity_filter`):

```powershell
.\.venv\Scripts\python.exe train.py --node entity_filter
```

## Data Format

- **Location**: `src/datasets/[node_name]/train.jsonl`
- **Format**: Each line is a JSON object.
  ```json
  {
    "input": {
      "question": "立讯精密最近公告...",
      "entities": { ... }
    },
    "gold": "Target Output Role Mapping"
  }
  ```

---
> [!NOTE]
> This project contains Windows-specific patches in `train.py` to handle `agentlightning` dependencies on Unix modules like `fcntl` and `pwd`.
