import yaml
from pathlib import Path

# ----- Dynamically get the project root (parent of config.py) -----
BASE_DIR = Path(__file__).resolve().parent

# Load YAML configuration
with open(BASE_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# ----- GENERATION -----
TEMPERATURE = cfg["generation"]["temperature"]
MAX_TOKENS = cfg["generation"]["max_tokens"]
LLM_MODEL_NAME = cfg["generation"]["llm_model_name"]