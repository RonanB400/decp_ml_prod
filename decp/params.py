import os

MODELS_DIR = os.environ.get('MODELS_DIR')
PIPELINE_PATH = os.environ.get('PIPELINE_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')
PROFILES_PATH = os.environ.get('PROFILES_PATH')

API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', 8000))
API_WORKERS = int(os.environ.get('API_WORKERS', 1))
