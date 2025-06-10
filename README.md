# DECP Production Environment
📁 decp_prod/
├── 📄 .gitignore
├── 📄 Dockerfile
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
│
├── 📁 api/
│   ├── 📄 __init__.py
│   ├── 📄 fast.py            # FastAPI application
│   ├── 📄 preprocessing.py   # Preprocessing utilities
│   └── 📄 prediction.py      # Prediction utilities
│
├── 📁 decp/
│   ├── 📄 __init__.py
│   └── 📄 params.py         # Configuration parameters
│
├── 📁 models/
│   ├── 📄 .gitkeep          # To keep the directory in git
│   ├── 📄 README.md         # Instructions for model files
│
├── 📁 tests/
│   ├── 📄 __init__.py
│   └── 📄 test_api.py       # Tests for the API
