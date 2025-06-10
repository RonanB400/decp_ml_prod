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
│   └── 📄 fastapi.py
│
├── 📁 decp/
│   ├── 📄 __init__.py
│   ├── 📁 interface/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py
│   │   └── 📄 main_local.py
│   └── 📁 ml_logic/
│       ├── 📄 __init__.py
│       ├── 📄 config.py
│       ├── 📄 predict.py
│       └── 📄 preprocess.py
│
├── 📁 models/
│   ├── 📄 cluster_profiles.csv
│   ├── 📄 full_pipeline.pkl
│   ├── 📄 hdbscan_model.pkl
│   └── 📄 ...
│
└── 📁 tests/
    └── 📄 ...
