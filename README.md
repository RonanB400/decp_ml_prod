
📁 decp_prod/
├── 📄 .gitignore
├── 📄 Dockerfile
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📄 tests
├── 📄 api
    └── 📄 __init__.py
    ├── 📄 fastapi.py
└── 📁 decp
    ├── 📁 interface/
    │   └── 📄 main.py
    │   └── 📄 main_local.py
    ├── 📁 models/
    │   ├── 📄 full_pipeline.pkl
    │   ├── 📄 hdbscan_model.pkl
    │   ├── 📄 ...
    │   └── 📄 cluster_profiles.csv
    └── 📁 ml_logic
        ├── 📄 __init__.py
        ├── 📄 config.py
        ├── 📄 predict.py
        └── 📄 preprocess.py
