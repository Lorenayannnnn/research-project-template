# research-project-template
Template for research project

## Structure
```
└── datasets
└── outputs
└── scripts
    └── main.sh: bash script to run the main.py. Override config options here.
└── src
    ├── analysis_module: scripts for analysis, plotting, etc.
    └── configs
        ├── base_configs.yaml: contains the base configurations for the main.py script
        └── ...
    ├── data_module: data loading and processing (tokenization...)
    ├── eval_module: code for evaluation
    ├── model_module: model definition and initialization
    ├── train_module: code for training
    ├── common_utils.py: common utility functions
    └── main.py: main entry point for the project. config is specified via the hydra decorator, which can be overridden from bash
└── .gitignore
└── requirements.txt: list of dependencies
└── ...
```

## Usage
### Setup
```
conda create -n YOUR_PROJECT_NAME python=YOUR_PYTHON_VERSION
conda activate YOUR_PROJECT_NAME
pip install -r requirements.txt
```
### Run
- Go to [main.sh](scripts/main.sh) and modify the configurations as needed
- Run:
```
bash main.sh
```
