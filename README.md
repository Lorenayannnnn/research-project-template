# research-project-template
Template for research project

## Structure
```
└── configs
    ├── base_configs.yaml: contains the base configurations for the project
    ├── overwrite_configs.yaml: include variations that you want to overwrite to the base configurations
    └── ...
└── src
    ├── analysis_module: scripts for analysis
    ├── data_module: data loading and processing (tokenization...)
    ├── model_module: model definition and loading
    ├── train_module: code for training
    ├── (Optional) eval_module: code for evaluation
    └── ...
└── .gitignore
└── main.sh: main entry point for the project
└── requirements.txt: list of dependencies
└── ...
```

## Usage
### Setup
```
conda create -n YOUR_PROJECT_NAME python=3.8
conda activate YOUR_PROJECT_NAME
pip install -r requirements.txt
```
### Code Implementation for Each Module
### Run
- Go to [main.sh](main.sh) and modify the configurations in the `configs` folder
- Run:
```
bash main.sh
```
