# Deciphering Transcription Factors Action Mechanism in Reprogramming Using Machine Learning
*BINP39 Research Project (30 cr) for the Master's Programme in Bioinformatics, Lund University*

## Overview
This GitHub repository (https://github.com/kaburi0214/BINP39_research_project.git) contains all required analysis pipelines and code for the BINP39 research project. This project is based on scRNA-seq analysis with machine learning, which aims to:
- Establish quantitative relationships between transcription factor (TF) combinations and cell reprogramming outcomes using discriminative models
- Analyze synergistic mechanisms of key TFs determining dendritic cell reprogramming fate  
- Predict reprogramming effects of hold-out TF combinations using generative models

For detailed usage instructions, see the sections below on 1) working directory setup, 2) data acquisition, 3) environment configuration, and 4) script running. 
All key output figures from notebooks are recorded in `results/figures/` in this repository.

## 1) Working directory setup
We start by creating a working directory for this project to store all datasets, scripts, and results. The recommended working directory structure is:
```
working_folder/
├── raw_data/          # Raw datasets
├── results/           # Analysis output files, figures, trained models
│  ├── anndata
│  ├── csv
│  ├── figures
│  ├── model
├── kt_tuning/         # model tuning details 
├── python scripts     # Python modules defining model architectures and the estimator framework imported by notebooks (models_v1.py, estimator_v1.py)
├── jupyter notebooks  # Jupyter notebooks for data processing and analysis (processing_v2.ipynb, predict_00.ipynb, predict_01.ipynb, gen_v3.ipynb)
├── environment files  # Docker and conda environment files (Dockerfile_base, Dockerfile_cellflow, Dockerfile_cr, env_cellflow.yml, env_cr.yml)
└── README.md          # Project documentation 
```
**Note:** Only the `raw_data/` folder and its contents, along with all python scripts, jupyter notebooks and environment files need to be prepared initially. Other directories (`results/`, `kt_tuning/`) can be automatically generated during analysis execution.

## 2) Data acquisition
All required datasets are available through this [Google Drive](https://drive.google.com/drive/folders/1ofBo4uFd4TTm_Nqm2lzn2D142y4k3VCI?usp=sharing)

The h5ad file: *sc_divergentDC.48TF_Arrayed.h5ad* contains the endogenous gene expression matrix while the csv file: *brd_divergentDC.48TF_Arrayed.csv* contains the 48 exogenous TF expression matrix.
Download both files and place them in `raw_data/` directory of your working folder.

## 3) Environment configuration
We provide 2 ways to help to set up the appropriate environment for this project:
1. **Conda environment** 
2. **Docker containers**

The key dependencies include: Python 3.9+, numpy, pandas, scipy, scanpy, anndata, scikit-learn, tensorflow, matplotlib, seaborn, celltypist, cellflow.
See `env_cr.yml` and `env_cellflow.yml` for complete dependency lists.

### Conda environment
```bash
# For notebooks apart from gen_v3.ipynb
conda env create -f env_cr.yml
conda activate cellrp
```
```bash
# For gen_v3.ipynb
conda env create -f env_cellflow.yml
conda activate cellflow
```
### Docker containers
```bash
# build docker images with Dockerfiles
docker build -f ./Dockerfile_base -t cellrp_base .
docker build -f ./Dockerfile_cr -t cellrp_cr .
docker build -f ./Dockerfile_cellflow -t cellrp_cellflow .
```
You can also directly pull docker images from DockerHub instead of building them with Dockerfiles
```bash
docker pull tongrui214/cellrp_base:latest
docker pull tongrui214/cellrp_cr:latest
docker pull tongrui214/cellrp_cellflow:latest
```

```bash
# set up environments to run notebooks apart from gen_v3.ipynb
docker run -it -p 8888:8888 -v /path/to/your/working_folder:/usr/local/cellrp cellrp_cr
conda activate cellrp
python -m ipykernel install --user --name cellrp --display-name "Python (cellrp)"
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
```bash
# set up environments to run gen_v3.ipynb
docker run -it -p 8888:8888 -v /path/to/your/working_folder:/usr/local/cellrp cellrp_cellflow
conda activate cellflow
python -m ipykernel install --user --name cellflow --display-name "Python (cellflow)"
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 4) Script running
After activating the specific conda environment for certain notebook(s), run:
```bash
jupyter notebook
```
.You'll find all required data and notebooks in the Jupyter interface (skip this step if using Docker containers as Jupyter is already launched).
Afterwards, just run each notebook:
 - **`processing_v2.ipynb`** should be the first one to execute, which preprocesses the raw datasets for later analyses.
 - **`predict_00.ipynb`** is for reprogramming cell type prediction using the decision tree model, including downstream validation and interpretation.
 - **`predict_01.ipynb`** is for reprogramming transcriptomic expression prediction using linear and non-linear neural network models.
 - **`gen_v3.ipynb`** predicts reprogramming effects for hold-out TF combinations using the CellFlow generative model.

 **Notes:** While running all cells sequentially is recommended for complete reproduction, you can also pre-download all processed datasets and pre-trained models from `results/` folder from the server (inf-48-2024@130.235.8.214:/home/inf-48-2024/binp39/cell_reprogram/cellrp)
to skip computationally intensive steps. Some smaller model files are also available in this repository's `results/` folder.
