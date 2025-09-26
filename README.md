# Leveraging Self-Attention Mechanisms in Small Language Models for In-File Vulnerability Localisation in C/C++ Code
This repository contains the code developed in partial fulfilment of the dissertation project titled "Leveraging Self-Attention Mechanisms in Small Language Models for In-File Vulnerability Localisation in C/C++ Code"


## Requirements
- **Python**: v3.12.3+.
- **Operating System**: Linux/Unix-based. All following instructions are intended for Linux/Unix-based systems.
- **Hardware**: If running experiments locally, a GPU is strongly advised.

Dependencies are listed in `requirements.txt'

## Set-up Instructions

To clone this repository and install dependencies:
```bash
git clone https://github.com/zorbzers/com6013-slm-vuln-localisation.git
cd slm-vuln-localisation
mkdir cache
mkdir logs
pip install -r requirements.txt
```

It is reccomended to set up and work inside a Python virtual environment. To set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

To install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To download and extract the raw dataset:
```bash
pip install gdown
sudo apt install unzip
cd data
gdown --id 1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X
unzip MSR_data_cleaned.zip
cd ../
```
The first time a script that requires the dataset is ran, preprocessing and caching of the dataset will take place. Beyond this, the cached version of the dataset will always be used unless corrupt or deleted.


## Usage
The instructions to obtain all the necessary results is detailed in this section.

#### A Note on CodeLlama Access
Running experiments with CodeLlama models requires a valid [Hugging Face](https://huggingface.co/) account and access token, as these models are gated by Meta.
1. First, access must be requested to the CodeLlama 7B model on Hugging Face at https://huggingface.co/codellama/CodeLlama-7b-hf.
2. Once access is granted, a personal access token can be generated from your Hugging Face account settings.
3. Log in locally using the Hugging Face CLI:
    ```bash
    huggingface-cli login
    ```
    and pased the access token when prompted.

Without this step, any attempts to run the CodeLlama 7B model will fail with an authentication error.

### Obtaining Baseline Results
The baseline results can be obtained either locally, or via a HPC cluster. The HPC cluster used for this project was [The University of Sheffield's Stanage HPC cluster](https://docs.hpc.shef.ac.uk/en/latest/stanage/index.html#gsc.tab=0). All the `sbatch` commands detailed will be specific to Stanage, and may not behave as intended on different systems.

To obtain a full set (8383) of baseline results for a specific model and shot combination locally:
```bash
for i in $(seq 0 8384); do
    python -m scripts.obtain_baseline \
        --model starcoder2-3b \
        --shots 0 \
        --index $i
done
```
The possible models are:
- `starcoder2-3b`
- `starcoder2-7b`
- `tinyllama`
- `codellama-7b`
- `phi-1.5`
- `deepseek-1.3b`
- `deepseek-6.7b`

The possible shots are 0, 1 or 3. 

To obtain results using the Stanage HPC cluster:
```bash
for offset in 0 1000 2000 3000 4000 5000 6000 7000; do
    sbatch --wait --array=0-999 \
           --export=MODEL=starcoder2-3b,SHOTS=0,OFFSET=$offset \
           scripts/obtain_baseline_script.sh
done
sbatch --array=0-383 \
       --export=MODEL=starcoder2-3b,SHOTS=0,OFFSET=8000 \
       scripts/obtain_baseline_script.sh
```

### Extracting and Caching Attention Matrices
To obtain a full set of cached attention matrices for a given model and shot combination locally, the following command can be ran:

```bash
for i in $(seq 0 8384); do
    python -m scripts.compute_attention \
        --model bigcode/starcoder2-3b \
        --shots 0 \
        --index $i
done
```
The possible models are:
- `bigcode/starcoder2-3b`
- `bigcode/starcoder2-7b`
- `TinyLlama/TinyLlama_v1.1`
- `meta-llama/CodeLlama-7b-hf`
- `microsoft/phi-1_5`
- `deepseek-ai/deepseek-coder-1.3b-base`
- `deepseek-ai/deepseek-coder-6.7b-base`

The possible shots are 0, 1 or 3. 

To obtain matrices using the Stanage HPC cluster:
```bash
for offset in 0 1000 2000 3000 4000 5000 6000 7000; do
    sbatch --wait --array=0-999 \
           --export=MODEL=bigcode/starcoder2-3b,SHOTS=0,OFFSET=$offset \
           scripts/compute_attention_script.sh
done
sbatch --array=0-383 \
       --export=MODEL=bigcode/starcoder2-3b,SHOTS=0,OFFSET=8000 \
       scripts/compute_attention_script.sh
```

### Running 5-Fold Cross Validation
To run 5-fold cross validation for a given model, shots and classifier combination locally:

```bash
python -m scripts.train_attn_cv.py --model_name bigcode/starcoder2-3b --classifier LSTM_small --shots 0
```

The possible models are:
- `bigcode/starcoder2-3b`
- `bigcode/starcoder2-7b`
- `TinyLlama/TinyLlama_v1.1`
- `meta-llama/CodeLlama-7b-hf`
- `microsoft/phi-1_5`
- `deepseek-ai/deepseek-coder-1.3b-base`
- `deepseek-ai/deepseek-coder-6.7b-base`

The possible shots are 0, 1 or 3. 

The possible classifiers are:
- `LSTM_small`
- `LSTM_moderate`
- `LSTM_large`
- `BiLSTM_small`
- `BiLSTM_moderate`
- `BiLSTM_large`
- `LogReg_default`
- `LogReg_lowC`
- `LogReg_l1`
- `XGB_small`
- `XGB_moderate`
- `XGB_large`

To run 5-fold cross validation using the Stanage HPC cluster:

```bash
sbatch --export=MODEL_NAME=bigcode/starcoder2-3b,CLASSIFIER=BiLSTM_moderate,SHOTS=0,BATCH_SIZE=16 \
       scripts/train_attn_cv.sh
```

### Evaluating Results
To evaluate the results obtained, the Jupyter Notebook `evaluate_results.ipynb` is used.
