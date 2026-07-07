# Dicentric Chromosomes Detection
Little framework to perform image preprocessing and dicentric chromosomes detection from metaphase-cell images.

# Reproducing the Experiments

This repository contains the code used in the experiments presented in the paper.

## 1. Install the dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

> **Note**
>
> The provided `requirements.txt` corresponds **exactly** to the Python environment used to run the experiments for the paper. Depending on your operating system, Python version, CUDA version, or installed libraries, you may need to remove or modify some dependencies before installation.

---

## 2. Prepare the dataset

Place the dataset inside the `dataset` directory following the structure below:

```text
dataset/
└── train
    ├── data.yaml
    ├── images
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── labels
        ├── image_001.txt
        ├── image_002.txt
        └── ...

```

---

## 3. Initialize the experiment

Run the setup script:

```bash
bash setup.sh
```

This script:

- Creates the required directory structure for the experiments.
- Configures the corresponding paths used throughout the project.

---

## 4. Preprocess the dataset

Run the preprocessing scripts:

```bash
python preprocess.py
python gen_test.py
```

These scripts generate the processed datasets required for training and evaluation.

---

## 5. Train and validate the models

Launch the training jobs using:

```bash
sbatch launch.sh
```

This script submits the training and validation jobs to **SLURM**.

The experiments to be executed can be modified directly inside `launch.sh`.

---

## 6. Evaluate on the test set

Submit the test jobs with:

```bash
sbatch test.sh
```

As with the training script, the experiments to evaluate can be selected by editing `test.sh`.

---

## 7. Compute ensemble results

After all individual models have been evaluated, obtain the ensemble results by running:

```bash
python custom_validator.py
```

---

## 8. Benchmark (optional)

To reproduce the benchmarking experiments reported in the paper, execute:

```bash
sbatch bench.sh
```

This script performs the complete benchmarking procedure described in the article.
