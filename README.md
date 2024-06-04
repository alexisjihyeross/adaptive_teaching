# Toward In-Context Teaching

This repository contains code for our paper, [Toward In-Context Teaching: Adapting Examples to Students' Misconceptions](https://arxiv.org/abs/2405.04495).

## Citation
```bibtex
@inproceedings{ross2024incontext,
    title = "Toward In-Context Teaching: Adapting Examples to Students' Misconceptions",
    author = "Alexis Ross and Jacob Andreas",
    booktitle = "ACL 2024",
    publisher = "Association for Computational Linguistics",
    url= "https://arxiv.org/abs/2405.04495",
}
```

## Installation

1.  Clone the repository.
    ```bash
    git clone https://github.com/alexisjihyeross/pedagogy
    cd mice
    ```

2.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

3.  Create a Conda environment.

    ```bash
    conda create -n pedagogy python=3.7
    ```
 
4.  Activate the environment.

    ```bash
    conda activate pedagogy
    ```
    
5.  Download the requirements.

    ```bash
    pip3 install -r requirements.txt
    ```

6.  <b>Set Environment Variables</b> 

    Experiments with GPT-based models require setting OPENAI environment variables.

    ```bash
    export OPENAI_API_KEY={KEY}
    export OPENAI_ORGANIZATION={KEY}
    export PYTHONPATH=./
    ```

## Synthetic Experiments

1.  <b>Run Experiments</b>

    The scripts below contain code for evaluating AToM, GPT4-based teachers, and other baselines with the synthetic learners in the AdapT evaluation framework.

    - <b>Functions</b>: [`scripts/run_functions.sh`](scripts/run_functions.sh)

    - <b>Fractions</b>: [`scripts/run_fractions.sh`](scripts/run_fractions.sh)

    - <b>Verbs</b>: [`scripts/run_verbs.sh`](scripts/run_verbs.sh)

    For example, to run experiments for functions, you could use the following command:

    ```bash
    bash scripts/run_functions.sh
    ```

    The code defaults to logging with wandb. Set the `WANDB_PROJECT` variable in these scripts to determine which wandb projects results are logged to.

2. <b>View Results</b>

    You can use the following command to download results from wandb:

    ```bash
    python src/analyze.py --entity ${ENTITY} --project ${PROJECT}
    ```

## Human Experiments

The script [`scripts/run_human.sh`](scripts/run_human.sh) contains the script for running a server for human experiments. 

By default, it runs the experiments in the paper: 22 experimental conditions (11 target concepts, 2 student types), 5 seeds each, for 3 different teachers: Random, ATOM, and GPT4. 

Results are saved locally to `results/human/experiments`.