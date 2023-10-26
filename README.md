# Indirect reciprocity with quantitative assessment

This repository is the code about simulations and numerical calculations of the paper

- *Schmid, Laura, et al. "Quantitative assessment can stabilize indirect reciprocity under imperfect information."
  Nature Communications 14.1 (2023): 2086.*

```bibtex
@article{schmid2023quantitative,
  title={Quantitative assessment can stabilize indirect reciprocity under imperfect information},
  author={Schmid, Laura and Ekbatani, Farbod and Hilbe, Christian and Chatterjee, Krishnendu},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={2086},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Introduction

This repository provides the **Python3** script to simulate the reputation dynamics and calculate the selection-mutation
equilibrium when players can choose between three strategies:

- a leading eight norm with quantitative assessment,
- ALLC,
- ALLD.

To exemplify this, this repository also provides sample data of a simulation run with social norm L1.
Additionally, this repository provides the code used for Fig. S3 (Recovery times), both for norms with one absorbing
state, and norms with two absorbing states.

## Running

### Requirement

The requirement of this repository is

```commandline
deeptime 0.4.4 Python library for analysis of time series data including dimensionality reduction, clustering, and Markov model estimation.
├── numpy >=1.20
├── scikit-learn >=1.1
│   ├── joblib >=1.1.1 
│   ├── numpy >=1.17.3,<2.0 
│   ├── scipy >=1.5.0 
│   │   └── numpy >=1.21.6,<1.28.0 (circular dependency aborted here)
│   └── threadpoolctl >=2.0.0 
├── scipy >=1.9.0
│   └── numpy >=1.21.6,<1.28.0 
└── threadpoolctl >=3.1.0
numpy 1.26.1 Fundamental package for array computing in Python
pytz 2023.3.post1 World timezone definitions, modern and historical
pyyaml 6.0.1 YAML parser and emitter for Python
tqdm 4.66.1 Fast, Extensible Progress Meter
└── colorama *
tzlocal 5.2 tzinfo object for the local timezone
└── tzdata *
```

You can run the simulating the reputation dynamics and calculating the selection-mutation equilibrium in a population where players can choose between three strategies of this repository by

```shell
sh ./eval.sh
```

or

```shell
python src/evol_quant_levels_matrix/evol.py --multiprocess --leading8idx=0
```

For the simulating the recovery process

```shell
sh ./recovery.sh
```

or

```shell
python src/recovery/recovery.py --leading8idx=0
```

where `--multiprocess` flag indicate multiprocess will be used to accelerate this program, `--leading8idx` flag identify the `$(the index of leading eight norm)-1`.

### Via [poetry](https://python-poetry.org/)

Make sure the [`poetry`](https://python-poetry.org/) are met.

#### Installation [`poetry`](https://python-poetry.org/)

You can install [`poetry`](https://python-poetry.org/) by following
the [guide](https://python-poetry.org/docs/#installation).

Or you can just install [`poetry`](https://python-poetry.org/) manually using `pip` and the `venv` module in `.venv/`:

```shell
export VENV_PATH="$(pwd)/.venv"
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/poetry install
```

and execute by

```shell
$VENV_PATH/bin/python src/evol_quant_levels_matrix/evol.py --multiprocess --leading8idx=0

$VENV_PATH/bin/python src/recovery/recovery.py --leading8idx=0
```

## License

The code in this repository is released under the MIT License.
