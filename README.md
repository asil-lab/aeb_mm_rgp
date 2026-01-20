# Multiple Model Recursive Gaussian Process for Robust Target Tracking

## Paper Title
**Multiple Model Recursive Gaussian Process for Robust Target Tracking**

## General Description

This repository contains the code required to replicate the experiments presented in the paper  
[*Multiple Model Recursive Gaussian Process for Robust Target Tracking*](https://ieeexplore.ieee.org/document/11304544),  
published in the **IEEE Open Journal of Signal Processing**.

## Requirements

### Software Requirements

- Tested on Python 3.11, newer version should also work fine
  
### Hardware Requirements

- Tested on macOS

### Setup Instructions

Follow the steps below to set up the project on your local machine:

1. Clone this repository:

    ```bash
    git clone https://github.com/asil-lab/aeb_mm_rgp
    ```

2. Navigate into the project directory:

    ```bash
    cd aeb_mm_rgp
    ```

3. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## File/Folder Details

Here is a brief overview of the structure of the repository:

```
aeb_mm_rgp/
│
├── src/                        # Contains the source code for the project
│   ├── algorithms.py           # Contains Python classes for the algorithms of interest
│   └── utils.py                # Utility functions
│
├── data/                       # Required data is extracted here automatically
│
├── figures/                    # Figures are saved in this folder
│
├── simulation_mc.ipynb         # Monte Carlo simulations
├── uzh_data_experiment.ipynb   # Real data experiment with UZH data
├── simulation_single_run.ipynb # Single run plots on simulation data
├── toy_scenario.ipynb          # Plots for the toy scenario
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation
```


## Cite This Paper

If you use this code in your research, please cite the following paper:

```bibtex
@ARTICLE{11304544,
  author={BALCı, ALI EMRE and Rajan, Raj Thilak},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Multiple Model Recursive Gaussian Process for Robust Target Tracking}, 
  year={2026},
  volume={7},
  pages={23--31},
  keywords={Target tracking;Kernel;Radar tracking;Computational modeling;Vehicle dynamics;Tuning;Gaussian processes;Adaptation models;Vectors;Real-time systems;Gaussian process;target tracking;online learning;adaptive filtering},
  doi={10.1109/OJSP.2025.3646127}
}
