# Tree Model Comparison Regression

This repository contains a collection of regression models using different tree-based algorithms for predicting house prices. The goal is to compare the performance of various tree-based regression models such as Random Forest, Extra Trees, AdaBoost, and XGBoost.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the project, you need to install the necessary dependencies. Follow these steps to set up your environment.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Matlpb/Tree-Model-Comparison-Regression.git
   cd Tree-Model-Comparison-Regression

2. **Create a virtual environment:**
   python3 -m venv venv

On macOS/Linux:

source venv/bin/activate

On Windows:

.\venv\Scripts\activate

## Dependencies
   pip install -r requirements.txt

## Usage
   python main.py

## File Structure

The project has the following directory and file structure:

- `main.py`: The main script that trains models, makes predictions, and generates results.
- `requirements.txt`: A file that lists the Python dependencies required for the project.
- `data/`: Folder where data files (e.g., `train.csv`, `test.csv`) will be stored after downloading.
- `README.md`: This file with the project overview, setup instructions, and guidelines.
- `.gitignore`: Specifies which files and directories should be ignored by Git (e.g., `venv/`, `.DS_Store`).
- `venv/`: The virtual environment directory (not tracked in Git).

### `engine/`: 
Contains the core functions and logic for training models, preprocessing data, and making predictions.

- `processing.py`: Contains functions for data preparation (e.g., encoding categorical variables, handling missing values, etc.).
- `modeling.py`: Contains functions for training and evaluating regression models (e.g., Random Forest, XGBoost, etc.).

### `visualizations/`: 
Contains scripts for generating plots and visualizations related to the model's performance and predictions.

- `plots.py`: Contains functions to generate plots (e.g., comparing actual vs predicted values, distribution of house prices, etc.).
- `metrics.py`: Contains functions to visualize model performance metrics (e.g., R-squared, Mean Squared Error).


## Contributing
If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. All contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.



