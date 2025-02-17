# Tree Model Comparison Regression

This repository contains an API to make predictions on a given dataset (here, house prices from Kaggle). The user can either propose data of their choice for testing with respect to the original training dataset or provide a number that will load the corresponding line in the testing file. The API will then provide a prediction using the trained model.

The user can check the embedding of their testing dataset (whether it is preloaded or chosen by the user) in the file `X_test_transformed.csv`. This file will be overwritten each time the user changes the input data.

Model Training :
If the user wants, they can retrain models.

For further details on how the API works, follow the instructions below.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

**Clone the repository:**

```bash
git clone https://github.com/Matlpb/Tree-Model-Comparison-Regression.git
cd Tree-Model-Comparison-Regression
```

## Usage

### 1. Launch the API

Use Python 3.10.16.

#### On macOS/Linux:

```bash
chmod +x start_api.sh
./start_api.sh
```

If it does not work, try:

```bash
source venv/bin/activate
./start_api.sh
```

#### On Windows:


double click on the file `start_api.bat`


If it does not work, try:
```bash
start start_api.bat
```
Else: 
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn API.app:app --reload
```

It might take a few seconds to install dependencies and launch the API. Then, copy and paste the provided link into a browser.

### 2. Training/Re-Training

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The following folder must be deleted before retraining:

- `models`

You may also delete the following folders:

- `house_prices/data`
- `transformer_params`
- `data_test`

To retrain the model, run:

```bash
python main.py
```

In the code, the seed has been set, so the training results will always be the same as long as the seed remains unchanged.

### 3. Use the API

The user can enter their own data on the interface. The required fields and their types will be displayed, preventing incorrect data input. Any columns left unfilled will be set as NaN and encoded as 0 in the file X_test_transformed.csv. The user can then select the model for prediction and click on "Predict."

If the user does not wish to enter data manually, they can input a number corresponding to a line in the "test.csv" file, click "Load Raw," select the model, and then click "Predict."

If the user initially enters a number but later wishes to input custom data, they can delete the field "Enter a number between 1-1450," click "Load Raw," enter the desired data, and then click "Predict."

## Contributing
If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. All contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
