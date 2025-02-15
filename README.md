# Tree Model Comparison Regression

This repository contains an API to make predictions on a given data set (here house prices in kaggle). The user can either propose a datas of its choice for test with respect to the original data set of train, or fill a number that will load the corresponding line in the testing file. The 
API will provide a prediction for the model used. The user can check the embedding of his testing data set (either if is preloaded, or choose by the user), inth efile "X_test_transfromed.csv". This fill will be overwrite each time that the user change the input data.
If the user wants so, he can retrain models, for that he has to delete the following folder of the repertory: models

He can also delete folders : house_prices/data, transformer_params, data_test
if he runs python main.py the following will be downloaded, then run commands to lauch the API

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Matlpb/Tree-Model-Comparison-Regression.git
   cd Tree-Model-Comparison-Regression
   ```

2. **Launch the API with the following command lines:**
   
   ```bash
   Use python 3.10.16

   if you are on mac os/linux :

   chmod +x start_api.sh
   ./start_api.sh


   if it does not work, try
    
   source venv/bin/activate
   ./start_api.sh

   On windows

   .\venv\Scripts\activate
   double click on the file start_api.bat

   if it does not work, try
   uvicorn API.app:app --reload



   It might take a few seconds either to install dependencies, and also for launching the API. 
   Then copie paste the link in a browser



   ```


## Training
   ```bash
   On mac os/linux

   source venv/bin/activate
   pip install -r requirements.txt

   On windows

   .\venv\Scripts\activate
   pip install -r requirements.txt


   python main.py
   ```


## Contributing
If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. All contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.



