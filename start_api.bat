@echo off
REM Create a virtual environment
python3.10 -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependances
pip install -r requirements.txt

REM Launch the API
uvicorn API.app:app --reload
