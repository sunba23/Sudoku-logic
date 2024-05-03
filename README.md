
# Sudoku OCR app


This is the python OCR backend used for the sudoku solving flutter mobile app (https://github.com/sunba23/Sudoku-app)
## Usage

> **⚠️** Steps explained here assume usage of local server instead of AWS.

Clone the repository and cd into it:
```bash
git clone https://github.com/sunba23/Sudoku-logic ./sudoku-logic
cd sudoku-logic
```

Install tesseract and add it to your PATH:
https://github.com/tesseract-ocr/tessdoc

create and activate a virtual environment (https://docs.python.org/3/library/venv.html) or use the global interpreter. After that, install the dependencies and run the server.

```bash
pip3 install -r requirements.txt
python3 app.py
```
Now that you have backend set up, go and use the solver app!
