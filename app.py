from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from sudoku_extractor import SudokuExtractor
from sudoku_solver import SudokuSolver

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_sudoku():
    try:
        image_data = request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        img_array = np.array(image)
        sudoku_extractor = SudokuExtractor(image=img_array)
        sudoku = sudoku_extractor.sudoku_array

        response_body = {'sudoku': sudoku}
        return jsonify(response_body), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    try:
        print("sudoku is: ", request.json['sudoku'])
        correct_sudoku_str = request.json['sudoku']
        solved_sudoku = SudokuSolver(correct_sudoku_str).solved_sudoku

        # flatten the solved sudoku array of arrays
        solved_sudoku = flatten_extend(solved_sudoku)
        result = ""
        for number in solved_sudoku:
            result += str(number)

        response_body = {'sudoku': result}
        return jsonify(response_body), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
