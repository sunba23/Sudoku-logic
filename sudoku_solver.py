import numpy as np


class SudokuSolver:

    def __init__(self, sudoku):

        # when testing number detection, comment out self.get_dummy_sudokus line; when testing algorithm, comment out self.sudoku = sudoku line
        self.sudoku = sudoku
        # self.sudoku = self.get_dummy_sudokus("algorithm_testing_sudokus.txt")[0]

        self.solved_sudoku = self.solve_sudoku()
        self.solution_image = self.create_solution_image()

    def solve_sudoku(self):
        '''
        algorithm to solve sudoku
        '''
        return self.sudoku
    
    def create_solution_image(self):
        '''
        picture solution on input image
        '''
        return np.array(self.solved_sudoku)

    @staticmethod
    def get_dummy_sudokus(filename):
        '''
        get dummy sudokus for testing
        '''
        with open (filename, 'r') as f:
            sudokus = f.read().splitlines()
        return sudokus