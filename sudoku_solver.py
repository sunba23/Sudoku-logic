import numpy as np


class SudokuSolver:

    def __init__(self, sudoku):
        self.sudoku = sudoku
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
    