import numpy as np

def is_grid_valid(grid, row, column, new_number):
    check_row = new_number not in grid[row]
    check_column = new_number not in [grid[i][column] for i in range(9)]
    check_box = new_number not in [grid[i][j] for i in range((row//3)*3, (row//3)*3 + 3) for j in range((column//3)*3, (column//3)*3 + 3)]
    return (check_row and check_column and check_box)

def solve_sudoku_alg(grid, row, column):
    if row == 9:
        return True # we checked every row
    elif column == 9:
        return solve_sudoku_alg(grid, row+1, 0) # time to change the row
    elif grid[row][column]!=0:
        return solve_sudoku_alg(grid, row, column+1) # nothing to do right here
    else:
        for number in range(1,10):
            if is_grid_valid(grid, row, column, number):
                grid[row][column] = number
                if solve_sudoku_alg(grid, row, column+1):
                    return True # we want to check if that number is correct in that place
                grid[row][column] = 0


class SudokuSolver:

    def __init__(self, sudoku):

        # when testing number detection, comment out self.get_dummy_sudokus line; when testing algorithm, comment out self.sudoku = sudoku line
        # self.sudoku = sudoku
        self.sudoku = self.get_dummy_sudokus("algorithm_testing_sudokus.txt")[0]
        self.grid = [[0] * 9 for _ in range(9)]
        self.solved_sudoku = self.solve_sudoku()
        self.solution_image = self.create_solution_image()

    def prepare_sudoku(self):
    
        for i in range(9):
            for j in range(9):
                char = self.sudoku[i * 9 + j]
                if char != '.':
                    self.grid[i][j] = int(char)

    def solve_sudoku(self):
        '''
        algorithm to solve sudoku
        '''
        self.prepare_sudoku()
        solve_sudoku_alg(self.grid, 0, 0)
        print(self.grid)
        return self.grid
    
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