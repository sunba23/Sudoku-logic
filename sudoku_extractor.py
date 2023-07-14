from utils import Utils


class SudokuExtractor:
    
    def __init__(self, image):
        self.utils = Utils()
        self.image = image
        self.preprocess()
        self.sudoku_grid = self.find_grid()
        self.sudoku_array = self.find_numbers()

    def preprocess(self):
        self.image = self.utils.preprocess_image(self.image)

    def find_grid(self):
        '''
        find the sudoku grid in the image
        '''
        contour = self.utils.largest_contour(self.image.copy())
        perspective_transformed_cut_out_sudoku = self.utils.cut_out_sudoku(self.image.copy(), contour)
        return perspective_transformed_cut_out_sudoku
    
    def find_numbers(self):
        '''
        find the numbers in the sudoku grid
        '''
        return self.sudoku_grid