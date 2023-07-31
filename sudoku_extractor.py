from utils import Utils


class SudokuExtractor:
    
    def __init__(self, image):
        self.utils = Utils()
        self.image = image
        self.image_copy = image.copy()

        self.preprocess_image_for_sudoku_edge_detection()
        self.sudoku_grid = self.find_grid()
        self.preprocess_grid_for_number_recognition()
        self.sudoku_array = self.find_numbers()

    def preprocess_image_for_sudoku_edge_detection(self):
        '''
        preprocess original image for edge detection
        '''
        self.image = self.utils.preprocess_original_image(self.image)

    def preprocess_grid_for_number_recognition(self):
        '''
        preprocess cut-out sudoku grid that is seen from birds eye view for number recognition
        '''
        self.sudoku_grid = self.utils.preprocess_sudoku_grid(self.sudoku_grid)

    def find_grid(self):
        '''
        find sudoku grid in the preprocessed sudoku image
        '''
        contour = self.utils.largest_contour(self.image.copy())
        perspective_transformed_cut_out_sudoku = self.utils.cut_out_sudoku(self.image_copy, contour)
        return perspective_transformed_cut_out_sudoku
    
    def find_numbers(self):
        '''
        find the numbers in the preprocessed sudoku grid
        '''
        sudoku_array = self.utils.read_numbers(self.sudoku_grid)
        return self.sudoku_grid