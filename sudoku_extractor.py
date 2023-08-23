from utils import Utils

class SudokuExtractor:
    
    def __init__(self, image):
        self.utils = Utils()
        self.image = image
        self.image_copy = image.copy()

        self.preprocess_image_for_sudoku_edge_detection()
        self.sudoku_grid = self.find_grid()
        self.sudoku_array = self.find_numbers()

    def preprocess_image_for_sudoku_edge_detection(self):
        '''
        preprocess original image for edge detection
        '''
        self.image = self.utils.preprocess_original_image(self.image)

    def find_grid(self):
        '''
        find sudoku grid in the preprocessed sudoku image
        '''
        contour = self.utils.largest_contour(self.image.copy())
        perspective_transformed_cut_out_sudoku = self.utils.cut_out_sudoku(self.image_copy, contour)
        return perspective_transformed_cut_out_sudoku
    
    def find_numbers(self):
        '''
        find the numbers in the preprocessed sudoku grid and return sudoku array
        '''
        cells_images = self.utils.segment_image(self.sudoku_grid)
        sudoku_array = self.utils.predict_numbers(cells_images)
        print(len(sudoku_array))
        return sudoku_array