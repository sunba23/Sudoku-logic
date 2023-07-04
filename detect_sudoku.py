import cv2
from utils import preprocess

def detect_sudoku(image):
    preprocessed_image = preprocess(image)
    return preprocessed_image
