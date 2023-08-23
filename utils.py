import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.datasets import mnist
import pytesseract

class Utils:

    def preprocess_original_image(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 3)
        img_threshold = self.thresholdify_image(img_blur)
        return img_threshold

    def thresholdify_image(self, image):
        img_threshold = cv2.adaptiveThreshold(image.astype(np.uint8),
                                              255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 11, 3)
        return 255 - img_threshold

    def largest_contour(self, image):
        contours, h = cv2.findContours(
                image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    def cut_out_sudoku(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        image = image[y:y + h, x:x + w]
        contour[:, 0, 0] -= x
        contour[:, 0, 1] -= y
        return self.transform_perspective(image, contour)

    def transform_perspective(self, image, contour, side_length=300):
        corners = self.get_corner_points(contour)
        # # drawing contour and corners for visualization
        # img_copy = image.copy()
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(img_copy, [contour], -1, (30, 30, 100), 10)
        # for corner in corners:  
        #     cv2.circle(img_copy, tuple(corner), 10, (0, 255, 0), -1)
        # cv2.imshow('image', cv2.resize(img_copy, (500, 500)))
        # cv2.waitKey(0)
        pt_A, pt_B, pt_C, pt_D = corners[0], corners[1], corners[2], corners[3]
        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                        [0, side_length - 1],
                        [side_length - 1, side_length - 1],
                        [side_length - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        transformed_image = cv2.warpPerspective(image,M,(side_length, side_length),flags=cv2.INTER_LINEAR)
        transformed_image = cv2.flip(transformed_image, 1)  #TODO see why image is unflipped in the first place
        return transformed_image

    @staticmethod
    def zoom_at(img, zoom=1, angle=0, coord=None):
        cy, cx = [ i/2 for i in img.shape ] if len(img.shape) == 2 else (coord[1], coord[0])
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        result = cv2.warpAffine(img, rot_mat, img.shape[::-1], flags=cv2.INTER_LINEAR)
        return result


    @staticmethod
    def get_corner_points(contour):
        contour_points = np.array(contour)
        hull = cv2.convexHull(contour_points)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)
        return approx_corners[:, 0, :]

    def segment_image(self, sudoku_image):
        cells = []
        img_gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 3)
        img_threshold = self.thresholdify_image(img_blur)
    # first method (less reliable): dividing sudoku image using contours
    #     contours, h = cv2.findContours(
    #             img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         middle_point = (x + w // 2, y + h // 2)
    #         if w > 20 and h > 20 and w < 100 and h < 100:
    #             cell_with_coords = sudoku_image[y:y + h, x:x + w], middle_point
    #             cells.append(cell_with_coords)
    #     # sorting cells by y coordinate
    #     cells.sort(key=lambda x: x[1][1])
    #     #creating 9 lists of 9 cells
    #     cells = [cells[i:i + 9] for i in range(0, len(cells), 9)]
    #     #sorting each row by x coordinate
    #     for row in cells:
    #         row.sort(key=lambda x: x[1][0])
    #     #flattening the list
    #     cells = [cell for row in cells for cell in row]
    #     #removing coordinates so that only images remain
    #     cells = [cell[0] for cell in cells]
    #     return cells

    # second method: dividing sudoku image into 81 cells using numpy slicing
        cell_side_length = sudoku_image.shape[0] // 9
        for i in range(9):
            for j in range(9):
                cell = sudoku_image[i*cell_side_length:(i+1)*cell_side_length, j*cell_side_length:(j+1)*cell_side_length]
                cells.append(cell)
        return cells

    @staticmethod
    def preprocess_cell_image(cell_image):
        # for tesseract to work properly, we need to preprocess the image
        cell_image[cell_image < 30] = 0
        cell_image[cell_image >= 100] = 255
        zoomed = Utils.zoom_at(cell_image, zoom=1.21)
        denoised = cv2.fastNlMeansDenoising(zoomed, None, 10, 7, 21)
        img_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        bordered = cv2.copyMakeBorder(img_rgb, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return bordered

    def predict_numbers(self, cells_images):
        #! in lambda use tesseract lambda layer
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        sudoku_array = []
        for cell_image in cells_images:
            cell_image = self.preprocess_cell_image(cell_image[:, :, 0])
            prediction = pytesseract.image_to_string(cell_image, config='--psm 10 -c tessedit_char_whitelist=123456789')
            # if the prediction is neither a single digit or empty, then tesseract probably detected either the left or right border of the cell as '1'
            # and created something like '121' or '12' or '21' instead of '2'
            if len(prediction) > 1:
                if prediction[0] == '1':
                    prediction = prediction[1]
                elif prediction[-1] == '1':
                    prediction = prediction[-2]
            sudoku_array.append(prediction)

        # doing this because tesseract returns stuff like '1\n' instead of '1'
        converted_sudoku = [cell.strip() for cell in sudoku_array]
        return converted_sudoku