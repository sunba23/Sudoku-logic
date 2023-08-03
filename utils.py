import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.datasets import mnist

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
        transformed_image = self.zoom_at(transformed_image, zoom=1.04, coord=(side_length/2, side_length/2))    #TODO: find better solution for getting rid of the black border
        return transformed_image

    @staticmethod
    def zoom_at(img, zoom=1, angle=0, coord=None):
        cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
        rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def get_corner_points(contour):
        contour_points = np.array(contour)
        hull = cv2.convexHull(contour_points)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)
        return approx_corners[:, 0, :]

    def segment_image(self, sudoku_image):
        # dividing sudoku image to get 81 cell images
        cells = []
        cell_width = sudoku_image.shape[0] // 9
        cell_height = sudoku_image.shape[1] // 9
        for i in range(9):
            for j in range(9):
                cell = sudoku_image[i*cell_width:(i+1)*cell_width, j*cell_height:(j+1)*cell_height]
                cell = self.zoom_at(cell, zoom=1.3, coord=(cell_width/2, cell_height/2))    #TODO fix; this works well but is kinda silly
                cells.append(cell)
        # # visualize cells
        # for i in range(5):
        #     cv2.imshow('cell', cv2.resize(cells[i], (100, 100)))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return cells

    @staticmethod
    def preprocess_cell_image(cell_image):
        cell_image = cv2.resize(cell_image, (28, 28))
        cell_image = np.invert(cell_image)
        #normalize pixels to range [0, 1]
        cell_image = cell_image.astype('float32') / 255
        cell_image = np.expand_dims(cell_image, axis=0)
        return cell_image

    def predict_numbers(self, cells_images, threshold=0.95):
        model = load_model('digit_recognition_model_new_2.keras')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        sudoku_array = []
        cells_images = enumerate(cells_images)
        for index, cell_image in cells_images:
            cell_image = self.preprocess_cell_image(cell_image[:, :, 0])
            prediction = model.predict(cell_image)
            predicted_number = np.argmax(prediction)
            max_prob = np.max(prediction) 
            if max_prob >= threshold:
                sudoku_array.append(predicted_number)
            else:
                sudoku_array.append(0)
            # #TESTING PREDICTIONS: show and predict numbers on the first 9 cells
            # if index < 9:
            #     print('Prediction:', predicted_number, 'Probability:', max_prob)
            #     cv2.imshow('cell', cv2.resize(cell_image.squeeze(), (100, 100)))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
        print('sudoku array: ', sudoku_array)
        return sudoku_array