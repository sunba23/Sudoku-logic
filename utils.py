import cv2
import numpy as np


class Utils:

    def preprocess_image(self, image):
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
        print(corners)

        img_copy = image.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_copy, [contour], -1, (30, 30, 100), 10)
        for corner in corners:  
            cv2.circle(image, tuple(corner), 10, (0, 0, 0), -1)

        cv2.circle(img_copy, tuple(corners[0]), 20, (255, 0, 0), -1)
        cv2.circle(img_copy, tuple(corners[1]), 20, (0, 255, 0), -1)
        cv2.circle(img_copy, tuple(corners[2]), 20, (255, 0, 255), -1)
        cv2.circle(img_copy, tuple(corners[3]), 20, (0, 0, 255), -1)

        #show image in 500 x 500 window
        cv2.imshow('image', cv2.resize(img_copy, (500, 500)))
        cv2.waitKey(0)

        return image

    @staticmethod
    def get_corner_points(contour): #TODO: improve this garbage approximation
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        ordered_points = box.astype(int)

        top_left = ordered_points[1]
        top_right = ordered_points[2]
        bottom_right = ordered_points[3]
        bottom_left = ordered_points[0]

        return np.array([top_left, top_right, bottom_right, bottom_left])
