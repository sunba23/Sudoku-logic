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
        
        #TODO: see what the image needs to look like for number recognition, perhaps transform the image w/o preprocessing

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
        print(output_pts)
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        transformed_image = cv2.warpPerspective(image,M,(side_length, side_length),flags=cv2.INTER_LINEAR)
        transformed_image = cv2.flip(transformed_image, 1)  #TODO see why image is unflipped in the first place

        return transformed_image

    @staticmethod
    def get_corner_points(contour):
        contour_points = np.array(contour)
        hull = cv2.convexHull(contour_points)

        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)

        return approx_corners[:, 0, :]

