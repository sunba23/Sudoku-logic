import cv2
import numpy as np
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
        pt_A, pt_B, pt_C, pt_D = corners[0], corners[1], corners[2], corners[3]
        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                        [0, side_length - 1],
                        [side_length - 1, side_length - 1],
                        [side_length - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        transformed_image = cv2.warpPerspective(image,M,(side_length, side_length),flags=cv2.INTER_LINEAR)
        transformed_image = cv2.flip(transformed_image, 1)
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
        if len(cell_image.shape) == 3:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        cell_image[cell_image < 30] = 0
        cell_image[cell_image >= 100] = 255

        line_image = cell_image.copy()
        h, w = line_image.shape[:2]
        border_size = 6
        cv2.line(line_image, (0, 0), (w, 0), 255, border_size)
        cv2.line(line_image, (0, h), (w, h), 255, border_size)
        cv2.line(line_image, (0, 0), (0, h), 255, border_size)
        cv2.line(line_image, (w, 0), (w, h), 255, border_size)

        return line_image

    def predict_numbers(self, cells_images):
        sudoku_string = ''
        for cell_image in cells_images:
            prediction = pytesseract.image_to_string(cell_image, config='--psm 10 -c tessedit_char_whitelist=123456789')
            # doing this because tesseract returns stuff like '1\n' instead of '1'
            prediction = prediction.strip()
            # if the prediction is neither a single digit or empty, then tesseract probably detected either the left or right border of the cell as '1'
            # and created something like '121' or '12' or '21' instead of '2'
            if len(prediction) > 1:
                if prediction[0] == '1':
                    prediction = prediction[1]
                elif prediction[-1] == '1':
                    prediction = prediction[-2]
            if prediction == '' or int(prediction) > 9:
                prediction = '.'
            sudoku_string += prediction

        return sudoku_string
    