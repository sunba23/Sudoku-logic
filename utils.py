import cv2  #! not built in

def preprocess(image):
    img_blur = cv2.GaussianBlur(image, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_threshold
