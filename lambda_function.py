import json

# import boto3      #TODO uncomment in lambda
# this library allows interaction with AWS services

from time import gmtime, strftime
from io import BytesIO
import base64
import uuid

from PIL import Image
import numpy as np

from sudoku_extractor import SudokuExtractor

now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


def lambda_handler(event, context):
    image_data = event['body']['image']             #get the image
    image_bytes = base64.b64decode(image_data)      #decode image
    image = Image.open(BytesIO(image_bytes))        #open the image
    

    img_array = np.array(image)

    # extract the sudoku and create sudoku array
    sudoku_extractor = SudokuExtractor(image=img_array)
    sudoku = sudoku_extractor.sudoku_array

    # send sudoku array back to user so that they can correct it
    sudoku_response_array = sudoku
    print('sudoku array: ', sudoku_response_array)
    
    response_body = {'sudoku': sudoku_response_array}
        
    response = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response_body)
    }

    return response


#TODO comment in lambda, here for testing
if __name__ == '__main__':

    #current_directory = os.getcwd()
    #print("Current working directory:", current_directory)

    img_path = 'test_sudokus/5.jpg'
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()

    encoded_data = base64.b64encode(image_data).decode('utf-8')

    event = {'body': {'image': encoded_data}}
    response = lambda_handler(event, None)