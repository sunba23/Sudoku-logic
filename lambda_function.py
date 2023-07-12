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
from sudoku_solver import SudokuSolver

#TODO uncomment in lambda
# dynamodb = boto3.resource('dynamodb')
# table = dynamodb.Table('SudokuResults2')

now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


def lambda_handler(event, context):
    image_data = event['body']['image']             #get the image
    image_bytes = base64.b64decode(image_data)      #decode image
    image = Image.open(BytesIO(image_bytes))        #open the image
    

    img_array = np.array(image)

    sudoku_extractor = SudokuExtractor(image=img_array)
    sudoku = sudoku_extractor.sudoku_array
    sudoku_solver = SudokuSolver(sudoku=sudoku)
    solved_image = sudoku_solver.solution_image

    #TODO comment in lambda, here for testing
    solution_image = Image.fromarray(solved_image)
    solution_image.show()

    solved_image_data = base64.b64encode(solved_image).decode('utf-8')
    
    #TODO uncomment in lambda
    # item_id = str(uuid.uuid4())                     # create unique key for id
    # table.put_item(                                 # add solution to the database
    #     Item={
    #         'ID': item_id,
    #         'result': str(solved_image_data),
    #         'resultTimestamp': now
    #     }
    # )
    
    response_body = {'solution': solved_image_data}
        
    response = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response_body)
    }

    return response


#TODO comment in lambda, here for testing
if __name__ == '__main__':

    img_path = 'test_sudokus/3.jpg'
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()

    encoded_data = base64.b64encode(image_data).decode('utf-8')

    event = {'body': {'image': encoded_data}}
    response = lambda_handler(event, None)