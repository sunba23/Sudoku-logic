import json

# import boto3      #TODO uncomment in lambda
# this library allows interaction with AWS services

from time import gmtime, strftime
import base64
from io import BytesIO
from PIL import Image   #! not built in
import numpy as np    #! not built in
import uuid

from detect_sudoku import detect_sudoku
from solve_sudoku import solve_sudoku
from create_solution_image import create_solution_image

#TODO uncomment in lambda
# dynamodb = boto3.resource('dynamodb')
# table = dynamodb.Table('SudokuResults2')

now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


def lambda_handler(event, context):
    image_data = event['body']['image']             #get the image
    image_bytes = base64.b64decode(image_data)      #decode image
    image = Image.open(BytesIO(image_bytes))        #open the image
    
    if image.mode != 'L':                           # q: convert to grayscale?
        image = image.convert('L')
    
    img_array = np.array(image)
    sudoku = detect_sudoku(img_array)
    solved_sudoku = solve_sudoku(sudoku)
    solved_image = create_solution_image(solved_sudoku) #returns numpy array

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
    with open('test_image.txt', 'r') as f:
        image_data = f.read()
    event = {'body': {'image': image_data}}
    response = lambda_handler(event, None)