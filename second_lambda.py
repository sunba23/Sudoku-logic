import json

# import boto3      #TODO uncomment in lambda
# this library allows interaction with AWS services

from time import gmtime, strftime
import uuid

from sudoku_solver import SudokuSolver

#TODO uncomment in lambda
# dynamodb = boto3.resource('dynamodb')
# table = dynamodb.Table('SudokuResults2')

now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


def lambda_handler(event, context):

    correct_sudoku_str = event['body']['data']
    solved_sudoku = SudokuSolver(correct_sudoku_str).solved_sudoku
    
    #TODO uncomment in lambda
    # item_id = str(uuid.uuid4())                     # create unique key for id
    # table.put_item(                                 # add solution to the database
    #     Item={
    #         'ID': item_id,
    #         'result': str(solved_sudoku),
    #         'resultTimestamp': now
    #     }
    # )
    
    response_body = {'sudoku': solved_sudoku}
        
    response = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response_body)
    }

    return response


#TODO comment in lambda, here for testing
# if __name__ == '__main__':