from Task1 import task_1
from Task2 import task_2
from Task3 import task_3
import sys


def main(argv):
    if argv == '1':
        print('doing task 1.')
        history_1 = task_1()
        
    elif argv == '2':
        print('doing task 2.')
        history_2 = task_2()
        
    elif argv == '3':
        print('doing task 3.')
        history_3 = task_3()   
        
    else:
        print('wrong task number')


if __name__ == "__main__":
    #input in the console is the number of the task
    task = input("Enter the number of task to perform: ")
    main(task)
