from Lab4.Task1 import task_1
import sys


def main(argv):
    if argv == 1:
        history_1 = task_1()


if __name__ == "__main__":
    #input in the console is the number of the task
    main(sys.argv[1:])
