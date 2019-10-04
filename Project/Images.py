import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def main():
    #image = Image.open('DIC-C2DH-HeLa\01\t001.tif')
    #image.show()
    image = plt.imread('man_track009.tif')

    plt.imshow(image)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    #input in the console is the number of the task
    main()