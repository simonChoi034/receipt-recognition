import os

from skimage import io

error_image = []

dir = input("image dir: ")

for img in os.listdir(dir):
    a = io.imread(dir + img)
    if len(a.shape) != 3 or a.shape[2] != 3:
        print(img)
        error_image.append(img)

print("Image that is not in RGB channel:")
print(error_image)
