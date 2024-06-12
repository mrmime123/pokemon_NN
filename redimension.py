import os
from PIL import Image

def redimension():
    for file in os.listdir('.\\test'):
        if file.endswith('.png'):
            path = os.path.join('.\\test', file)
            image = Image.open(path)
            image = image.resize((103, 113))
            image.save(path)


if __name__ == '__main__':
    redimension()