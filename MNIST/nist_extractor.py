import os
import sys
from struct import *


def read_images(file_name):
    images = []
    with open(file_name, "rb") as image_f, open(file_name.replace("images-idx3", "labels-idx1"), 'rb') as label_f:
        magic_num_image = unpack('>i', image_f.read(4))[0]
        magic_num_label = unpack('>i', label_f.read(4))[0]
        num_items = unpack('>i', image_f.read(4))[0]
        label_items = unpack('>i', label_f.read(4))[0]
        if num_items != label_items:
            print("Mismatch in items!")
            sys.exit()
        num_rows = unpack('>i', image_f.read(4))[0]
        num_columns = unpack('>i', image_f.read(4))[0]

        while len(images) != num_items:
            image = []
            for row_num in range(num_rows):
                row = image_f.read(num_columns)
                image.append(row)
            label = unpack('b', label_f.read(1))[0]
            images.append([label, image])
    return images


def find_and_read_images():
    found_images = False
    t10k = ""
    train = ""

    cwd = os.getcwd()
    cwd_parts = os.path.split(cwd)
    while len(cwd_parts) > 1 and not found_images:
        for root, dirs, files in os.walk(cwd):
            if "run_" in root or ".git" in root or "venv" in root:
                continue
            for file in files:
                if "t10k-images-idx3-ubyte" in file:
                    t10k = os.path.join(root, file)
                elif "train-images-idx3-ubyte" in file:
                    train = os.path.join(root, file)
            if t10k != "" and train != "":
                found_images = True
                break
        cwd = os.path.join(*cwd_parts[:-1])
    if not found_images:
        print("Couldn't find training images!")
        sys.exit()
    t10k_images = read_images(t10k)
    train_images = read_images(train)
    return t10k_images, train_images
