from chitra.image import Chitra
import matplotlib.pyplot as plt
import os
import pandas as pd

label = 'Dog'

needed_coordinates = [0, 1, 4, 5] # these coordinates are needed to draw the bboxes on the images

bb_paths = os.listdir('images\\train\\labelTxt')  # txt file path

image_paths = os.listdir('images\\train\\images') # list of images in the folder

bb = [] # list of bbox coordinates
for txt_file in bb_paths: #get the coordinates of bboxes from txt file
    txt_path = 'images\\train\\labelTxt\\' + txt_file  # txt file path with the bbox coordinates
    data = pd.read_csv(txt_path, header = None, sep = ' ')
    #print(data.T[0].values[0])
    array = []
    for ind in needed_coordinates:
        array.append(data.T[0].values[ind])
    print(array)
    bb.append(array)


fix, ax = plt.subplots(2, 5, figsize=(25, 10))

for i, file in enumerate(image_paths): # loop over the images
    image_path = 'images\\train\\images\\' + file   # image file path
    image_orig = Chitra(image_path, bboxes=bb[i], labels=label) # image object

    image_resized = Chitra(image_path, bboxes=bb[i], labels=label)   # image object for resize
    image_resized.resize_image_with_bbox((640, 640))  # resize image
    
    #original images
    ax[0,i].axis('off')
    ax[0,i].imshow(image_orig.draw_boxes())
    #resized images
    ax[1,i].axis('off')
    ax[1,i].imshow(image_resized.draw_boxes())