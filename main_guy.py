import re
import os
import cv2
import copy
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
import functions

#####-----------------Start editing-----------------#####
IMG_DIR = 'images/'
image_name = 'cream.jpg'
conf_thresh = 40.
#####-----------------End editing-----------------#####



# Plot original image
original_image = cv2.imread(os.path.join(IMG_DIR, image_name))
image = copy.deepcopy(original_image)
d = pytesseract.image_to_data(image, output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    if float(d['conf'][i]) > conf_thresh:
        # write your function here
        if len(d['text'][i].split('/')) > 1:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # no need for rectangle, just output the date
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
show_img = functions.show_plot(rgb_img)

# Output with outputbase digits

custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(image, config=custom_config))