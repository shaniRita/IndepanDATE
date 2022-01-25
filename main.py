import re
import os
import cv2
import copy
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
import functions

IMG_DIR = 'images/'
image_name = 'cream.jpg'
# Plot original image
original_image = cv2.imread(os.path.join(IMG_DIR, image_name))
image = copy.deepcopy(original_image)
b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
plt.imshow(rgb_img)
plt.show()

# Preprocess image
gray = functions.get_grayscale(image)
thresh = functions.thresholding(gray)
opening = functions.opening(gray)
canny = functions.canny(gray)
images = {'gray': gray,
          'thresh': thresh,
          'opening': opening,
          'canny': canny}

# Plot images after preprocessing
fig = plt.figure(figsize=(13, 13))
ax = []

rows = 2
columns = 2
keys = list(images.keys())
for i in range(rows*columns):
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title('report - ' + keys[i])
    plt.imshow(images[keys[i]], cmap='gray')

# Plot character boxes on image using pytesseract.image_to_boxes() function
h, w, c = image.shape
boxes = pytesseract.image_to_boxes(image)
for b in boxes.splitlines():
    b = b.split(' ')
    image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
show_img = functions.show_plot(rgb_img)

# Plot word boxes on image using pytesseract.image_to_data() function

image = copy.deepcopy(original_image)
d = pytesseract.image_to_data(image, output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())
n_boxes = len(d['text'])
for i in range(n_boxes):
    # condition to only pick boxes with a confidence > 60%
    if float(d['conf'][i]) > float(40):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
show_img = functions.show_plot(rgb_img)

#Plot boxes around text that matches a certain regex template
#In this example we will extract the date from the sample invoice

image = copy.deepcopy(original_image)
date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])(/(19|20)\d\d)?$/'

n_boxes = len(d['text'])
for i in range(n_boxes):
    if float(d['conf'][i]) > float(50):
        if len(d['text'][i].split('/')) > 1:
        # if re.match(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
show_img = functions.show_plot(rgb_img)

# Output with outputbase digits

custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(image, config=custom_config))