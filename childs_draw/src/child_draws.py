# Examples
#(img, df, imageHeader) = load_image_data(7554, 10)
#(img, df, imageHeader) = load_image_data(7557, 12)

img.shape

# Imports
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

class ImageHeader:
    def __init__(self, img, df):
        self.image_size = (img.shape[0], img.shape[1])
    
def load_image_data(folder_id: int, image_id: int):
    file_name = './data/' + str(folder_id) + '/SimpleTest/' + str(image_id)
    
    if not os.path.isfile(file_name + '.png'):
        print("Image not exits")
        return None
    
    if not os.path.isfile(file_name + '.csv'):
        print("Data not exits")
        return None
    
    img = cv2.imread(file_name + '.png', cv2.IMREAD_GRAYSCALE)
    
    df = pd.read_csv(file_name + '.csv')
    
    return (img, df, ImageHeader(img, df))

show_counter = 0
    
def save_image(img, imageHeader=None, resize=True):
    global show_counter
    show_counter += 1
    number_as_str = str(show_counter)
    while len(number_as_str) < 3:
        number_as_str = '0' + number_as_str
    file_name = 'show_' + number_as_str + '.png'
    
    if imageHeader != None:
        if resize:
            img = img.copy()
            img.shape = imageHeader.image_size
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    return cv2.imwrite(file_name, img)

df.columns = map(lambda x: x.replace(" ", ""), df.columns.tolist())

last_message1 = df.iloc[-2]['X']
last_message2 = df.iloc[-1]['X']

df = df.iloc[:-2].copy()

print(last_message1)
print(last_message2)

number_columns = list(filter(lambda x: x != 'Time', df.columns.tolist()))

last_value = 0
def to_float(x):
    global last_value
    try:
        x = float(x)
        last_value = x
    except:
        x = last_value
    return x

for column in number_columns:
    last_value = 0
    df[column] = df[column].apply(to_float)

last_value = 0
def time_to_number(x):
    global last_value
    if not x:
        return last_value
    
    if type(x) == float:
        return x
    
    (m, s) = x.split(':')
    m = float(m)
    s = float(s)
    return m * 60 + s

df['Time'] = df['Time'].apply(time_to_number)

def add_column_diff(column):
    global df
    now = np.array(df[column].tolist() + [df[column][len(df[column])-1]])
    before = np.array([0] + df[column].tolist())
    df[column + 'Diff'] = (now - before)[1:]
    
    fig = plt.figure()
    fig.set_size_inches(18, 6)
    
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(column)
    ax.plot(df[column])
    
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(column + ' Diff')
    ax.plot(df[column + 'Diff'])

    plt.show()

add_column_diff('Time')
add_column_diff('Pressure')
add_column_diff('X')
add_column_diff('Y')
add_column_diff('TiltX')
add_column_diff('TiltY')

img_show = img.copy()

for x in np.arange(int(df['X'].min()), int(df['X'].max()) + 1, 1):
    for y in np.arange(int(df['Y'].min()), int(df['Y'].max()) + 1, 1):
        img_show[y,x] = 255
save_image(img_show)
print(int(df['Y'].max()), " ", int(df['X'].max()))
img_show.shape

drawed_img = img.copy()

bound_x = [int(df['X'].min()), int(df['X'].max())]
if bound_x[1] < df['X'].max():
    bound_x[1] += 1

bound_y = [int(df['Y'].min()), int(df['Y'].max())]
if bound_y[1] < df['Y'].max():
    bound_y[1] += 1

drawed_img = drawed_img[bound_y[0]:bound_y[1], bound_x[0]:bound_x[1]]

save_image(drawed_img)

drawed_img_backup = drawed_img

drawed_img = np.ndarray((drawed_img_backup.shape[0] + 2, drawed_img_backup.shape[1] + 2))
drawed_img[:, :] = 255
drawed_img[1:-1, 1:-1] = drawed_img_backup

drawed_img = drawed_img > 255/2
drawed_img = drawed_img * 255

save_image(drawed_img)

full_drawed_img = drawed_img.copy()

def save_trace_image(img):
    global show_counter
    show_counter += 1
    number_as_str = str(show_counter)
    while len(number_as_str) < 3:
        number_as_str = '0' + number_as_str
    file_name = 'show_' + number_as_str + '.png'
    
    rgb_img = np.ndarray((img.shape[0], img.shape[1], 3))
    
    rgb_img[:, :, :] = 0
    
    found = np.where(trace_img == 1)
    for i in range(len(found[0])):
        rgb_img[found[0][i],found[1][i], 0] = 255

    found = np.where(trace_img == 2)
    for i in range(len(found[0])):
        rgb_img[found[0][i],found[1][i], 2] = 255
    
    rgb_img[(rgb_img[:, :, 0] == 0) & (rgb_img[:, :, 1] == 0) & (rgb_img[:, :, 2] == 0)] = 255
    
    return cv2.imwrite(file_name, rgb_img)

trace_img = drawed_img.copy()
trace_img[drawed_img == 0] = 1

save_trace_image(trace_img)

# try 2
def red_spider(img):
    
    # color bounds
    img[[0,-1], :] = 2
    img[:, [0,-1]] = 2
    
    points = list()
    for x in range(img.shape[0]):
        points.append((x, 1))
        points.append((x, img.shape[1] - 2))

    for y in range(img.shape[1]):
        points.append((1, y))
        points.append((img.shape[0] - 2, y))
    
    steps = 0
    while len(points) > 0:
        (x,y) = points.pop()
        if img[x,y] == 0:
            steps += 1
            
            if steps % 10**5 == 0:
                 print(steps, len(points))
            
            img[x,y] = 2            
            for n in [0, 1, 2, 3, 5, 6, 7, 8]:
                dx = n % 3 - 1
                dy = int(n/3) - 1
                points.append((x+dx, y+dy))
        
    return img

# load
trace_img = drawed_img.copy()
trace_img[drawed_img == 0] = 1
trace_img[drawed_img == 255] = 0

trace_img = red_spider(trace_img)
save_trace_image(trace_img)

if (trace_img == 0).sum() == 0:
    print("Not exists open shapes")
else:
    print("Exists open shapes")
