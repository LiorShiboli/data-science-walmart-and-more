# Examples
# import src.childs_draw as childs_draw
# childs_draw.child_draw_descrive(7554, 10, True)
# childs_draw.child_draw_descrive(7557, 12, True)

# Imports
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

data_root_folder = './data'
show_images_folder = '.'

image_show_number_size = 3

g_save_images_enabled = True
save_image_counter = 0

def set_data_root_folder(folder):
    global data_root_folder
    data_root_folder = folder
    
def set_show_images_folder(folder):
    global show_images_folder
    show_images_folder = folder
    
def load_image_data(folder_id: int, image_id: int):
    file_name = data_root_folder + '/' + str(folder_id) + '/SimpleTest/' + str(image_id)
    
    if not os.path.isfile(file_name + '.png'):
        print("Image not exits")
        return (None, None)
    
    if not os.path.isfile(file_name + '.csv'):
        print("Data not exits")
        return (None, None)
    
    img = cv2.imread(file_name + '.png', cv2.IMREAD_GRAYSCALE)
    
    df = pd.read_csv(file_name + '.csv')
    
    return (img, df)

def save_image(img):
    global g_save_images_enabled
    if not g_save_images_enabled:
        return True
    
    global save_image_counter
    global image_show_number_size
    global show_images_folder
    
    save_image_counter += 1
    number_as_str = str(save_image_counter)
    while len(number_as_str) < image_show_number_size:
        number_as_str = '0' + number_as_str
    file_name = show_images_folder + '/show_' + number_as_str + '.png'
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    return cv2.imwrite(file_name, img)

def save_trace_image(img):
    global g_save_images_enabled
    if not g_save_images_enabled:
        return True
    
    rgb_img = np.ndarray((img.shape[0], img.shape[1], 3))
    
    rgb_img[:, :, :] = 0
    
    found = np.where(img == 1)
    for i in range(len(found[0])):
        rgb_img[found[0][i],found[1][i], 0] = 255

    found = np.where(img == 2)
    for i in range(len(found[0])):
        rgb_img[found[0][i],found[1][i], 2] = 255
    
    rgb_img[(rgb_img[:, :, 0] == 0) & (rgb_img[:, :, 1] == 0) & (rgb_img[:, :, 2] == 0)] = 255
        
    return save_image(rgb_img)


def child_draw_describe(folder_id: int, image_id: int, save_images_enabled: bool = False):
    global g_save_images_enabled
    g_save_images_enabled = save_images_enabled
    
    (img, df) = load_image_data(folder_id, image_id)
    
    if img is None or df is None:
        return None
    
    results = {
        'Hands Up': None,
        'Lines Count': None,
        'Exist Open Shapes': None,
        'Open Shapes Count': None
    }
    
    df.columns = map(lambda x: x.replace(" ", ""), df.columns.tolist())
    
    # Maybe not needabel
    last_message1 = df.iloc[-2]['X']
    last_message2 = df.iloc[-1]['X']
    
    df = df.iloc[:-2].copy()
    
    # convert string columns to float
    number_columns = list(filter(lambda x: x != 'Time', df.columns.tolist()))
    
    last_value = 0
    def to_float(x):
        try:
            x = float(x)
            last_value = x
        except:
            x = last_value
        return x
    
    for column in number_columns:
        last_value = 0
        df[column] = df[column].apply(to_float)
    
    # conver Time from string to float
    last_value = 0
    def time_to_number(x):
        if not x:
            return last_value
    
        if type(x) == float:
            return x
    
        (m, s) = x.split(':')
        m = float(m)
        s = float(s)
        return m * 60 + s

    df['Time'] = df['Time'].apply(time_to_number)
    
    # create Diff columns
    def add_column_diff(column):
        now = np.array(df[column].tolist() + [df[column][len(df[column])-1]])
        before = np.array([0] + df[column].tolist())
        df[column + 'Diff'] = (now - before)[1:]
        
    add_column_diff('Time')
    add_column_diff('Pressure')
    add_column_diff('X')
    add_column_diff('Y')
    add_column_diff('TiltX')
    add_column_diff('TiltY')
    
    # get zoomed images
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

    # make it black and white
    drawed_img = drawed_img > 255/2
    drawed_img = drawed_img * 255

    save_image(drawed_img)

    full_drawed_img = drawed_img.copy()

    trace_img = drawed_img.copy()
    trace_img[drawed_img == 0] = 1

    save_trace_image(trace_img)

    # red_spider function
    def red_spider(img, points:list = None):
    
        # color bounds
        img[[0,-1], :] = 2
        img[:, [0,-1]] = 2
        
        if points is None: # First Time
            points = list()
            for x in range(img.shape[0]):
                points.append((x, 1))
                points.append((x, img.shape[1] - 2))
    
            for y in range(img.shape[1]):
                points.append((1, y))
                points.append((img.shape[0] - 2, y))
        else: # Try to get more shapes
            pass
        
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

    # red_spider
    trace_img = drawed_img.copy()
    trace_img[drawed_img == 0] = 1
    trace_img[drawed_img == 255] = 0

    trace_img = red_spider(trace_img)
    save_trace_image(trace_img)

    if (trace_img == 0).sum() == 0:
        results['Exist Open Shapes'] = False
        results['Open Shapes Count'] = 0
    else:
        results['Exist Open Shapes'] = True
        results['Open Shapes Count'] = 1
    
    return results
    