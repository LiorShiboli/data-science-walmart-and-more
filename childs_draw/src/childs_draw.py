# Example:
# import src.childs_draw as childs_draw
# res = childs_draw.child_draw_describe(7557, 12, True)
# res

# Imports
import os
import sys
import shutil

import time
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os
import sys
import shutil

import time
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

import src.childs_draw as childs_draw

# Flags
DELTA_XY = 40
DIRECT_COUNT_RATIO = 10
MAX_DIFF = 10

# Configs
data_root_folder = './data'
save_images_folder = '.'

image_show_number_size = 3

g_save_images_enabled = True
save_image_counter = 0

def set_data_root_folder(folder):
    global data_root_folder
    data_root_folder = folder
    
def set_save_images_folder(folder):
    global save_images_folder
    save_images_folder = folder


# Loaders
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

def load_draw(folder_id: int, image_id: int):
    (img, df) = load_image_data(folder_id, image_id)
    
    if img is None or df is None:
        return None
    
    df.columns = map(lambda x: x.replace(" ", ""), df.columns.tolist())
    
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
    
    return (img, df)


# Lines Info
def get_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

def get_next_block(sub_df: pd.DataFrame, current, distance: float):
    if current not in sub_df.index.tolist():
        return None
            
    x1, y1 = sub_df.iloc[current][['X', 'Y']]
    
    for i in range(current+1, len(sub_df)):
        x2, y2 = sub_df.iloc[i][['X', 'Y']]
        
        distance -= get_distance(x1, y1, x2, y2)
        
        x1, y1 = x2, y2
        
        if distance <= 0:
            return i
    return len(sub_df) - 1

def get_lines_in_sub_draw(sub_df: pd.DataFrame):
    lines_cons = [0]
    last_con_index = 0
    
    cons_x, cons_y = [], []
    con_direct_lines_x, con_direct_lines_y = [], []
    
    # Step 1
    start_index = 0

    start_index = 0
    ls = sub_df.index.tolist()
        
    # Step 2
    while start_index < len(ls):
        end_index = get_next_block(sub_df, start_index, DELTA_XY)
                
        if start_index < end_index:
            x1, y1 = sub_df.iloc[ls[start_index]][['X', 'Y']]
            x2, y2 = sub_df.iloc[ls[end_index]][['X', 'Y']]
            
            points = sub_df.iloc[ls[start_index:end_index]][['X', 'Y']]
            
            direct_line_size = DIRECT_COUNT_RATIO * len(points.index)
            direct_line_x = np.linspace(x1, x2, num=direct_line_size).tolist()
            direct_line_y = np.linspace(y1, y2, num=direct_line_size).tolist()
            
            max_diff = 0
            con_x, con_y = x1, y1
            for point_i in points.index:
                x,y = points.loc[point_i][['X', 'Y']]
                
                min_diff_point = get_distance(x, y, x1, y1)
                
                for di in range(direct_line_size):
                    
                    dis = get_distance(x, y, direct_line_x[di], direct_line_y[di])
                    if dis > 0:
                        if min_diff_point == 0:
                            min_diff_point = dis
                        else:
                            min_diff_point = min([min_diff_point, dis])
                
                max_diff = max([max_diff, min_diff_point])
                if max_diff == min_diff_point:
                    con_x, con_y = x, y
                    con_index = point_i
                    
                
            if max_diff > MAX_DIFF:
                start_con_index = last_con_index
                last_con_index = con_index
                
                start_index = last_con_index
                
                lines_cons += [last_con_index]
            else:
                start_index += 1
        else:
            start_index += 1
            
    lines_cons += [len(sub_df.index) - 1]

    lines = []
    
    for i in range(len(lines_cons) - 1):
        start_index, end_index = lines_cons[i], lines_cons[i + 1]
        
        line = sub_df.loc[start_index: end_index]
        
        distance = 0
        
        for j in range(start_index, end_index):
            p1 = sub_df.loc[j]
            p2 = sub_df.loc[j + 1]
            distance += get_distance(p1['X'], p1['Y'], p2['X'], p2['Y'])
    
        lines += [{
            'Distance': distance,
            'Mean Pressure': line['Pressure'].mean(),
            'X': line['X'],
            'Y': line['Y']
        }]
        
    return lines

def get_lines_info(df: pd.DataFrame):
    hands_up = []
    lines = []
    
    disDiff = ((df.XDiff * df.XDiff) + (df.YDiff * df.YDiff)) ** 0.5
    
    limits = [df.head(1).index[0], df.tail(1).index[0]] + df[df.TimeDiff > 5 * df.TimeDiff.std()].index.tolist()
    limits = np.unique(list(sorted(limits))).tolist()
    
    index = 0

    for i in range(len(limits) - 1):
        start = limits[i]
        end = limits[i+1]

        sub_df = df.iloc[start:end].copy()
        sub_df.index = range(len(sub_df.index))

        sub_df = sub_df.iloc[1:]
        sub_df.index = range(len(sub_df.index))

        hands_up += [(sub_df['X'], sub_df['Y'])]

        lines += get_lines_in_sub_draw(sub_df)

    lines = pd.DataFrame(lines)
    
    res = {
        'Hands Up Count': len(hands_up),
        'Lines Count': len(lines),
        'Hands Up Info': hands_up,
        'Lines Info': lines
    }
    
    return res

# Images Saver
def save_image(img):
    global g_save_images_enabled
    if not g_save_images_enabled:
        return True
    
    global save_image_counter
    global image_show_number_size
    global save_images_folder
    
    save_image_counter += 1
    number_as_str = str(save_image_counter)
    while len(number_as_str) < image_show_number_size:
        number_as_str = '0' + number_as_str
    file_name = save_images_folder + '/show_' + number_as_str + '.png'
    
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

# collect 'Exist Closed Shapes' and 'Closed Shapes Count'
def get_shapes_count(img, df: pd.DataFrame):
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
        
    drawed_img_backup = drawed_img.copy()
    
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
            (x,y) = points[0]
            points.remove(points[0])
            if img[x,y] == 0:                
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
        return (0, drawed_img)
    
    closed_shapes_count = 0
    while (trace_img == 0).sum() > 0:
        closed_shapes_count += 1
            
        found = np.where(trace_img == 0)
        points = list()
        points.append((found[0][0], found[1][0]))
        trace_img = red_spider(trace_img, points)
        save_trace_image(trace_img)
    
    return (closed_shapes_count, drawed_img_backup)

def child_draw_describe(folder_id: int, image_id: int, save_images_enabled: bool = False):
    global g_save_images_enabled
    g_save_images_enabled = save_images_enabled
   
    results = {
        'Hands Up Count': None,
        'Lines Count': None,
        'Exist Closed Shapes': None,
        'Closed Shapes Count': None,
        'Hands Up Info': None, # for ChildDrawDescribe
        'Lines Info': None # for ChildDrawDescribe
    }

    img, df = load_draw(folder_id, image_id)

    lines_info = get_lines_info(df)
    (closed_shapes_count, drawed_img) = get_shapes_count(img, df)

    results['Hands Up Count'] = lines_info['Hands Up Count']
    results['Lines Count'] = lines_info['Lines Count']
    results['Hands Up Info'] = lines_info['Hands Up Info']
    results['Lines Info'] = lines_info['Lines Info']

    results['Exist Closed Shapes'] = closed_shapes_count != 0
    results['Closed Shapes Count'] = closed_shapes_count
    results['Drawed Image'] = drawed_img

    return results

class ChildDrawDescribe:
    def __init__(self, folder_id, image_id, save_images_enabled: bool = False):
        # Get Data
        self.results = child_draw_describe(folder_id, image_id, save_images_enabled)
    
    def get_results(self):
        return {
            'Hands Up Count': self.results['Hands Up Count'],
            'Lines Count': self.results['Lines Count'],
            'Exist Closed Shapes': self.results['Exist Closed Shapes'],
            'Closed Shapes Count': self.results['Closed Shapes Count'],
            'Lines Info': self.results['Lines Info'][['Distance', 'Mean Pressure']]
        }        
    
    def print_image(self):
        drawed_img = self.results['Drawed Image']
        plt.imshow(drawed_img, cmap='gray')
        plt.show()
    
    def print_lines_plots(self):
        hands_up = self.results['Hands Up Info']
        lines = self.results['Lines Info']
        
        fig = plt.figure(figsize=(14, 7))
        
        ax = fig.add_subplot(1,2,1)
        ax.set_title('Hands Up')
        for points in hands_up:
            plt.scatter(points[0], points[1])

        ax = fig.add_subplot(1,2,2)
        ax.set_title('Lines')
        for line_index in lines.index:
            plt.scatter(lines.loc[line_index]['X'], lines.loc[line_index]['Y'].tolist())
        
        plt.show()