import json
import os
from sys import argv
import cv2
import numpy as np
import pandas as pd


def create_dataframe(image_folder_path, csv_filename):
    '''
    Creates a dataframe using the Labelbox data
    '''
    df = pd.read_csv(csv_filename)
    df = df[['External ID', 'Label', 'Labeled Data', 'ID']]
    df = df[df['Label'] != 'Skip']
    df['local_file'] = df['ID'] + '-' + df['External ID']
    df['dimensions'] = df.apply(lambda row: get_dimensions(row['local_file'], image_folder_path), axis=1)
    df['width'] = df['dimensions'].apply(lambda dimensions: int(str(dimensions).split(',')[0][1:]))
    df['height'] = df['dimensions'].apply(lambda dimensions: int(str(dimensions).split(',')[1].strip()[:-1]))
    df['centers'] = df['Label'].apply(lambda label: get_box_centers_all_data(label))
    return df


def get_box_centers_all_data(row):
    lab = json.loads(row)
    centers = {}
    for key, val in lab.items():
        lab_df = pd.DataFrame(val)
        centers[key] = lab_df.iloc[:, 0].apply(lambda s: get_box_center(s)).tolist()
    return centers


def get_box_center(box_row):
    x_coords = []
    y_coords = []
    for item in box_row:
        x_coords.append(item['x'])
        y_coords.append(item['y'])
    x_center = (np.max(x_coords) - np.min(x_coords)) / 2 + np.min(x_coords)
    y_center = (np.max(y_coords) - np.min(y_coords)) / 2 + np.min(y_coords)
    box_width = np.max(x_coords) - np.min(x_coords)
    box_height = np.max(y_coords) - np.min(y_coords)
    return x_center, y_center, box_width, box_height


def get_dimensions(file_name, image_folder_path):
    file_path = os.path.join(image_folder_path, file_name)
    img = cv2.imread(file_path)
    if img is not None:
        img_height, img_width, channels = img.shape
        print('Successfully returned local dimensions for {}: ({}, {})'.format(file_name, img_width, img_height))
    else:
        img_height, img_width = 0, 0
        print('Failed to load image for {}'.format(file_name))
    return img_width, img_height


def get_yolo_formats(label, width, height):
    yolo_formats_to_write = []
    if width == 0 or height == 0:
        print('This file has a width or height that is 0. Skipping this file.')
        return yolo_formats_to_write
    for id, boxes in label.items():
        for box in boxes:
            if id == "Mom's Car":
                class_id = 0
            else:
                continue
            box_x_center, box_y_center, box_width, box_height = box[0], box[1], box[2], box[3]
            yolo_formats_to_write.append(' '.join([str(class_id),
                                                   str(box_x_center / width),
                                                   str(box_y_center / height),
                                                   str(box_width / width), str(box_height / height)]))

    return yolo_formats_to_write


def main():
    '''
    Example input:
    python3 format_data_for_yolo2.py /Users/shane/Desktop/labeled_car_data moms_car_data.csv
    '''
    script_name, image_folder_path, labelbox_csv = argv
    labelbox_data = create_dataframe(image_folder_path, labelbox_csv)
    for (img_filename, center, width, height) in zip(labelbox_data['local_file'],
                                                     labelbox_data['centers'],
                                                     labelbox_data['width'],
                                                     labelbox_data['height']):
        yolo_formats = get_yolo_formats(center, width, height)
        txt_filename = img_filename.split('.')[0] + '.txt'
        with open(os.path.join(image_folder_path, txt_filename), 'w') as f:
            for box in yolo_formats:
                f.write('%s\n' % box)
            print('Saved for file {}'.format(img_filename))


if __name__ == '__main__':
    main()
