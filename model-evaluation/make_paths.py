import os
from sys import argv


def make_path(folder_path, file_prefix, output_path):
    file_list = os.listdir(output_path)
    train_file = open(os.path.join(output_path, file_prefix + '_train.txt'), 'w')
    valid_file = open(os.path.join(output_path, file_prefix + '_valid.txt'), 'w')
    all_label_files = []
    good_image_files = []
    for file1 in file_list:
        if '.txt' in file1:
            all_label_files.append(file1)
            image_file_jpg = file1.split('.txt')[0] + '.JPG'
            if image_file_jpg in file_list:
                good_image_files.append(image_file_jpg)
            else:
                print('Could not find image file for : {}'.format(file1))

    for idx, img in enumerate(good_image_files):
        img_path = folder_path + img

        if idx % 10 == 0 or idx % 10 == 5:
            valid_file.write(img_path + '\n')
            print('Successfully wrote image {} to validation dataset'.format(img_path))
        else:
            train_file.write(img_path + '\n')
            print('Successfully wrote image {} to training dataset'.format(img_path))


    train_file.close()
    print('Training data saved')
    valid_file.close()
    print('Validation data saved')


def main():
    '''
    Example input:
    python3 make_paths.py /Users/shane/Desktop/labeled_car_data moms_car
    '''
    make_path(argv[1], argv[2], argv[3])


if __name__ == '__main__':
    main()