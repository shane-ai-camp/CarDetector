import os
import cv2

from ai import yolo_forward, get_yolo_net


def evaluate():

    labels = ["Mom's Car"]
    positive_img_dir = os.path.join(os.getcwd(), 'labeled_car_data')
    negative_img_dir = os.path.join(os.getcwd(), 'unlabeled_car_data')
    confidence_level = 0.5
    threshold = 0.3
    img_list = []
    total_counter = 0
    counter = 0

    y_actu = []
    y_pred = []

    net_clement = get_yolo_net('yolov3.cfg', 'yolov3_final.weights')

    for img in os.listdir(positive_img_dir):
        if img.endswith('.JPG'):
            image = cv2.imread(os.path.join(positive_img_dir, img))
            img_detect = yolo_forward(net_clement, labels, image, confidence_level, threshold)
            total_counter += 1
            if img_detect == ([], [], [], []):
                y_actu.append(1)
                y_pred.append(0)
                continue
            else:
                img_list.append(img_detect)
                print(img_detect)
                counter += 1
                y_actu.append(1)
                y_pred.append(1)
        else:
            continue

    for img in os.listdir(negative_img_dir):
        if img.endswith('.JPG'):
            image = cv2.imread(os.path.join(negative_img_dir, img))
            img_detect = yolo_forward(net_clement, labels, image, confidence_level, threshold)
            total_counter += 1
            if img_detect == ([], [], [], []):
                y_actu.append(0)
                y_pred.append(0)
                continue
            else:
                img_list.append(img_detect)
                print(img_detect)
                counter += 1
                y_actu.append(0)
                y_pred.append(1)
        else:
            continue

    print('\n{} images detected out of {} total'.format(counter, total_counter))

    moms_car_counter = 0

    for i in img_list:
        moms_car_counter += len(i[0])

    print("Mom's car detected {} times".format(moms_car_counter))

    return y_actu, y_pred


if __name__ == '__main__':
    main()