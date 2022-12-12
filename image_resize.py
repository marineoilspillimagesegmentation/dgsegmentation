import os
import cv2


def resize_img(ORIGINAL, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    path = os.path.join(ORIGINAL, data_k)
    img_list = os.listdir(path)

    for i in img_list:
        if i.endswith('.png'):
            img_array = cv2.imread((path + '/' + i), cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
            img_name = str(i)
            save_path = path + '_new/'
            if os.path.exists(save_path):
                print(i)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)


if __name__ == '__main__':
    DATADIR = "/scratch/oilspill/fc160/myproject/marine_oilspill_segmentation/DGNet_MarineOilSeg/"
    data_k = 'ORIGINAL'
    img_size = [128, 128]
    resize_img(DATADIR, data_k, img_size)
    #resize_img(DATADIR, data_k, img_size)
