import cv2
import numpy as np

img_type = 'test2'

if __name__ == '__main__':
    orig_img = cv2.imread('image/' + img_type + '.jpg', cv2.IMREAD_GRAYSCALE)

    psnrs = []
    for i in range(20):
        img = cv2.imread('image/' + img_type + '/' + img_type + '-' + str((i+1)*5) + '.jpg', cv2.IMREAD_GRAYSCALE)

        #calc PSNR
        img_diff = orig_img - img
        img_diff_square = img_diff * img_diff
        mse = np.mean(img_diff_square)

        psnr = 10 * np.log10(255**2 / mse)
        print(psnr)

