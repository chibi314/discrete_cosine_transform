from PIL import Image
import numpy as np
import os.path

img_type = 'test2'

if __name__ == '__main__':

    orig_img = Image.open('./image/' + img_type + '.jpg')
    orig_img = orig_img.convert('L')
    orig_img = np.asarray(orig_img)

    psnrs = []
    for i in range(400):
        img = Image.open('./image/' + img_type + '/' + img_type + '-' + '{:.3f}'.format((i+1)/1000.0) + '.jpc')
        img = img.convert('L')
        img = np.asarray(img)

        img_diff = orig_img - img
        img_diff_square = img_diff * img_diff
        mse = np.mean(img_diff_square)

        psnr = 10 * np.log10(255**2 / mse)

        filesize = os.path.getsize('./image/' + img_type + '/' + img_type + '-' + '{:.3f}'.format((i+1)/1000.0) + '.jpc')
        bpp = filesize / (img.shape[0] * img.shape[1])
        #print(str(bpp) + ' ' + str(psnr))
        print(psnr)
