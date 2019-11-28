#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_type = 'test2'

"""
参考:
[1]『画像処理とパターン認識入門』酒井幸市 著
[2] scipy.fftpack.dct http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
"""

class DCT:
        def __init__(self,N):
                self.N = N     # データ数．
                # 1次元，2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
                self.phi_1d = np.array([ self.phi(i) for i in range(self.N) ])

                # Nが大きいとメモリリークを起こすので注意
                # MNISTの28x28程度なら問題ない
                self.phi_2d = np.zeros((N,N,N,N))
                for i in range(N):
                        for j in range(N):
                                phi_i,phi_j = np.meshgrid(self.phi_1d[i],self.phi_1d[j])
                                self.phi_2d[i, j] = phi_i*phi_j

        def dct(self,data):
                """ 1次元離散コサイン変換を行う """
                return self.phi_1d.dot(data)

        def idct(self,c):
                """ 1次元離散コサイン逆変換を行う """
                return np.sum( self.phi_1d.T * c ,axis=1)

        def dct2(self,data):
                """ 2次元離散コサイン変換を行う """
                return np.sum(self.phi_2d.reshape(N*N,N*N)*data.reshape(N*N),axis=1).reshape(N,N)

        def idct2(self,c):
                """ 2次元離散コサイン逆変換を行う """
                return np.sum((c.reshape(N,N,1)*self.phi_2d.reshape(N,N,N*N)).reshape(N*N,N*N),axis=0).reshape(N,N)

        def phi(self,k):
                """ 離散コサイン変換(DCT)の基底関数 """
                # DCT-II
                if k == 0:
                        return np.ones(self.N)/np.sqrt(self.N)
                else:
                        return np.sqrt(2.0/self.N)*np.cos((k*np.pi/(2*self.N))*(np.arange(self.N)*2+1))
                # DCT-IV(試しに実装してみた)
                #return np.sqrt(2.0/N)*np.cos((np.pi*(k+0.5)/self.N)*(np.arange(self.N)+0.5))

if __name__=="__main__":
        N = 8 #基底は8x8
        dct = DCT(N)  # 離散コサイン変換を行うクラスを作成

        img = cv2.imread('image/' + image_type + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = img[0:(img.shape[0] / 8) * 8, 0:(img.shape[1] / 8 * 8)] #8の倍数にしておく

        num_h = img.shape[0] / 8
        num_w = img.shape[1] / 8

        dct_coeffs = []
        dct_coeff_img = np.zeros(img.shape)

        for i in range(num_h):
                for j in range(num_w):
                        img_block = img[i*8 : (i+1)*8, j*8 : (j+1)*8]
                        dct_coeff = dct.dct2(img_block)
                        dct_coeffs.append(dct_coeff)
                        dct_coeff_img[i*8 : (i+1)*8, j*8 : (j+1)*8] = dct_coeff

        dct_coeff_img /= np.max(dct_coeff_img)

        #圧縮
        decomp_img = np.zeros(img.shape).astype(np.uint8)

        for i in range(num_h):
                for j in range(num_w):
                        dct_coeff = dct_coeffs[i*num_w+j]
                        idct_img = dct.idct2(dct_coeff)
                        idct_img = np.clip(idct_img, 0, 255)
                        decomp_img[i*8 : (i+1)*8, j*8 : (j+1)*8] = idct_img


        merged_img = np.zeros((img.shape[0], img.shape[1]*2)).astype(np.uint8)
        merged_img[:, 0:img.shape[1]] = img
        merged_img[:, img.shape[1]:img.shape[1]*2] = decomp_img
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", merged_img)
        cv2.waitKey(0)

        #calc variation
        dct_coeffs = np.stack(dct_coeffs)
        dct_coeffs_var = np.var(dct_coeffs, axis=0)

        print(dct_coeffs_var)
