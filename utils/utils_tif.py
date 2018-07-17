import cv2
import numpy as np

def LBV(D1,D2,D3,D4):
    B = 2.220521*D1 + 1.148545*D2 - 0.382848*D3 - 2.986217*D4
    V = 0.630762*D1 - 2.350279*D2 + 0.883035*D3 + 0.836482*D4
    L = 0.630762*D1 - 2.350279*D2 + 0.883035*D3 + 0.836482*D4
    return L,B,V


def min_max_scaler(arr, feature_range=(0, 1)):
    return (arr - feature_range[0]) / (feature_range[1] - feature_range[0])


def stretch_8bit(bands, lower_percent=5, higher_percent=95):
    print(bands.shape)
    out = np.zeros_like(bands).astype(np.uint8)
    # for i in range(3):
    a = 0
    b = 255
    c = np.percentile(bands, lower_percent)
    d = np.percentile(bands, higher_percent)
    t = a + (bands - c) * (b - a) / (d - c)
    t[t<a] = a
    t[t>b] = b
    out[:,:] =t
    return out.astype(np.uint8)


def percentile_scaler(im):
    """
    按照百分比
    :return:
    """
    maxv = np.percentile(im, 99)
    # maxv = np.max(im)
    # minv = np.min(im)
    minv = np.percentile(im, 1)
    im = (im - minv)/(maxv-minv)

    meanv = im.mean()
    print(meanv)
    im += (0.5-meanv)
    im *= 255
    # np.clip(im, 0, 255)
    return im.astype(np.uint8)
