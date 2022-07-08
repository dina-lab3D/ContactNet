import numpy as np
import scipy
import math
from scipy import signal
from skimage import filters
from matplotlib import patches
import sys
import os.path as osp
import os

class patch:
    def __init__(self,row,col,patch_size,im):
        self.row=row
        self.col=col
        self.patch_size=patch_size
        top_row, bottom_row, right_col, left_col, w, h = get_cords(im, row, col, kernal_size=patch_size)
        assert top_row < bottom_row
        # print(col," ",left_col," ",right_col)
        assert left_col < right_col
        self.top_row=top_row
        self.left_col=left_col
        self.bottom_row=bottom_row
        self.right_col=right_col
        self.w=w
        self.h=h

    def intersection(self,row,col,im):
        top_row, bottom_row, right_col, left_col, w, h = get_cords(im, row, col, kernal_size=self.patch_size)
        x_left = max(self.left_col, left_col)
        y_top = max(self.top_row, top_row)
        x_right = min(self.right_col, right_col)
        y_bottom = min(self.bottom_row, bottom_row)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area +self.patch_size-w+self.patch_size-h

    def remove_patch(self,im,padding=0):
        im[max(self.top_row-padding,0):min(self.bottom_row+padding,im.shape[0]),max(self.left_col-padding,0):min(self.right_col+padding,im.shape[1])]=0
        return im


    # def plot_rectangle(self,ax):
    #     plt.plot(self.col, self.row, 'r.')
    #     rec = patches.Rectangle((self.left_col, self.top_row), self.patch_size, self.patch_size, linewidth=1,
    #                             edgecolor='r', facecolor='none')
    #     ax.add_patch(rec)

    def extract_patch(self,distogram):
       return distogram[max(self.top_row,0):min(self.bottom_row,distogram.shape[0]),max(self.left_col,0):min(self.right_col,distogram.shape[1])]

def cat_range(t,range):
    return t[range[0]:range[1],:]

def k_largest_index_argpartition(array, k):
    # x_shape=tf.shape(x)
    # op_values, top_indices = tf.nn.top_k(tf.keras.layers.Flatten()(x), k)
    # top_indices = tf.stack(((top_indices // x_shape[1]), (top_indices % x_shape[1])), -1)
    # return top_indices
    idx = np.argsort(-array.ravel())[:k]
    cords=np.column_stack(np.unravel_index(idx, array.shape))

    return cords

def culc_bound_overflow(row,col,im,patch_size=30):
    return max(0,row-im.shape[0]+patch_size/2)+max(0,col-im.shape[1]+patch_size/2)

def get_intersection(row,col,im,centers,patch_size=30):
    # row,col=cords
    intersection=0
    for c in centers:
        intersection+=c.intersection(row,col,im)
    bound_overflow=culc_bound_overflow(row,col,im,patch_size=patch_size)
    return intersection +bound_overflow

def get_max_activation(distogram, centers,patch_size=30):
    activated = scipy.signal.convolve(distogram, np.ones((patch_size//2, patch_size//2)),mode="same")
    highest_activation = k_largest_index_argpartition(activated, 200)
    if np.any(highest_activation[:,0]>distogram.shape[0]) or np.any(highest_activation[:,1]>distogram.shape[1]):
        print("shit!")
    intersecionsd = np.apply_along_axis(lambda x: get_intersection(x[0],x[1], distogram, centers,patch_size=patch_size), 1,
                                        highest_activation)
    least_intrsection = np.argmin(intersecionsd)
    row, col = highest_activation[least_intrsection]
    return row, col

def get_cords(array,row,col,kernal_size=30):
    offset_up=abs(min(0,row-(kernal_size / 2)))
    offset_down = abs(min(0, array.shape[0]-(row + (kernal_size / 2))))
    offset_left = abs(min(0, col - (kernal_size / 2)))
    offset_right = abs(min(0,array.shape[1]-( col + (kernal_size / 2))))

    top_row = int(max(row - (kernal_size / 2)-offset_down, 0))
    bottom_row = int(min(row + (kernal_size / 2)+offset_up, array.shape[0]))
    left_col = int(max(col - (kernal_size / 2)-offset_right,0))
    right_col = int(min((col + kernal_size / 2)+offset_left, array.shape[1]))
    h = bottom_row - top_row
    w =  right_col-left_col
    return top_row,bottom_row,right_col,left_col,w,h

# given a distogram extract centers with window size of patch_size
def find_centers_by_heuristic(im, denosied=False,centers=8, patch_size=30):
    if denosied:
        im_denosed = filters.median(im, selem=np.ones((3, 3)))
        print(im_denosed.shape)
    else:
        # im_denosed=tf.identity(im)
        im_denosed = np.copy(im)
    patches=[]
    # ax=plt.gca()
    for c in range(centers):
        # activated=scipy.signal.convolve(im,np.ones(20,20))
        row,col = get_max_activation(im_denosed,patches,patch_size=patch_size)#### apply activation
        p= patch(row,col,patch_size,im_denosed)
        # p.plot_rectangle(ax)
        im_denosed=p.remove_patch(im_denosed,padding=0)
        patches.append(p)
        # if np.sum(im_denosed)<2:
        #     break
    return patches

def patch_to_tensors(patches):
    ls=[get_cords_tensor(p.top_row,+p.left_col) for p in patches]
    return ls

def slice_patch(im,cords):
    return im[max(cords[2], 0):min(cords[3], im.shape[0]), max(cords[0], 0):min(cords[1], im.shape[1])]
