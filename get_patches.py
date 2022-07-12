from patch_extractor import find_centers_by_heuristic
import os.path as osp
import os
import numpy as np
import sys

def patch_to_tensors(patches):
    ls=[[p.left_col,+p.top_row ]for p in patches]
    return ls

def padTo(mat, dim, val=0, is_np=False):
    s = mat.shape
    padding = [[0, m - s[i]] for (i, m) in enumerate(dim)]
    # print(padding)
    # print(mat.shape)
    if is_np:
        mat = np.pad(mat, padding, constant_values=val)
    else:
        mat = tf.pad(mat, padding, 'CONSTANT', constant_values=val)
    return mat

def extract_patches(im, patches, patch_size=30):
    out =[]
    for p in patches:
        extracted=im[p[0]:p[0]+patch_size,p[1]:p[1]+patch_size]
        out.append(padTo(extracted,(patch_size,patch_size),is_np=True))
    return np.stack(out)

def check_patches(ls,distogram,patch_size):
    for p in ls:
        if p[0] < 0 or p[1] < 0:
            raise IndexError
        if p[0] > distogram.shape[0] or p[1] > distogram.shape[1]:
            raise IndexError

def add_to_batch(geo_patches,cords,geo_batch,cords_batch):
    geo_batch.append(geo_patches)
    cords_batch.append(cords)

def get_distogram(distogram_file, size_r, size_l):
    mat = np.load(distogram_file, allow_pickle=True)
    shape = mat.shape
    if shape[0] > size_r or shape[1] > size_l:
        print("XXXX", path)
        print("size_r ",size_r," size_l ",size_l)
        print("size over th:", shape)
        raise OverflowError
    return mat

# save together
def preprosses_patches_batched(distogram_file,centers=6, patch_size=30,batch_size=50,size_r=750,size_l=250):
    batch_num=1
    geo_batch = []
    cords_batch = []

    file = open(distogram_file, 'r')
    lines = file.readlines()
    print(len(lines))

    for line in lines:
        distogram_file = line.strip()
        print(distogram_file)
        #distogram = get_distogram(distogram_file, size_r, size_l)
        try:
            distogram = get_distogram(distogram_file, size_r, size_l)
            if not (distogram > 0).any():
                print("all distogram are zero")
                continue
            # raise ZeroDivisionError
        except ValueError :
            with open("errors.txt", 'a') as f:
                f.write("crashed on line " + line + " at file "+ distogram_file + " \n")
            continue

        patches = find_centers_by_heuristic(distogram,centers=centers,patch_size=patch_size)
        #extract patches
        ls = [[p.top_row, p.left_col] for p in patches]
        check_patches(ls,distogram,patch_size)
        geo_patches = extract_patches(distogram, ls,patch_size=patch_size)
        add_to_batch(geo_patches,np.stack(ls),geo_batch,cords_batch)
        if len(geo_batch)==batch_size:
            # if not osp.exists(osp.join(out_dir,"batch_geo_"+str(batch_num))):
            np.savez_compressed("batch_geo_"+str(batch_num), np.stack(geo_batch), allow_pickle=True)
            np.savez_compressed("batch_cords_"+str(batch_num), np.stack(cords_batch), allow_pickle=True)
            print("saved batch")
            # else:
            #     print("skipping")
            batch_num+=1
            geo_batch=[]
            cords_batch=[]

    if len(geo_batch) > 0:
        np.savez_compressed("batch_geo_" + str(batch_num), np.stack(geo_batch), allow_pickle=True)
        np.savez_compressed("batch_cords_" + str(batch_num), np.stack(cords_batch), allow_pickle=True)
        print("saved batch")


if __name__ == '__main__':
    try:
        distogram_filenames_file = sys.argv[1]
        preprosses_patches_batched(distogram_filenames_file,size_r=750,size_l=250,batch_size=2500,centers=8,patch_size=20)
    except Exception as e :
        print(e)
        with open("errors.txt",'a')as f:
            f.write("crash on "+distogram_filenames_file+" \n")
