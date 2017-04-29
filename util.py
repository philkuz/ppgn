import numpy as np
import scipy.misc
import subprocess
from scipy.linalg.blas import sgemm

def normalize(img, out_range=(0.,1.), in_range=None):
    if not in_range:
        min_val = np.min(img)
        max_val = np.max(img)
    else:
        min_val = in_range[0]
        max_val = in_range[1]

    result = np.copy(img)
    result[result > max_val] = max_val
    result[result < min_val] = min_val
    result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
    return result

def deprocess(images, out_range=(0.,1.), in_range=None):
    num = images.shape[0]
    c = images.shape[1]
    ih = images.shape[2]
    iw = images.shape[3]

    result = np.zeros((ih, iw, 3))

    # Normalize before saving
    result[:] = images[0].copy().transpose((1,2,0))
    result = normalize(result, out_range, in_range)
    return result

def get_image_size(data_shape):
    '''
    Return (227, 227) from (1, 3, 227, 227) tensor.
    '''
    if len(data_shape) == 4:
        return data_shape[2:]
    else:
        raise Exception("Data shape invalid.")

def save_image(img, name):
    '''
    Normalize and save the image.
    '''
    img = img[:,::-1, :, :] # Convert from BGR to RGB
    output_img = deprocess(img, in_range=(-120,120))
    scipy.misc.imsave(name, output_img)

def write_label_to_img(filename, label):
    # Add a label below each image via ImageMagick
    subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
    subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%s\" -border 0x0 -append %s" %
         (filename, 30, label, filename)], shell=True)


def convert_words_into_numbers(vocab_file, words):
    # Load vocabularty
    f = open(vocab_file, 'r')
    lines = f.read().splitlines()

    numbers = [ lines.index(w) + 1 for w in words ]
    numbers.append( 0 )     # <unk>
    return numbers

def gram(matrix, gram_scale=1):
    ''' calculates a grammian matrix'''
    matrix = matrix.reshape((matrix.shape[0], matrix.shape[1] * matrix.shape[2]))
    transpose = matrix.transpose()
    return gram_scale* matrix.dot( transpose)

def save_checkerboard(images, path):
    all_rows = []
    # left padding
    all_rows.append(np.zeros((227, 227*8,3)))
    for col in images:
        print(all_rows[-1].shape)
        new_col = []
        for i in col:
            i = i[::-1,:,:]
            new_col += [deprocess(i, in_range=(-120, 120))]
        all_rows.append(np.concatenate(new_col, axis=1))
        print(all_rows[-1].shape)

    out_image = np.concatenate(all_rows, axis=0)
    # for r in range((len(images) / 8)+1):
    #     row = []
    #     for i in images[r:(r+1)*8]:
    #        row += [i.reshape((3,227,227))]
    #     all_rows += [np.concatenate(row, axis=2)]
    #     print 'allrows,',all_rows[r].shape
    # out_image = np.concatenate(all_rows, axis=1).transpose((1, 2, 0))
    print(out_image.shape)
    scipy.misc.imsave(path, out_image)

