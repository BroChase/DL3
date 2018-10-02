# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np


def get_im2col_indices(x_shape, field_h, field_w, padding, stride):
    # First figure out what the size of the output should be
    n, c, h, w = x_shape

    assert(h + 2 * padding - field_h) % stride == 0
    assert(w + 2 * padding - field_h) % stride == 0
    out_height = (h + 2 * padding - field_h)/stride + 1
    out_width = (w + 2 * padding - field_w)/stride + 1

    i0 = np.repeat(np.arange(field_h, dtype='int32'), field_w)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_w), field_h * c)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c, dtype='int32'), field_h * field_w).reshape(-1, 1)

    return k, i, j


def im2col(x, field_h, field_w, padding, stride):
    # zero-pad the input todo if statement for if there is padding ?
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_h, field_w, padding, stride)

    cols = x_padded[:, k, i, j]
    c = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_h * field_w * c, -1)
    return cols


def col2im(cols, x_shape, field_h, field_w, padding, stride):
    n, c, h, w = x_shape
    h_padded, w_padded = h + 2 * padding, w + 2 * padding
    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_h, field_w, padding, stride)
    cols_reshaped = cols.reshape(c * field_h * field_w, -1, n)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
