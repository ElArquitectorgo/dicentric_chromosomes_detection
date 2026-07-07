# Fast GLCM methods implemented by tzm030329
# https://github.com/tzm030329/GLCM/

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def lbp(img, radius=3, method='uniform'):
    n_points = 8 * radius # eight directions around the center pixel
    lbp = local_binary_pattern(img, n_points, radius, method)
    lbp_u8 = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lbp_u8

def overlay_labels(img, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=img, bg_label=0, alpha=0.5)

def lbp_edges(img, radius=3, method='uniform'):
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method)
    w = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    return cv2.normalize(overlay_labels(img, lbp, edge_labels), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def lbp_flat(img, radius=3, method='uniform'):
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method)
    w = radius - 1
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    return cv2.normalize(overlay_labels(img, lbp, flat_labels), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def lbp_corner(img, radius=3, method='uniform'):
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method)
    w = radius - 1
    i_14 = n_points // 4  # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
    corner_labels = list(range(i_14 - w, i_14 + w + 1)) + list(
        range(i_34 - w, i_34 + w + 1)
    )
    return cv2.normalize(overlay_labels(img, lbp, corner_labels), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def raw(img):
    return img

def he(img):
    res = cv2.equalizeHist(img)
    return res

def clahe(img, clip_limit=10.0, tile_size=5):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))
    res = clahe.apply(img)
    return res

def lt(img):
    img = img.astype(np.float32)
    c = 255 / np.log1p(img.max())
    res = c * np.log1p(img)
    return np.clip(res, 0, 255).astype(np.uint8)

def gc(img, gamma=0.5):
    table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
    res = cv2.LUT(img, table)
    return res

def otsu(img, morph=None, kernel_size=2):
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    if morph == "open":    
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    elif morph == "close":
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    return res

def watershed(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = cv2.erode(sure_fg, np.ones((5,5),np.uint8)) # remove very little surfaces
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    counts = np.bincount(markers.ravel())
    small_objects_labels = np.where(counts < 3400)[0]
    for label in small_objects_labels:
        if label > 1:  # ignore unknown and background
            markers[markers == label] = 1 # set as background
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(image,markers)
    image[markers > 1] = [255,255,255]
    image[markers == -1] = [255,255,255]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)