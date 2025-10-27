from skimage.feature import hog, local_binary_pattern
from skimage import color
import numpy as np


def extract_hog(images):
    features = []
    for img in images:
        feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2))
        features.append(feat)
    return np.array(features)


def extract_lbp(images):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59))
        features.append(hist / np.sum(hist))
    return np.array(features)


def extraer_hog(imagen, orientaciones=9, pixeles_por_celda=(8, 8), celdas_por_bloque=(2, 2)):
    
    if len(imagen.shape) == 3:
        imagen_gris = color.rgb2gray(imagen)
    else:
        imagen_gris = imagen
    
    caracteristicas = hog(
        imagen_gris,
        orientations=orientaciones,
        pixels_per_cell=pixeles_por_celda,
        cells_per_block=celdas_por_bloque,
        visualize=False,
        feature_vector=True
    )
    
    return caracteristicas


def extraer_lbp(imagen, radio=3, n_puntos=24, metodo='uniform'):
    if len(imagen.shape) == 3:
        imagen_gris = color.rgb2gray(imagen)
    else:
        imagen_gris = imagen
    
    lbp = local_binary_pattern(imagen_gris, n_puntos, radio, method=metodo)
    
    n_bins = n_puntos + 2 if metodo == 'uniform' else 2**n_puntos
    histograma, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    histograma = histograma.astype(float)
    histograma /= (histograma.sum() + 1e-7)
    
    return histograma
