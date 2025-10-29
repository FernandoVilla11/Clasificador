import numpy as np
from skimage.feature import hog, local_binary_pattern
from typing import List, Tuple
import cv2
from skimage.feature import graycomatrix, graycoprops

# La dimensión objetivo debe coincidir con la usada en el entrenamiento y la GUI
TARGET_SIZE = (128, 128) 

# --- Extracción de HOG ---

def extract_hog(X_images: np.ndarray) -> np.ndarray:
    """
    Extrae características Histogram of Oriented Gradients (HOG) de un conjunto de imágenes.

    Args:
        X_images: Array de imágenes de entrada (numpy array de imágenes en escala de grises de 128x128).

    Returns:
        Array de características HOG aplanadas.
    """
    hog_features = []
    
    # Configuración de HOG (DEBE COINCIDIR CON LA EXTRACCIÓN EN LA GUI)
    hog_params = {
        'pixels_per_cell': (8, 8), 
        'cells_per_block': (2, 2), 
        'feature_vector': True,
        'block_norm': 'L2-Hys' # Normalización estándar para robustez
    }
    
    print(f"  Aplicando HOG con {hog_params['pixels_per_cell']} PPC y {hog_params['cells_per_block']} CPB...")

    for img in X_images:
        # La imagen ya debe estar en escala de grises y redimensionada
        features = hog(img, **hog_params)
        hog_features.append(features)
        
    return np.array(hog_features)

# --- Extracción de LBP Simple (para entrenamiento básico) ---

def extract_lbp(X_images: np.ndarray) -> np.ndarray:
    """
    Extrae características Local Binary Pattern (LBP) de un conjunto de imágenes.
    Retorna el histograma normalizado de LBP uniforme.
    """
    lbp_features = []
    
    P = 8  # Número de puntos vecinos
    R = 1  # Radio
    
    print(f"  Aplicando LBP Uniforme con P={P}, R={R}...")

    for img in X_images:
        # La imagen ya debe estar en escala de grises y redimensionada
        lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
        
        # Calcular el histograma del patrón binario local
        n_bins = int(lbp.max() + 1)
        
        # Generar histograma normalizado
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)
        lbp_features.append(hist)
        
    return np.array(lbp_features)
