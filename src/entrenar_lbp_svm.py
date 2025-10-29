from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import os
import cv2
import numpy as np
from preprocesamiento import load_and_preprocess
from skimage.feature import local_binary_pattern

# Asegurarse de que el directorio de modelos exista
os.makedirs('modelos', exist_ok=True)
TARGET_SIZE = (128, 128)

def extraer_lbp_mejorado_entrenamiento(X_images: np.ndarray) -> np.ndarray:
    """Extrae LBP mejorado con características estadísticas y de gradiente."""
    all_features = []
    
    print("  Aplicando LBP Mejorado (Uniforme + Estadísticas + Gradiente)...")
    for img in X_images:
        caracteristicas = []
        
        # 1. Histogramas LBP de múltiples radios
        configuraciones = [
            {'P': 8, 'R': 1, 'method': 'uniform'},
            {'P': 16, 'R': 2, 'method': 'uniform'}, 
            {'P': 24, 'R': 3, 'method': 'uniform'}
        ]
        
        for config in configuraciones:
            lbp = local_binary_pattern(img, P=config['P'], R=config['R'], method=config['method'])
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)
            caracteristicas.extend(hist)
        
        # 2. Características estadísticas
        caracteristicas.extend([
            np.mean(img), np.std(img), np.var(img), np.min(img), np.max(img)
        ])
        
        # 3. Características de gradiente (Sobel)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        caracteristicas.extend([np.mean(magnitud), np.std(magnitud)])
        
        # 4. Características de Contraste
        kernel = np.ones((3,3), np.float32) / 9
        imagen_suavizada = cv2.filter2D(img, -1, kernel)
        contraste = np.abs(img.astype(float) - imagen_suavizada.astype(float))
        caracteristicas.extend([np.mean(contraste), np.std(contraste)])
        
        all_features.append(np.array(caracteristicas))
        
    return np.array(all_features)


print("Cargando dataset...")
# Nota: load_and_preprocess retorna imágenes en escala de grises
X, y, classes = load_and_preprocess('data/imagenes')

print("Extrayendo características LBP Mejorado...")
X_lbp = extraer_lbp_mejorado_entrenamiento(X)

# 1. Escalar características (muy importante para SVMs)
print("Escalando características LBP...")
# Usamos un scaler específico para LBP para evitar conflictos con HOG
scaler_lbp = StandardScaler()
X_lbp_scaled = scaler_lbp.fit_transform(X_lbp)

# Guardar el scaler para su uso en la GUI
with open('modelos/scaler_lbp.pkl', 'wb') as f:
    pickle.dump(scaler_lbp, f)
print(" Scaler de LBP guardado en modelos/scaler_lbp.pkl")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_lbp_scaled, y, test_size=0.2, random_state=42)

print("Entrenando SVM (LBP Mejorado)...")
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluación
y_pred = svm.predict(X_test)

print("\n\n🔹 Resultados del modelo SVM (LBP Mejorado):")
print(classification_report(y_test, y_pred, target_names=classes))

# Guardar el modelo
dump(svm, 'modelos/svm_lbp.pkl')
print("\n✅ Modelo SVM (LBP Mejorado) guardado en modelos/svm_lbp.pkl")
