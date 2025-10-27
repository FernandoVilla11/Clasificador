import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from joblib import dump, load
import argparse

def cargar_imagenes(directorio_datos, tamaño_img=128):
    imagenes, etiquetas, nombres_clases = [], [], []
    
    for item in os.listdir(directorio_datos):
        ruta_item = os.path.join(directorio_datos, item)
        if os.path.isdir(ruta_item):
            nombres_clases.append(item)
    
    nombres_clases.sort()
    print(f"Clases encontradas: {nombres_clases}")
    
    for idx, clase in enumerate(nombres_clases):
        ruta_clase = os.path.join(directorio_datos, clase)
        contador = 0
        
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                ruta_imagen = os.path.join(ruta_clase, archivo)
                img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (tamaño_img, tamaño_img))
                    imagenes.append(img)
                    etiquetas.append(idx)
                    contador += 1
        
        print(f"Cargadas {contador} imágenes de {clase}")
    
    return np.array(imagenes), np.array(etiquetas), nombres_clases

def extraer_hog(imagenes):
    caracteristicas = []
    print("Extrayendo características HOG...")
    
    for img in imagenes:
        feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2))
        caracteristicas.append(feat)
    
    return np.array(caracteristicas)


def extraer_lbp(imagenes):
    caracteristicas = []
    print("Extrayendo características LBP...")
    
    for img in imagenes:
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59))
        caracteristicas.append(hist / np.sum(hist))
    
    return np.array(caracteristicas)

def entrenar_svm_hog(X, y, nombres_clases):
    print("\n" + "="*50)
    print("ENTRENANDO SVM CON HOG")
    print("="*50)
    
    X_hog = extraer_hog(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Entrenando SVM...")
    modelo = SVC(kernel='linear', probability=True, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"Precisión: {precision:.2%}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    os.makedirs('modelos', exist_ok=True)
    dump(modelo, 'modelos/svm_hog.pkl')
    print("Modelo guardado en modelos/svm_hog.pkl")
    return modelo, precision


def entrenar_svm_lbp(X, y, nombres_clases):
    print("\n" + "="*50)
    print("ENTRENANDO SVM CON LBP")
    print("="*50)
    
    X_lbp = extraer_lbp(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_lbp, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Entrenando SVM...")
    modelo = SVC(kernel='rbf', probability=True, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión: {precision:.2%}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    os.makedirs('modelos', exist_ok=True)
    dump(modelo, 'modelos/svm_lbp.pkl')
    print("Modelo guardado en modelos/svm_lbp.pkl")
    return modelo, precision


def entrenar_red_neuronal(X, y, nombres_clases):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        print("TensorFlow no está disponible. Saltando entrenamiento de red neuronal.")
        return None, 0
    
    print("\n" + "="*50)
    print("ENTRENANDO RED NEURONAL CON HOG")
    print("="*50)
    
    X_hog = extraer_hog(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    print("Creando modelo de red neuronal...")
    modelo = Sequential([
        Dense(512, activation='relu', input_shape=(X_hog.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(len(nombres_clases), activation='softmax')
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Entrenando red neuronal...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    historia = modelo.fit(
        X_train_scaled, y_train_cat,
        validation_data=(X_test_scaled, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    loss, precision = modelo.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"Precisión: {precision:.2%}")
    
    os.makedirs('modelos', exist_ok=True)
    modelo.save('modelos/nn_hog.keras')
    
    with open('modelos/scaler_nn_hog.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Modelo guardado en modelos/nn_hog.keras")
    print("Scaler guardado en modelos/scaler_nn_hog.pkl")
    return modelo, precision

def evaluar_modelos():
    print("\n" + "="*60)
    print("EVALUACIÓN DE MODELOS ENTRENADOS")
    print("="*60)
    
    print("Cargando datos de prueba...")
    X, y, nombres_clases = cargar_imagenes('datos/imagenes')
    
    resultados = {}

    if os.path.exists('modelos/svm_hog.pkl'):
        try:
            modelo = load('modelos/svm_hog.pkl')
            X_hog = extraer_hog(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_hog, y, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = modelo.predict(X_test)
            precision = accuracy_score(y_test, y_pred)
            resultados['SVM-HOG'] = precision
            print(f"SVM-HOG: {precision:.2%}")
        except Exception as e:
            print(f"Error evaluando SVM-HOG: {e}")
    
    if os.path.exists('modelos/svm_lbp.pkl'):
        try:
            modelo = load('modelos/svm_lbp.pkl')
            X_lbp = extraer_lbp(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_lbp, y, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = modelo.predict(X_test)
            precision = accuracy_score(y_test, y_pred)
            resultados['SVM-LBP'] = precision
            print(f"SVM-LBP: {precision:.2%}")
        except Exception as e:
            print(f"Error evaluando SVM-LBP: {e}")

    if os.path.exists('modelos/nn_hog.keras'):
        try:
            from tensorflow.keras.models import load_model
            modelo = load_model('modelos/nn_hog.keras')
            
            with open('modelos/scaler_nn_hog.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            X_hog = extraer_hog(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_hog, y, test_size=0.2, random_state=42, stratify=y
            )
            X_test_scaled = scaler.transform(X_test)
            
            loss, precision = modelo.evaluate(X_test_scaled, np.eye(len(nombres_clases))[y_test], verbose=0)
            resultados['NN-HOG'] = precision
            print(f"NN-HOG: {precision:.2%}")
        except Exception as e:
            print(f"Error evaluando NN-HOG: {e}")

    if resultados:
        print(f"\nRESUMEN DE RENDIMIENTO:")
        mejor_modelo = max(resultados.items(), key=lambda x: x[1])
        for nombre, precision in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
            print(f"  {nombre}: {precision:.2%}")
        print(f"\nMejor modelo: {mejor_modelo[0]} ({mejor_modelo[1]:.2%})")

def main():
    import sys
    
    if len(sys.argv) == 1:
        print("Iniciando interfaz gráfica del clasificador...")
        from gui import ClasificadorSimpleGUI
        import tkinter as tk
        root = tk.Tk()
        app = ClasificadorSimpleGUI(root)
        root.mainloop()
        return
    
    parser = argparse.ArgumentParser(description='Clasificador de Autos y Motos')
    parser.add_argument('--accion', choices=['entrenar', 'evaluar', 'todo', 'gui'], 
                       default='gui', help='Acción a realizar')
    parser.add_argument('--modelo', choices=['svm-hog', 'svm-lbp', 'nn-hog', 'todos'], 
                       default='todos', help='Modelo a entrenar')
    
    args = parser.parse_args()
    
    if args.accion == 'gui':
        print("Iniciando interfaz gráfica del clasificador...")
        from gui import ClasificadorSimpleGUI
        import tkinter as tk
        root = tk.Tk()
        app = ClasificadorSimpleGUI(root)
        root.mainloop()
        return
    print("CLASIFICADOR DE AUTOS Y MOTOS")
    print("="*50)
    
    if args.accion in ['entrenar', 'todo']:
        print("Cargando datos de entrenamiento...")
        X, y, nombres_clases = cargar_imagenes('datos/imagenes')
        print(f"Total de imágenes: {len(X)}")
        print(f"Distribución: {dict(zip(nombres_clases, np.bincount(y)))}")
        resultados_entrenamiento = {}
        
        if args.modelo in ['svm-hog', 'todos']:
            modelo, precision = entrenar_svm_hog(X, y, nombres_clases)
            resultados_entrenamiento['SVM-HOG'] = precision
        
        if args.modelo in ['svm-lbp', 'todos']:
            modelo, precision = entrenar_svm_lbp(X, y, nombres_clases)
            resultados_entrenamiento['SVM-LBP'] = precision
        
        if args.modelo in ['nn-hog', 'todos']:
            modelo, precision = entrenar_red_neuronal(X, y, nombres_clases)
            if modelo is not None:
                resultados_entrenamiento['NN-HOG'] = precision
        
        if resultados_entrenamiento:
            print(f"\nRESUMEN DE ENTRENAMIENTO:")
            for nombre, precision in resultados_entrenamiento.items():
                print(f"  {nombre}: {precision:.2%}")
    
    if args.accion in ['evaluar', 'todo']:
        evaluar_modelos()
    
    print(f"\n¡Proceso completado!")
    print(f"Modelos guardados en la carpeta 'modelos/'")
    print(f"Para usar la interfaz gráfica, ejecute: python main.py")


if __name__ == "__main__":
    main()