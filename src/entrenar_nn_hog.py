from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import os
import numpy as np

# Configuración de Tensorflow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from preprocesamiento import load_and_preprocess
from caracteristicas import extract_hog

# Asegurarse de que el directorio de modelos exista
os.makedirs('modelos', exist_ok=True)

print("Cargando dataset...")
X, y, classes = load_and_preprocess('data/imagenes')

print("Extrayendo características HOG...")
X_hog = extract_hog(X)

# El scaler_hog.pkl debería existir ya del script SVM-HOG, lo cargamos o creamos
try:
    with open('modelos/scaler_hog.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(" Scaler de HOG cargado.")
except FileNotFoundError:
    print(" Creando nuevo Scaler de HOG...")
    scaler = StandardScaler()
    scaler.fit(X_hog)
    with open('modelos/scaler_hog.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
X_hog_scaled = scaler.transform(X_hog)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_hog_scaled, y, test_size=0.2, random_state=42)

# Convertir etiquetas a formato one-hot encoding para Keras
y_train_one_hot = to_categorical(y_train, num_classes=len(classes))
y_test_one_hot = to_categorical(y_test, num_classes=len(classes))

print("Entrenando Red Neuronal (HOG)...")

# Construcción del modelo NN (MLP simple)
input_dim = X_train.shape[1] # Dimensiones de las características HOG
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax') # Capa de salida para clasificación binaria
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(X_train, y_train_one_hot,
                    epochs=20, # Número de épocas de entrenamiento
                    batch_size=32,
                    validation_data=(X_test, y_test_one_hot),
                    verbose=1)

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)

print("\n\n🔹 Resultados del modelo Red Neuronal (HOG):")
print(f"Pérdida (Loss): {loss:.4f}")
print(f"Precisión (Accuracy): {accuracy:.4f}")

# Guardar el modelo
model.save('modelos/nn_hog.keras')
print("\n✅ Modelo Red Neuronal (HOG) guardado en modelos/nn_hog.keras")
