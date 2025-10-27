from preprocesamiento import load_and_preprocess
from caracteristicas import extract_hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np

print("Cargando dataset...")
X, y, classes = load_and_preprocess('../datos/imagenes')
print("Extrayendo características HOG...")
X_hog = extract_hog(X)
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_hog, y_cat, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Creando modelo de red neuronal...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_hog.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("Entrenando red neuronal con HOG...")
historia = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Precisión final: {accuracy:.2%}")

model.save('../modelos/nn_hog.keras')
print("Modelo NN (HOG) guardado en ../modelos/nn_hog.keras")

import pickle
with open('../modelos/scaler_nn_hog.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler guardado en ../modelos/scaler_nn_hog.pkl")
