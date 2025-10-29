"""
Script de entrenamiento SVM con característica seleccionable (SIFT, SURF o LAB)
Autor: Sistema de clasificación de imágenes
Fecha: 2025-10-29
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump
import os
from preprocesamiento import load_and_preprocess
from caracteristicas import extract_sift, extract_surf, extract_lab

# ============================================================================
# CONFIGURACIÓN: Selecciona la característica a utilizar
# ============================================================================
# Opciones disponibles: 'sift', 'surf', 'lab'
CARACTERISTICA_SELECCIONADA = 'sift'  # 👈 Cambia aquí para usar otra característica
# ============================================================================

# Diccionario para mapear la característica seleccionada a su función correspondiente
EXTRACTORES = {
    'sift': extract_sift,
    'surf': extract_surf,
    'lab': extract_lab
}

def main():
    """
    Función principal para entrenar el modelo SVM con la característica seleccionada.
    """
    
    # Validar que la característica seleccionada sea válida
    if CARACTERISTICA_SELECCIONADA not in EXTRACTORES:
        print(f"❌ Error: '{CARACTERISTICA_SELECCIONADA}' no es una característica válida.")
        print(f"   Opciones disponibles: {list(EXTRACTORES.keys())}")
        return
    
    print("=" * 70)
    print(f"🚀 ENTRENAMIENTO SVM CON {CARACTERISTICA_SELECCIONADA.upper()}")
    print("=" * 70)
    
    # Paso 1: Cargar y preprocesar el dataset
    print("\n📂 Cargando dataset...")
    try:
        X, y, classes = load_and_preprocess('data/imagenes')
        print(f"✅ Dataset cargado: {len(X)} imágenes, {len(classes)} clases")
        print(f"   Clases detectadas: {classes}")
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return
    
    # Paso 2: Extraer características usando la función seleccionada
    print(f"\n🔍 Extrayendo características {CARACTERISTICA_SELECCIONADA.upper()}...")
    try:
        extractor = EXTRACTORES[CARACTERISTICA_SELECCIONADA]
        X_features = extractor(X)
        print(f"✅ Características extraídas: {X_features.shape[0]} imágenes × {X_features.shape[1]} features")
    except Exception as e:
        print(f"❌ Error al extraer características: {e}")
        return
    
    # Paso 3: Dividir en conjuntos de entrenamiento y prueba
    print("\n📊 Dividiendo dataset en entrenamiento (80%) y prueba (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Mantiene la proporción de clases
    )
    print(f"✅ Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")
    
    # Paso 4: Entrenar el modelo SVM
    print(f"\n🤖 Entrenando modelo SVM (kernel lineal)...")
    try:
        svm = SVC(kernel='linear', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        print("✅ Entrenamiento completado")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return
    
    # Paso 5: Evaluar el modelo
    print("\n📈 Evaluando modelo en conjunto de prueba...")
    y_pred = svm.predict(X_test)
    
    print("\n" + "=" * 70)
    print("🔹 RESULTADOS DEL MODELO SVM")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Paso 6: Calcular y mostrar precisión general
    accuracy = svm.score(X_test, y_test)
    print(f"🎯 Precisión general: {accuracy * 100:.2f}%")
    
    # Paso 7: Guardar el modelo entrenado
    print(f"\n💾 Guardando modelo...")
    
    # Crear directorio si no existe
    os.makedirs('modelos', exist_ok=True)
    
    # Guardar modelo
    model_filename = f'modelos/svm_{CARACTERISTICA_SELECCIONADA}.pkl'
    try:
        dump(svm, model_filename)
        print(f"✅ Modelo guardado exitosamente en: {model_filename}")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")
        return
    
    print("\n" + "=" * 70)
    print("✨ PROCESO COMPLETADO CON ÉXITO")
    print("=" * 70)

if __name__ == "__main__":
    main()