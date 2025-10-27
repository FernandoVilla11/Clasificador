import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from preprocesamiento import load_and_preprocess
from caracteristicas import extract_hog, extract_lbp


def cargar_modelo(ruta_modelo):
    if str(ruta_modelo).endswith('.pkl'):
        with open(ruta_modelo, 'rb') as f:
            return pickle.load(f)
    elif str(ruta_modelo).endswith('.keras'):
        from tensorflow.keras.models import load_model
        return load_model(ruta_modelo)
    else:
        raise ValueError(f"Formato de archivo no soportado: {ruta_modelo}")


def evaluar_modelo_svm(modelo, X_test, y_test, nombres_clases, titulo):
    print(f"\n {'='*50}")
    print(f"EVALUACIÓN: {titulo}")
    print(f"{'='*50}")
    
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precisión: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title(f'Matriz de Confusión - {titulo}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }


def evaluar_modelo_nn(modelo, scaler, X_test, y_test, nombres_clases, titulo):
    """Evalúa un modelo de red neuronal"""
    print(f"\n {'='*50}")
    print(f"EVALUACIÓN: {titulo}")
    print(f"{'='*50}")
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_prob = modelo.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precisión: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title(f'Matriz de Confusión - {titulo}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }


def comparar_modelos():
    print("Cargando datos de prueba...")
    X, y, nombres_clases = load_and_preprocess('../datos/imagenes')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    resultados = {}
    
    try:
        print("\nEvaluando SVM-HOG...")
        modelo_svm_hog = cargar_modelo('../modelos/svm_hog.pkl')
        X_test_hog = extract_hog(X_test)
        resultados['SVM-HOG'] = evaluar_modelo_svm(
            modelo_svm_hog, X_test_hog, y_test, nombres_clases, 'SVM con HOG'
        )
    except Exception as e:
        print(f"Error evaluando SVM-HOG: {e}")
    
    try:
        print("\nEvaluando SVM-LBP...")
        modelo_svm_lbp = cargar_modelo('../modelos/svm_lbp.pkl')
        X_test_lbp = extract_lbp(X_test)
        resultados['SVM-LBP'] = evaluar_modelo_svm(
            modelo_svm_lbp, X_test_lbp, y_test, nombres_clases, 'SVM con LBP'
        )
    except Exception as e:
        print(f"Error evaluando SVM-LBP: {e}")
    
    try:
        print("\nEvaluando NN-HOG...")
        modelo_nn_hog = cargar_modelo('../modelos/nn_hog.keras')
        with open('../modelos/scaler_nn_hog.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test_hog = extract_hog(X_test)
        resultados['NN-HOG'] = evaluar_modelo_nn(
            modelo_nn_hog, scaler, X_test_hog, y_test, nombres_clases, 'Red Neuronal con HOG'
        )
    except Exception as e:
        print(f"Error evaluando NN-HOG: {e}")
    
    if resultados:
        print(f"\n {'='*60}")
        print("RESUMEN COMPARATIVO DE MODELOS")
        print(f"{'='*60}")
        print(f"{'Modelo':<15} {'Precisión':<12} {'F1-Score':<12} {'Recall':<12}")
        print("-" * 60)
        
        for nombre, metricas in resultados.items():
            print(f"{nombre:<15} {metricas['accuracy']:<12.4f} "
                  f"{metricas['f1_score']:<12.4f} {metricas['recall']:<12.4f}")
        
        mejor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nMejor modelo: {mejor_modelo[0]} "
              f"(Precisión: {mejor_modelo[1]['accuracy']:.2%})")


def main():
    print("Iniciando evaluación de modelos...")
    comparar_modelos()
    print("\nEvaluación completada!")


if __name__ == "__main__":
    main()
