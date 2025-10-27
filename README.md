# Clasificador de Autos y Motos# Proyecto Clasificador de Autos y Bicicletas



Sistema inteligente de clasificación de imágenes que distingue entre automóviles y motocicletas usando Machine Learning.## Descripción

Este proyecto implementa un sistema de clasificación de imágenes para distinguir entre autos y bicicletas utilizando diferentes técnicas de machine learning y procesamiento de imágenes.

## Instalación y Uso

## Estructura del Proyecto

### 1. Instalar dependencias

```bash```

pip install -r requirements.txtproyecto_auto_bicicleta/

```│

├── datos/

### 2. Ejecutar aplicación│   ├── originales/

```bash│   │   ├── auto/          # Imágenes de autos para entrenamiento

python main.py│   │   └── bicicleta/     # Imágenes de bicicletas para entrenamiento

```│   ├── procesados/        # Datos procesados y características extraídas

│   └── muestras_prueba/   # Imágenes para probar los modelos

## Modelos Disponibles│

├── src/

| Modelo | Precisión | Descripción |│   ├── preprocesamiento.py    # Funciones de preprocesamiento de imágenes

|--------|-----------|-------------|│   ├── caracteristicas.py     # Extracción de características (HOG, LBP)

| **NN-HOG** | **94.13%** | Red Neuronal + características HOG |│   ├── entrenar_hog_svm.py    # Entrenamiento SVM con características HOG

| **SVM-HOG** | **91.75%** | Máquina de Vectores de Soporte + HOG |│   ├── entrenar_lbp_svm.py    # Entrenamiento SVM con características LBP

| **SVM-LBP** | **84.12%** | SVM + Local Binary Patterns |

## 🛠️ Estructura del Proyecto

```
├── main.py                    # Ejecutor principal
├── gui.py                     # Interfaz gráfica
├── clasificador.py           # Pipeline de entrenamiento│   ├── entrenar_nn_hog.py     # Entrenamiento Red Neuronal con HOG

│   ├── evaluar.py             # Evaluación y comparación de modelos

## Estructura del Proyecto│   ├── utilidades_modelo.py   # Utilidades para manejo de modelos

│   └── gui.py                 # Interfaz gráfica de usuario

```│

├── main.py                    # Ejecutor principal├── modelos/

├── gui_simple.py             # Interfaz gráfica│   ├── svm_hog.pkl           # Modelo SVM entrenado con HOG

├── clasificador_completo.py  # Pipeline de entrenamiento│   ├── svm_lbp.pkl           # Modelo SVM entrenado con LBP

├── requirements.txt          # Dependencias│   └── nn_hog.keras          # Modelo de Red Neuronal con HOG

├── modelos/                  # Modelos entrenados│

└── datos/imagenes/           # Dataset de imágenes├── requirements.txt          # Dependencias del proyecto

```└── README.md                # Este archivo

```

## Dependencias

## Características del Proyecto

- OpenCV: Procesamiento de imágenes

- scikit-learn: Algoritmos ML### Técnicas de Extracción de Características

- scikit-image: Características avanzadas- **HOG (Histogram of Oriented Gradients)**: Extrae características basadas en gradientes

- TensorFlow: Redes neuronales- **LBP (Local Binary Patterns)**: Extrae patrones locales de textura

- PIL: Manipulación de imágenes

- NumPy: Operaciones numéricas### Modelos de Machine Learning

- **SVM (Support Vector Machine)**: Con características HOG y LBP

## Uso- **Red Neuronal**: Perceptrón multicapa con características HOG



1. Ejecuta `python main.py`### Funcionalidades

2. Selecciona modelo (NN-HOG recomendado)- Preprocesamiento automático de imágenes

3. Carga modelo y imagen- Entrenamiento de múltiples modelos

4. Obtén clasificación instantánea- Evaluación y comparación de rendimiento

- Interfaz gráfica para clasificación interactiva

---

**Desarrollado para Procesamiento Digital de Imágenes**## Instalación

1. Clonar o descargar el proyecto
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Preparar los Datos
- Colocar imágenes de autos en `datos/originales/auto/`
- Colocar imágenes de bicicletas en `datos/originales/bicicleta/`
- Colocar imágenes de prueba en `datos/muestras_prueba/`

### 2. Entrenar los Modelos

#### SVM con características HOG:
```bash
cd src
python entrenar_hog_svm.py
```

#### SVM con características LBP:
```bash
cd src
python entrenar_lbp_svm.py
```

#### Red Neuronal con HOG:
```bash
cd src
python entrenar_nn_hog.py
```

### 3. Evaluar los Modelos
```bash
cd src
python evaluar.py
```

### 4. Usar la Interfaz Gráfica
```bash
cd src
python gui.py
```

## Dependencias Principales

- **OpenCV**: Procesamiento de imágenes
- **scikit-image**: Extracción de características
- **scikit-learn**: Algoritmos de machine learning
- **TensorFlow**: Redes neuronales
- **tkinter**: Interfaz gráfica
- **matplotlib/seaborn**: Visualización

## Flujo de Trabajo

1. **Preprocesamiento**: Las imágenes se redimensionan y normalizan
2. **Extracción de Características**: Se extraen características HOG o LBP
3. **Entrenamiento**: Se entrenan los modelos con validación cruzada
4. **Evaluación**: Se compara el rendimiento de los modelos
5. **Predicción**: Se usa la GUI para clasificar nuevas imágenes

## Métricas de Evaluación

- Precisión (Accuracy)
- Precisión por clase
- Recall
- F1-Score
- Matriz de confusión

## Notas Técnicas

### Parámetros por Defecto

#### Características HOG:
- Orientaciones: 9
- Píxeles por celda: (8, 8)
- Celdas por bloque: (2, 2)

#### Características LBP:
- Radio: 3
- Puntos de muestreo: 24
- Método: 'uniform'

#### Red Neuronal:
- Capas: [512, 256, 128, 64, 1]
- Activación: ReLU (capas ocultas), Sigmoid (salida)
- Optimizador: Adam
- Dropout: 0.2-0.3

## Extensiones Posibles

- Agregar más clases de vehículos
- Implementar técnicas de aumento de datos
- Usar redes neuronales convolucionales (CNN)
- Implementar detección de objetos
- Añadir más métricas de evaluación

## Solución de Problemas

### Error de importación
Si hay errores de importación, verificar que todas las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

### Falta de imágenes
Asegurarse de que hay suficientes imágenes en las carpetas de entrenamiento (mínimo 50 por clase recomendado).

### Problemas de memoria
Para conjuntos de datos grandes, considerar:
- Reducir el tamaño de las imágenes
- Procesar en lotes más pequeños
- Usar menos características

## Licencia

Este proyecto es para fines educativos y de investigación.

## Autor

Proyecto desarrollado para clasificación de imágenes usando técnicas de machine learning y procesamiento de imágenes.