# 📊 Dataset para el Clasificador

## ⚠️ Información Importante

El dataset de imágenes **NO está incluido** en este repositorio debido a su gran tamaño (varios GB). 

## 📁 Estructura Requerida del Dataset

Para usar este clasificador, necesitas crear la siguiente estructura de carpetas:

```
datos/
├── imagenes/
│   ├── Car/          # Imágenes de automóviles
│   │   ├── Car (1).jpeg
│   │   ├── Car (2).jpeg
│   │   └── ...
│   └── Motorcycle/   # Imágenes de motocicletas
│       ├── Motorcycle (1).jpeg
│       ├── Motorcycle (2).jpeg
│       └── ...
```

## 🔍 Fuentes de Dataset Recomendadas

Puedes obtener imágenes de automóviles y motocicletas de:

1. **Kaggle Datasets**
   - [Vehicle Classification Dataset](https://www.kaggle.com/datasets/kaggleashwin/vehicle-classification)
   - [Cars vs Motorcycles](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset)

2. **Open Images Dataset**
   - [Google Open Images](https://storage.googleapis.com/openimages/web/index.html)

3. **ImageNet**
   - Categorías: "motor vehicle", "motorcycle"

## 📏 Especificaciones del Dataset

- **Formato**: JPEG, PNG
- **Cantidad mínima recomendada**: 1000 imágenes por clase
- **Resolución**: Cualquier resolución (se redimensiona automáticamente)
- **Clases**: 2 (Car, Motorcycle)

## 🚀 Cómo Usar con tu Dataset

1. Descarga o recopila imágenes de automóviles y motocicletas
2. Organízalas en la estructura de carpetas mostrada arriba
3. Ejecuta `python clasificador.py` para entrenar nuevos modelos
4. O usa los modelos pre-entrenados incluidos con `python main.py`

## 📈 Modelos Pre-entrenados Incluidos

Los modelos en la carpeta `modelos/` fueron entrenados con:
- **~2000 imágenes de automóviles**
- **~1000 imágenes de motocicletas**
- **Precisión alcanzada**: 84.12% - 94.13%