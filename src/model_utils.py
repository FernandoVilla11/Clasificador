import pickle
import numpy as np
from pathlib import Path


def guardar_modelo(modelo, ruta_archivo):
    ruta_archivo = Path(ruta_archivo)
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    
    if ruta_archivo.suffix == '.pkl':
        with open(ruta_archivo, 'wb') as f:
            pickle.dump(modelo, f)
    elif ruta_archivo.suffix == '.keras':
        modelo.save(ruta_archivo)
    else:
        raise ValueError(f"Formato de archivo no soportado: {ruta_archivo.suffix}")
    
    print(f"Modelo guardado en: {ruta_archivo}")


def cargar_modelo(ruta_archivo):
    ruta_archivo = Path(ruta_archivo)
    
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_archivo}")
    
    if ruta_archivo.suffix == '.pkl':
        with open(ruta_archivo, 'rb') as f:
            modelo = pickle.load(f)
    elif ruta_archivo.suffix == '.keras':
        try:
            from tensorflow.keras.models import load_model
            modelo = load_model(ruta_archivo)
        except ImportError:
            raise ImportError("TensorFlow no está instalado. Instale tensorflow para cargar modelos .keras")
    else:
        raise ValueError(f"Formato de archivo no soportado: {ruta_archivo.suffix}")
    
    return modelo


def predecir_imagen(modelo, imagen, tipo_modelo='svm', scaler=None):
    if tipo_modelo == 'nn':
        if scaler is None:
            raise ValueError("Se requiere scaler para modelos de red neuronal")
        
        imagen_scaled = scaler.transform([imagen])
        probabilidad = modelo.predict(imagen_scaled)[0][0]
        prediccion = int(probabilidad > 0.5)
        
        return prediccion, probabilidad
    
    else:
        prediccion = modelo.predict([imagen])[0]
        
        try:
            probabilidades = modelo.predict_proba([imagen])[0]
            probabilidad = probabilidades[int(prediccion)]
        except:
            probabilidad = None
        
        return int(prediccion), probabilidad


def obtener_info_modelo(ruta_modelo):
    ruta_modelo = Path(ruta_modelo)
    
    info = {
        'nombre': ruta_modelo.name,
        'tipo': ruta_modelo.suffix,
        'tamaño_mb': ruta_modelo.stat().st_size / (1024 * 1024),
        'existe': ruta_modelo.exists()
    }
    
    if info['existe']:
        try:
            modelo = cargar_modelo(ruta_modelo)
            
            if ruta_modelo.suffix == '.pkl':
                if hasattr(modelo, 'n_features_in_'):
                    info['caracteristicas_entrada'] = modelo.n_features_in_
                if hasattr(modelo, 'classes_'):
                    info['clases'] = modelo.classes_.tolist()
                if hasattr(modelo, 'kernel'):
                    info['kernel'] = modelo.kernel
                if hasattr(modelo, 'C'):
                    info['C'] = modelo.C
                    
            elif ruta_modelo.suffix == '.keras':
                info['capas'] = len(modelo.layers)
                info['parametros'] = modelo.count_params()
                info['input_shape'] = modelo.input_shape
                info['output_shape'] = modelo.output_shape
                
        except Exception as e:
            info['error'] = str(e)
    
    return info


def listar_modelos(ruta_directorio):
    ruta_directorio = Path(ruta_directorio)
    
    if not ruta_directorio.exists():
        return []
    
    modelos = []
    extensiones_validas = ['.pkl', '.keras', '.h5']
    
    for archivo in ruta_directorio.iterdir():
        if archivo.suffix in extensiones_validas:
            info = obtener_info_modelo(archivo)
            modelos.append(info)
    
    return modelos


def validar_estructura_proyecto(ruta_proyecto):
    ruta_proyecto = Path(ruta_proyecto)
    
    estructura_requerida = {
        'datos': ['originales/auto', 'originales/moto', 'procesados', 'muestras_prueba'],
        'src': ['preprocesamiento.py', 'caracteristicas.py', 'entrenar_hog_svm.py', 
               'entrenar_lbp_svm.py', 'entrenar_nn_hog.py', 'evaluar.py', 
               'utilidades_modelo.py', 'gui.py'],
        'modelos': []
    }
    
    validacion = {
        'proyecto_existe': ruta_proyecto.exists(),
        'directorios': {},
        'archivos': {},
        'estructura_completa': True
    }
    
    if not validacion['proyecto_existe']:
        validacion['estructura_completa'] = False
        return validacion
    
    for directorio, subdirectorios in estructura_requerida.items():
        dir_path = ruta_proyecto / directorio
        validacion['directorios'][directorio] = {
            'existe': dir_path.exists(),
            'subdirectorios': {}
        }
        
        if not dir_path.exists():
            validacion['estructura_completa'] = False
            continue
        
        for subdir in subdirectorios:
            subdir_path = dir_path / subdir
            validacion['directorios'][directorio]['subdirectorios'][subdir] = subdir_path.exists()
            
            if not subdir_path.exists():
                validacion['estructura_completa'] = False
    
    return validacion