import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import pickle
import os
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools 


# Importaciones existentes para el clasificador
from skimage.feature import hog, local_binary_pattern
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler

# Dimensiones de las imágenes usadas para el entrenamiento
TARGET_SIZE = (128, 128)
# Nombres de las clases, deben coincidir con el entrenamiento
CLASE_NAMES = ['Bike', 'Car'] 
CLASE_MAP = {'Bike': 0, 'Car': 1} # Para mapeo de etiquetas verdaderas
CLASE_EMOJIS = {'Bike': '🏍️', 'Car': '🚘'}


# --- Extracción de Características (Mantenida) ---

def extraer_hog(imagen: np.ndarray) -> np.ndarray:
    """Extrae características HOG de una imagen redimensionada a 128x128."""
    return hog(imagen, 
               pixels_per_cell=(8, 8), 
               cells_per_block=(2, 2), 
               feature_vector=True,
               block_norm='L2-Hys')

def extraer_lbp_mejorado(imagen: np.ndarray) -> np.ndarray:
    """Extrae LBP mejorado con características estadísticas y de gradiente."""
    caracteristicas = []
    
    configuraciones = [
        {'P': 8, 'R': 1, 'method': 'uniform'},
        {'P': 16, 'R': 2, 'method': 'uniform'}, 
        {'P': 24, 'R': 3, 'method': 'uniform'}
    ]
    
    for config in configuraciones:
        lbp = local_binary_pattern(imagen, P=config['P'], R=config['R'], method=config['method'])
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)
        caracteristicas.extend(hist)
    
    caracteristicas.extend([
        np.mean(imagen), np.std(imagen), np.var(imagen), np.min(imagen), np.max(imagen)
    ])
    
    grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.sqrt(grad_x**2 + grad_y**2)
    caracteristicas.extend([np.mean(magnitud), np.std(magnitud)])
    
    kernel = np.ones((3,3), np.float32) / 9
    imagen_suavizada = cv2.filter2D(imagen, -1, kernel)
    contraste = np.abs(imagen.astype(float) - imagen_suavizada.astype(float))
    caracteristicas.extend([np.mean(contraste), np.std(contraste)])
    
    return np.array(caracteristicas)


# --- Interfaz Gráfica (Modificada) ---

class ClasificadorSimpleGUI:
    def __init__(self, root: tk.Tk, nombresModelos: List[str]|None = None):
        self.root = root
        self.root.title("Modelo clasificador PDI: Autos y Motos - Análisis de Métricas")
        
        self.nombresModelos = nombresModelos
        
        # ancho = root.winfo_screenwidth()
        # alto = root.winfo_screenheight()
        # self.root.geometry(f"{ancho}x{alto}") // Por si no funciona el zoomed
        root.state('zoomed')
        
        # Variables de estado
        self.rutas_imagenes: List[str] = []
        self.imagen_tk: ImageTk.PhotoImage | None = None
        self.modelo_cargado = None
        self.tipo_modelo: str | None = None
        self.scaler: StandardScaler | None = None
        self.scaler_lbp: StandardScaler | None = None
        
        # Variables de resultados (NUEVO)
        self.y_pred: np.ndarray = np.array([])
        self.y_true_manual: np.ndarray = np.array([])
        self.y_prob: np.ndarray | None = None
        
        self.configurar_interfaz()
        self.detectar_modelos()
        
    def configurar_interfaz(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        titulo = tk.Label(main_frame, text="Clasificador de Autos y Motos", 
                             font=('Arial', 24, 'bold'))
        titulo.pack(pady=(0, 20))
        
        control_frame = tk.LabelFrame(main_frame, text="Controles", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # --- Fila de controles 1: Modelos y Lote ---
        fila1 = tk.Frame(control_frame)
        fila1.pack(fill=tk.X, pady=(0, 10))

        # Columna izquierda: selección de modelo
        frame_modelo = tk.Frame(fila1)
        frame_modelo.pack(side=tk.LEFT, padx=10)

        tk.Label(frame_modelo, text="Modelo:").pack(side=tk.LEFT)
        self.combo_modelo = ttk.Combobox(frame_modelo, state="readonly", width=20)
        self.combo_modelo.pack(side=tk.LEFT, padx=5)
        tk.Button(frame_modelo, text="Cargar Modelo", command=self.cargar_modelo, bg='#A0D8F9').pack(side=tk.LEFT, padx=5)

        # Columna central: carga de imágenes
        frame_lote = tk.Frame(fila1)
        frame_lote.pack(side=tk.LEFT, padx=20)

        tk.Button(frame_lote, text="Cargar Imagen Individual", command=lambda: self.cargar_imagen(False),
                bg='#A3EBB1').pack(side=tk.LEFT, padx=5)
        tk.Button(frame_lote, text="Cargar Lote (Múltiple)", command=lambda: self.cargar_imagen(True),
                bg='#A3EBB1', foreground='black').pack(side=tk.LEFT, padx=5)
        
        self.label_lote = tk.Label(frame_lote, text="Lote: 0 archivos", width=15, anchor='w')
        self.label_lote.pack(side=tk.LEFT, padx=5)

        # Columna derecha: acciones de clasificación
        frame_acciones = tk.Frame(fila1)
        frame_acciones.pack(side=tk.RIGHT, padx=10)

        tk.Button(frame_acciones, text="Clasificar", command=self.clasificar_lote,
                bg='#FF9966', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        self.btn_metricas = tk.Button(frame_acciones, text="Ajustar y Calcular Métricas (Y-True)",
                                    command=self.abrir_ventana_metricas,
                                    bg='#FFAAAA', state=tk.DISABLED)
        self.btn_metricas.pack(side=tk.LEFT, padx=5)
        
        
        # Marco Principal de Contenido
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo: Imagen
        left_frame = tk.LabelFrame(content_frame, text="Imagen Actual (Muestra)", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.label_imagen = tk.Label(left_frame, text="No hay imagen cargada", 
                                     bg='white', relief=tk.SUNKEN, bd=2)
        self.label_imagen.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel derecho: Notebook para Resultados y Métricas (MODIFICADO)
        right_frame = tk.LabelFrame(content_frame, text="Análisis", padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Resultados de Predicción (Texto)
        self.tab_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_resultados, text='Resultados de Predicción')
        
        self.texto_resultado = tk.Text(self.tab_resultados, state=tk.DISABLED, wrap=tk.WORD,
                                     font=('Consolas', 9))
        scrollbar = tk.Scrollbar(self.tab_resultados, orient="vertical", 
                                 command=self.texto_resultado.yview)
        self.texto_resultado.configure(yscrollcommand=scrollbar.set)
        
        self.texto_resultado.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Gráficos de Métricas
        self.tab_metricas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metricas, text='Métricas y Gráficos')
        
        # Placeholder para el lienzo de matplotlib
        self.canvas_placeholder = tk.Label(self.tab_metricas, 
                                           text="Clasifique un lote y presione 'Ajustar y Calcular Métricas' para ver los gráficos.", 
                                           pady=20)
        self.canvas_placeholder.pack(fill=tk.BOTH, expand=True)
        
        self.label_estado = tk.Label(main_frame, text="Listo", relief=tk.SUNKEN, 
                                     bd=1, anchor='w')
        self.label_estado.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        self.mostrar_mensaje_inicial()
        
    def detectar_modelos(self):
        """Detecta los modelos definidos en self.nombresModelos que existan en la carpeta 'modelos'."""
        modelos_disponibles = []

        if not self.nombresModelos:
            self.actualizar_estado("ADVERTENCIA: No se proporcionaron nombres de modelos para detectar.")
            return

        # Revisa la existencia de cada modelo en la lista
        for ruta in self.nombresModelos:
            ruta_completa = os.path.join(os.getcwd(), ruta)

            if os.path.exists(ruta_completa):
                nombre_archivo = os.path.basename(ruta)  # ejemplo: 'svm_hog.pkl'

                # Generar un nombre legible
                if nombre_archivo.endswith('.keras'):
                    modelos_disponibles.append(nombre_archivo.replace('.keras', '').replace('_', '-').upper())
                elif nombre_archivo.endswith('.pkl'):
                    modelos_disponibles.append(nombre_archivo.replace('.pkl', '').replace('_', '-').upper())
            else:
                print(f"⚠️ No se encontró: {ruta_completa}")

        # Actualizar el combobox con los modelos encontrados
        if modelos_disponibles:
            self.combo_modelo['values'] = modelos_disponibles
            self.combo_modelo.current(0)
            self.actualizar_estado(f"Detectados {len(modelos_disponibles)} modelos disponibles")
        else:
            self.combo_modelo['values'] = []
            self.actualizar_estado("ADVERTENCIA: No se encontraron los modelos especificados")

        self.combo_modelo.update_idletasks()
        self.root.update_idletasks()

                
    def mostrar_mensaje_inicial(self):
        """Muestra la información inicial e instrucciones."""
        mensaje = (
            "CLASIFICADOR DE AUTOS Y MOTOS (PDI - TAREA 2)\n\n"
            "Sistema de clasificación de imágenes basado en descriptores (HOG, LBP) y modelos de ML/DL (SVM, NN).\n\n"
            "INSTRUCCIONES:\n"
            "1. Seleccione un modelo y cárguelo.\n"
            "2. Cargue una o varias imágenes.\n"
            "3. Presione 'Clasificar'.\n"
            "4. Presione 'Ajustar y Calcular Métricas' para simular y visualizar el rendimiento (Accuracy, F1, Matriz de Confusión)."
        )
        
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        self.texto_resultado.insert(tk.END, mensaje.strip())
        self.texto_resultado.config(state=tk.DISABLED)
    
    def cargar_modelo(self):
        """Carga el modelo de clasificación y su escalador (scaler) asociado."""
        modelo_seleccionado = self.combo_modelo.get()
        if not modelo_seleccionado:
            messagebox.showwarning("Advertencia", "Seleccione un modelo primero")
            return
        
        try:
            self.actualizar_estado("Cargando modelo...")
            self.modelo_cargado = None
            self.scaler = None
            self.scaler_lbp = None
            
            # --- 1. Lógica para Modelos SVM (joblib .pkl) ---
            if 'SVM' in modelo_seleccionado:
                nombre_modelo = modelo_seleccionado.lower().replace("-", "_")
                ruta_modelo = os.path.join('modelos', f'{nombre_modelo}.pkl')
                
                if not os.path.exists(ruta_modelo):
                    raise FileNotFoundError(f"Archivo de modelo '{os.path.basename(ruta_modelo)}' no encontrado")
                
                self.modelo_cargado = joblib_load(ruta_modelo)
                self.tipo_modelo = nombre_modelo # svm_hog o svm_lbp
                
                if modelo_seleccionado == 'SVM-LBP':
                    ruta_scaler = os.path.join('modelos', 'scaler_lbp.pkl')
                    if not os.path.exists(ruta_scaler): raise FileNotFoundError(f"Archivo de escalador '{os.path.basename(ruta_scaler)}' no encontrado")
                    self.scaler_lbp = joblib_load(ruta_scaler)
                    self.tipo_modelo = 'svm-lbp-mejorado' # Clave para que clasificar_lote use LBP mejorado
                elif modelo_seleccionado == 'SVM-HOG':
                    ruta_scaler = os.path.join('modelos', 'scaler_hog.pkl')
                    if not os.path.exists(ruta_scaler): raise FileNotFoundError(f"Archivo de escalador '{os.path.basename(ruta_scaler)}' no encontrado")
                    self.scaler = joblib_load(ruta_scaler)

            # --- 2. Lógica para Modelo NN-HOG (Keras .keras) ---
            elif modelo_seleccionado == 'NN-HOG':
                ruta_modelo = os.path.join('modelos', 'nn_hog.keras')
                ruta_scaler = os.path.join('modelos', 'scaler_hog.pkl')
                
                if not os.path.exists(ruta_modelo): raise FileNotFoundError("Archivo nn_hog.keras no encontrado")
                if not os.path.exists(ruta_scaler): raise FileNotFoundError(f"Archivo de escalador '{os.path.basename(ruta_scaler)}' no encontrado")
                
                # Carga de Keras
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                import warnings
                warnings.filterwarnings('ignore')
                
                from tensorflow.keras.models import load_model
                self.modelo_cargado = load_model(ruta_modelo)
                
                with open(ruta_scaler, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.tipo_modelo = 'nn-hog'
                
            # --- 3. Lógica para Modelo NN-LBP (joblib .pkl) ---
            elif modelo_seleccionado == 'NN-LBP':
                nombre_modelo = modelo_seleccionado.lower().replace("-", "_")
                ruta_modelo = os.path.join('modelos', f'{nombre_modelo}.pkl') # nn_lbp.pkl
                
                if not os.path.exists(ruta_modelo): 
                    raise FileNotFoundError(f"Archivo de modelo '{os.path.basename(ruta_modelo)}' no encontrado")
                
                # Carga de joblib (igual que SVM)
                self.modelo_cargado = joblib_load(ruta_modelo)
                
                # Carga de Scaler LBP
                ruta_scaler = os.path.join('modelos', 'scaler_lbp.pkl')
                if not os.path.exists(ruta_scaler): raise FileNotFoundError(f"Archivo de escalador '{os.path.basename(ruta_scaler)}' no encontrado")
                self.scaler_lbp = joblib_load(ruta_scaler)
                
                self.tipo_modelo = 'nn-lbp-mejorado' # Clave para que clasificar_lote use LBP mejorado

            # --- 4. Caso Desconocido ---
            else:
                raise ValueError(f"Modelo desconocido: {modelo_seleccionado}")
            
            self.actualizar_estado(f"Modelo {modelo_seleccionado} cargado correctamente")
            messagebox.showinfo("Éxito", f"Modelo {modelo_seleccionado} cargado exitosamente")
            
        except FileNotFoundError as e:
            error_msg = f"Error de archivo: {str(e)}\nAsegúrate de que los archivos existan en 'modelos/'."
            self.actualizar_estado("Error: Archivo no encontrado")
            messagebox.showerror("Error", error_msg)
            
        except Exception as e:
            error_msg = f"Error cargando modelo: {type(e).__name__}: {str(e)}"
            self.actualizar_estado(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def cargar_imagen(self, multiple: bool):
        """Carga una o múltiples imágenes."""
        # Limpiar resultados anteriores de métricas
        self.y_pred = np.array([])
        self.y_true_manual = np.array([])
        self.btn_metricas.config(state=tk.DISABLED)
        self.limpiar_graficos()
        self.notebook.select(self.tab_resultados)

        tipos_archivo = [("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
        
        if multiple:
            rutas = filedialog.askopenfilenames(title="Seleccionar lote de imágenes", filetypes=tipos_archivo)
            self.rutas_imagenes = list(rutas)
            
            if self.rutas_imagenes:
                self._mostrar_imagen_individual(self.rutas_imagenes[0])
                self.label_lote.config(text=f"Lote: {len(self.rutas_imagenes)} archivos")
                self.actualizar_estado(f"Lote de {len(self.rutas_imagenes)} imágenes cargado.")
            else:
                self.actualizar_estado("Carga de lote cancelada.")
        
        else:
            ruta = filedialog.askopenfilename(title="Seleccionar imagen individual", filetypes=tipos_archivo)
            if ruta:
                self.rutas_imagenes = [ruta]
                self._mostrar_imagen_individual(ruta)
                nombre_archivo = os.path.basename(ruta)
                self.label_lote.config(text=f"Lote: 1 archivo")
                self.actualizar_estado(f"Imagen individual cargada: {nombre_archivo}")
            else:
                self.actualizar_estado("Carga de imagen cancelada.")
    
    def _mostrar_imagen_individual(self, ruta_imagen: str):
        """Muestra una única imagen en el panel de previsualización."""
        try:
            imagen_pil = Image.open(ruta_imagen)
            ancho_disp = self.label_imagen.winfo_width() if self.label_imagen.winfo_width() > 1 else 350
            alto_disp = self.label_imagen.winfo_height() if self.label_imagen.winfo_height() > 1 else 350
            
            imagen_pil.thumbnail((ancho_disp, alto_disp), Image.Resampling.LANCZOS)
            
            self.imagen_tk = ImageTk.PhotoImage(imagen_pil)
            self.label_imagen.configure(image=self.imagen_tk, text="")
        except Exception as e:
            self.label_imagen.configure(image="", text="Error al cargar imagen")
            self.imagen_tk = None

    def clasificar_lote(self):
        """Clasifica todas las imágenes cargadas en el lote."""
        if not self.rutas_imagenes or len(self.rutas_imagenes) < 1:
            messagebox.showwarning("Advertencia", "Cargue un lote de al menos 2 imágenes primero")
            return
        
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Cargue un modelo primero")
            return
        
        self.actualizar_estado(f"Clasificando {len(self.rutas_imagenes)} imágenes...")
        
        features_list = []
        rutas_validas = []
        
        try:
            # 1. Extracción de Características
            for ruta in self.rutas_imagenes:
                img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, TARGET_SIZE)
                
                if 'hog' in self.tipo_modelo:
                    caracteristicas = extraer_hog(img)
                elif self.tipo_modelo == 'svm-lbp-mejorado':
                    caracteristicas = extraer_lbp_mejorado(img)
                else: # LBP simple (Fallback)
                    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
                    n_bins = int(lbp.max() + 1) 
                    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)
                    caracteristicas = hist
                
                features_list.append(caracteristicas)
                rutas_validas.append(ruta)
            
            if not features_list: raise ValueError("Ninguna imagen fue cargada correctamente.")

            X_features = np.array(features_list)
            
            # 2. Escalado de Características
            scaler_to_use = self.scaler if self.tipo_modelo in ['nn-hog', 'svm-hog'] else self.scaler_lbp
            X_features_scaled = scaler_to_use.transform(X_features) if scaler_to_use is not None else X_features 

            # 3. Predicción
            if self.tipo_modelo == 'nn-hog':
                import warnings
                warnings.filterwarnings('ignore')
                predicciones_prob = self.modelo_cargado.predict(X_features_scaled, verbose=0)
                predicciones = np.argmax(predicciones_prob, axis=1)
            else:
                predicciones = self.modelo_cargado.predict(X_features_scaled)
                try:
                    # Comprobación más robusta para predict_proba
                    if hasattr(self.modelo_cargado, 'predict_proba') and callable(self.modelo_cargado.predict_proba):
                         predicciones_prob = self.modelo_cargado.predict_proba(X_features_scaled)
                    else:
                        predicciones_prob = None
                except (AttributeError, ValueError):
                    predicciones_prob = None
            
            # Almacenar resultados para métricas
            self.rutas_imagenes = rutas_validas
            self.y_pred = predicciones
            self.y_prob = predicciones_prob
            
            # **CAMBIO CRÍTICO: Inicializar Y_True con inferencia del nombre/ruta**
            self.y_true_manual = self.inferir_y_true(rutas_validas) 
            
            # 4. Mostrar Resultados y habilitar botón de métricas
            self.mostrar_resultados_lote(rutas_validas, self.y_pred, self.y_prob)
            self.btn_metricas.config(state=tk.NORMAL)
            self.actualizar_estado(f"Clasificación de {len(rutas_validas)} imágenes completada. Listo para calcular métricas.")
            
        except Exception as e:
            error_msg = f"Error grave en clasificación de lote: {type(e).__name__}: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.actualizar_estado(error_msg)

    def mostrar_resultados_lote(self, rutas: List[str], predicciones: np.ndarray, 
                                 probabilidades: np.ndarray | None):
        """Muestra un resumen de los resultados del lote en el área de texto, destacando la clase."""
        
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        
        texto_inicial = "RESULTADOS DE CLASIFICACIÓN ({0} IMÁGENES)\n".format(len(rutas))
        texto_inicial += "=" * 45 + "\n\n"
        self.texto_resultado.insert(tk.END, texto_inicial)
        
        conteo_clases = {clase: 0 for clase in CLASE_NAMES}
        
        for i, ruta in enumerate(rutas):
            nombre = os.path.basename(ruta)
            prediccion_idx = predicciones[i]
            clase_predicha = CLASE_NAMES[prediccion_idx]
            conteo_clases[clase_predicha] += 1
            
            icono = CLASE_EMOJIS.get(clase_predicha, '❓')
            
            # Formato de la línea: [XX] 🏍️ NombreArchivo.jpg: Clase (Probabilidad%)
            linea = f"[{i+1:02d}] {icono} {nombre}: {clase_predicha}"
            
            if probabilidades is not None and prediccion_idx < probabilidades.shape[1]:
                prob = probabilidades[i][prediccion_idx] * 100
                linea += f" ({prob:.1f}%)"
            
            self.texto_resultado.insert(tk.END, linea + "\n")
            
        # Resumen Final
        texto_resumen = "\n\n--- RESUMEN FINAL ---\n"
        texto_resumen += f"Modelo Usado: {self.combo_modelo.get()}\n"
        texto_resumen += f"Total de Archivos Procesados: {len(rutas)}\n\n"
        
        for clase, conteo in conteo_clases.items():
            icono = CLASE_EMOJIS.get(clase, '❓')
            if len(rutas) > 0:
                texto_resumen += f"{icono} {clase}: {conteo} ({conteo/len(rutas)*100:.1f}%)\n"
            
        self.texto_resultado.insert(tk.END, texto_resumen)
        self.texto_resultado.config(state=tk.DISABLED)

    # --- Funciones para la Ventana de Métricas (NUEVO) ---
    
    def abrir_ventana_metricas(self):
        """Abre una ventana Toplevel para que el usuario introduzca las etiquetas verdaderas."""
        if self.y_pred.size == 0:
            messagebox.showwarning("Advertencia", "Clasifique un lote primero.")
            return

        self.ventana_metricas = tk.Toplevel(self.root)
        self.ventana_metricas.title("Ajuste de Etiquetas Verdaderas (Y-True)")
        # Ajustar el tamaño para que sea más cómodo
        self.ventana_metricas.geometry("550x380") 

        tk.Label(self.ventana_metricas, text="Asignación de Etiquetas Verdaderas (Ground Truth)", 
                  font=('Arial', 12, 'bold')).pack(pady=10)

        instrucciones = (
            "Para calcular las métricas, introduzca la etiqueta VERDADERA (Bike o Car) de CADA imagen "
            "separada por comas (sin espacios). El orden es el mismo que en los resultados.\n\n"
            "**Valores por defecto:** Se ha intentado inferir la etiqueta del nombre de archivo. "
            "Si el valor tiene un '?', debe ser revisado y corregido manualmente."
        )
        tk.Label(self.ventana_metricas, text=instrucciones, wraplength=500, justify=tk.LEFT).pack(padx=10)

        tk.Label(self.ventana_metricas, text=f"Total de imágenes: {self.y_pred.size}").pack(pady=(10, 5))
        
        # Campo de entrada para etiquetas verdaderas
        tk.Label(self.ventana_metricas, text="Etiquetas (Y-True):", font=('Arial', 10, 'bold')).pack(padx=10, anchor='w')
        self.entry_y_true = tk.Entry(self.ventana_metricas, width=70, font=('Arial', 10))
        self.entry_y_true.pack(padx=10)

        # **CAMBIO CRÍTICO: Inicializar el campo con las etiquetas INFERIDAS**
        y_true_default = []
        for i, true_idx in enumerate(self.y_true_manual):
            if true_idx in CLASE_MAP.values(): # Si se infirió correctamente (0 o 1)
                y_true_default.append(CLASE_NAMES[true_idx])
            else:
                # Si no se infirió del nombre de archivo (-1), usar la predicción con un '?'
                pred_idx = self.y_pred[i]
                y_true_default.append(CLASE_NAMES[pred_idx] + "?")
                
        y_true_default_str = ", ".join(y_true_default)
        self.entry_y_true.insert(0, y_true_default_str)
        
        tk.Button(self.ventana_metricas, text="Calcular y Mostrar Métricas", 
                  command=self.procesar_y_mostrar_metricas, 
                  bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(pady=20)
        
        self.ventana_metricas.grab_set() # Hacer que esta ventana sea modal
        self.root.wait_window(self.ventana_metricas)


    def procesar_y_mostrar_metricas(self):
        """Procesa la entrada del usuario, calcula las métricas y dibuja los gráficos."""
        try:
            input_str = self.entry_y_true.get().replace(" ", "")
            y_true_str = [s.strip() for s in input_str.split(',')]
            
            if len(y_true_str) != self.y_pred.size:
                raise ValueError(f"Debe proporcionar {self.y_pred.size} etiquetas, se encontraron {len(y_true_str)}.")

            # Mapear etiquetas de texto a índices numéricos
            y_true_list = []
            for label in y_true_str:
                if label not in CLASE_MAP:
                    raise ValueError(f"Etiqueta desconocida: '{label}'. Use solo 'Bike' o 'Car'.")
                y_true_list.append(CLASE_MAP[label])
            
            self.y_true_manual = np.array(y_true_list)
            
            # Cerrar la ventana de ajuste
            self.ventana_metricas.destroy()
            self.actualizar_estado("Calculando y mostrando métricas...")
            
            # Dibujar los gráficos en la pestaña de Métricas
            self.mostrar_metricas_graficas()
            self.notebook.select(self.tab_metricas)
            
        except ValueError as e:
            messagebox.showerror("Error de Etiquetado", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}")


    def limpiar_graficos(self):
        """Limpia el contenido anterior del panel de gráficos."""
        for widget in self.tab_metricas.winfo_children():
            widget.destroy()
        
        self.canvas_placeholder = tk.Label(self.tab_metricas, 
                                           text="Clasifique un lote y presione 'Ajustar y Calcular Métricas' para ver los gráficos.", 
                                           pady=20)
        self.canvas_placeholder.pack(fill=tk.BOTH, expand=True)

    def mostrar_metricas_graficas(self):
        """Calcula y muestra la matriz de confusión y el reporte de clasificación (texto)."""
        self.limpiar_graficos()
        
        # 1. Preparar el lienzo de matplotlib (Solo 1 panel ahora para la Matriz)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5)) # Tamaño ajustado para 1 gráfico
        fig.tight_layout(pad=3.0)
        
        # 2. Matriz de Confusión y Reporte (Calculado ANTES)
        # MODIFICACIÓN: usar todos los valores de CLASE_MAP para asegurar que la matriz sea correcta
        cm = confusion_matrix(self.y_true_manual, self.y_pred, labels=list(CLASE_MAP.values()))
        reporte = classification_report(self.y_true_manual, self.y_pred, 
                                        target_names=CLASE_NAMES, output_dict=True)
                                        
        # 3. Dibujar la Matriz de Confusión con las métricas en texto
        self.plot_confusion_matrix(ax, cm, CLASE_NAMES, reporte,
                                title='Matriz de Confusión y Reporte de Métricas')
        
        # 4. Integrar Matplotlib en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tab_metricas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        self.actualizar_estado(f"Métricas calculadas y mostradas para {self.y_pred.size} muestras.")




    def plot_confusion_matrix(self, ax, cm, classes, reporte, title='Matriz de Confusión'):
        """Dibuja la matriz de confusión e incluye las métricas Precision, Recall y F1-Score en texto."""
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12)
        
        # Ocultar ticks
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        
        # Asignar etiquetas
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        
        ax.set_ylabel('Etiqueta Verdadera', fontsize=10)
        ax.set_xlabel('Etiqueta Predicha', fontsize=10)

        # Añadir las anotaciones (Números)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
        
        # --- NUEVO FORMATO DE REPORTE EN TEXTO ---
        texto_adicional = f"Accuracy Global: {reporte['accuracy']:.4f}\n\n"
        
        for clase in classes:
            # Protección en caso de que la clase no tenga predicciones
            clase_reporte = reporte.get(clase, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0})
            texto_adicional += f"--- {clase} ---\n"
            texto_adicional += f"Precisión: {clase_reporte['precision']:.4f}\n"
            texto_adicional += f"Recall: {clase_reporte['recall']:.4f}\n"
            texto_adicional += f"F1-Score: {clase_reporte['f1-score']:.4f}\n\n"
            
        texto_adicional += f"Soporte Total: {reporte['macro avg']['support']}"

        # Posicionar el texto adicional (ajustado para un solo gráfico)
        ax.text(1.2, 0.05, texto_adicional, transform=ax.transAxes, fontsize=8, 
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':8}, ha='left', va='bottom')
        ax.grid(False)



    def actualizar_estado(self, mensaje: str):
        """Actualiza el mensaje de estado y fuerza la actualización de la GUI."""
        self.label_estado.config(text=mensaje)
        self.root.update_idletasks()
        
    def inferir_y_true(self, rutas: List[str]) -> np.ndarray:
        """
        Intenta inferir las etiquetas verdaderas (Y-True) basándose
        en si la ruta del archivo contiene el nombre de la clase.
        Devuelve el índice numérico de la clase, o -1 si no se puede inferir.
        """
        y_true_inferida = []
        for ruta in rutas:
            # Normalizar y convertir a minúsculas la ruta para la búsqueda
            ruta_norm = os.path.basename(ruta).lower()
            
            # Asume el orden de CLASE_MAP: {'Bike': 0, 'Car': 1}
            if 'bike' in ruta_norm:
                y_true_inferida.append(CLASE_MAP['Bike']) # 0
            elif 'car' in ruta_norm:
                y_true_inferida.append(CLASE_MAP['Car']) # 1
            else:
                # Si no se puede inferir del nombre, asignamos -1
                y_true_inferida.append(-1) 
                
        return np.array(y_true_inferida)


def main():
    """Función principal para inicializar la GUI."""
    try:
        root = tk.Tk()
        root.lift()
        root.attributes('-topmost', True)
        root.focus_force()
        
        app = ClasificadorSimpleGUI(root)
        
        root.update_idletasks()
        width = root.winfo_width() if root.winfo_width() > 1 else 1100
        height = root.winfo_height() if root.winfo_height() > 1 else 650
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.after(1000, lambda: root.attributes('-topmost', False))
        root.mainloop()
        
    except Exception as e:
        print(f"Error en GUI: {e}")


if __name__ == "__main__":
    main()
