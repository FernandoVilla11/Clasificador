import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import pickle
import os
from skimage.feature import hog, local_binary_pattern
from joblib import load as joblib_load


def extraer_lbp_mejorado(imagen):
    caracteristicas = []
    
    configuraciones = [
        {'P': 8, 'R': 1, 'method': 'uniform'},
        {'P': 16, 'R': 2, 'method': 'uniform'}, 
        {'P': 24, 'R': 3, 'method': 'uniform'}
    ]
    
    for config in configuraciones:
        lbp = local_binary_pattern(imagen, P=config['P'], R=config['R'], method=config['method'])
        n_bins = config['P'] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
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


class ClasificadorSimpleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Autos y Motos")
        self.root.geometry("900x600")
        
        self.imagen_actual = None
        self.ruta_imagen = None
        self.modelo_cargado = None
        self.tipo_modelo = None
        self.scaler = None
        
        self.configurar_interfaz()
        self.detectar_modelos()
    
    def configurar_interfaz(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        titulo = tk.Label(main_frame, text="Clasificador de Autos y Motos", 
                         font=('Arial', 16, 'bold'))
        titulo.pack(pady=(0, 20))
        
        control_frame = tk.LabelFrame(main_frame, text="Controles", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        fila1 = tk.Frame(control_frame)
        fila1.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(fila1, text="Modelo:").pack(side=tk.LEFT)
        self.combo_modelo = ttk.Combobox(fila1, state="readonly", width=15)
        self.combo_modelo.pack(side=tk.LEFT, padx=(5, 10))
        
        tk.Button(fila1, text="Cargar Modelo", command=self.cargar_modelo,
                 bg='lightblue').pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(fila1, text="Cargar Imagen", command=self.cargar_imagen,
                 bg='lightgreen').pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(fila1, text="Clasificar", command=self.clasificar_imagen,
                 bg='orange', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.LabelFrame(content_frame, text="Imagen", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.label_imagen = tk.Label(left_frame, text="No hay imagen cargada", 
                                   bg='white', relief=tk.SUNKEN, bd=2)
        self.label_imagen.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = tk.LabelFrame(content_frame, text="Resultados", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        self.texto_resultado = tk.Text(right_frame, width=40, height=25, 
                                      state=tk.DISABLED, wrap=tk.WORD,
                                      font=('Consolas', 9))
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", 
                               command=self.texto_resultado.yview)
        self.texto_resultado.configure(yscrollcommand=scrollbar.set)
        
        self.texto_resultado.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.label_estado = tk.Label(main_frame, text="Listo", relief=tk.SUNKEN, 
                                   bd=1, anchor='w')
        self.label_estado.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        self.mostrar_mensaje_inicial()
    
    def detectar_modelos(self):
        modelos_disponibles = []
        
        if os.path.exists('modelos/nn_hog.keras'):
            modelos_disponibles.append('NN-HOG')
        if os.path.exists('modelos/svm_hog.pkl'):
            modelos_disponibles.append('SVM-HOG')
        if os.path.exists('modelos/svm_lbp.pkl'):
            modelos_disponibles.append('SVM-LBP')
        elif os.path.exists('modelos/svm_lbp.pkl'):
            modelos_disponibles.append('SVM-LBP')
        
        self.combo_modelo['values'] = modelos_disponibles
        if modelos_disponibles:
            self.combo_modelo.current(0) 
            self.actualizar_estado(f"Detectados {len(modelos_disponibles)} modelos")
        else:
            self.actualizar_estado("No se encontraron modelos")
    
    def mostrar_mensaje_inicial(self):
        mensaje = """
CLASIFICADOR DE AUTOS Y MOTOS

Sistema de clasificación.

RENDIMIENTO DE MODELOS:
• Red Neuronal + HOG: 94.13%
• SVM + HOG: 91.75%  
• SVM + LBP: 73.62%

INSTRUCCIONES:
1. Seleccione un modelo
2. Haga clic en 'Cargar Modelo'
3. Cargue una imagen
4. Presione 'Clasificar'

        """
        
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        self.texto_resultado.insert(1.0, mensaje.strip())
        self.texto_resultado.config(state=tk.DISABLED)
    
    def cargar_modelo(self):
        modelo_seleccionado = self.combo_modelo.get()
        if not modelo_seleccionado:
            messagebox.showwarning("Advertencia", "Seleccione un modelo primero")
            return
        
        try:
            self.actualizar_estado("Cargando modelo...")
            
            if modelo_seleccionado == 'SVM-HOG':
                if not os.path.exists('modelos/svm_hog.pkl'):
                    raise FileNotFoundError("Archivo svm_hog.pkl no encontrado")
                self.modelo_cargado = joblib_load('modelos/svm_hog.pkl')
                self.tipo_modelo = 'svm-hog'
                    
            elif self.combo_modelo.get() == 'SVM-LBP':
                if not os.path.exists('modelos/svm_lbp.pkl'):
                    raise FileNotFoundError("Archivo svm_lbp.pkl no encontrado")
                if not os.path.exists('modelos/scaler_lbp.pkl'):
                    raise FileNotFoundError("Archivo scaler_lbp.pkl no encontrado")
                
                self.modelo_cargado = joblib_load('modelos/svm_lbp.pkl')
                self.scaler_lbp = joblib_load('modelos/scaler_lbp.pkl')
                self.tipo_modelo = 'svm-lbp-mejorado'
                
            elif modelo_seleccionado == 'SVM-LBP':
                if not os.path.exists('modelos/svm_lbp.pkl'):
                    raise FileNotFoundError("Archivo svm_lbp.pkl no encontrado")
                
                respuesta = messagebox.askyesno("Información", 
                    "SVM-LBP tiene precisión del 84.12%.\n\n"
                    "Se recomienda usar NN-HOG (94.13%) para máxima precisión.\n\n"
                    "¿Continuar con SVM-LBP?")
                
                if not respuesta:
                    self.actualizar_estado("Carga de modelo cancelada")
                    return
                
                self.modelo_cargado = joblib_load('modelos/svm_lbp.pkl')
                self.tipo_modelo = 'svm-lbp'
                    
            elif modelo_seleccionado == 'NN-HOG':
                if not os.path.exists('modelos/nn_hog.keras'):
                    raise FileNotFoundError("Archivo nn_hog.keras no encontrado")
                if not os.path.exists('modelos/scaler_nn_hog.pkl'):
                    raise FileNotFoundError("Archivo scaler_nn_hog.pkl no encontrado")
                
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                import warnings
                warnings.filterwarnings('ignore')
                
                from tensorflow.keras.models import load_model
                self.modelo_cargado = load_model('modelos/nn_hog.keras')
                self.tipo_modelo = 'nn-hog'
                
                with open('modelos/scaler_nn_hog.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.actualizar_estado(f"Modelo {modelo_seleccionado} cargado correctamente")
            messagebox.showinfo("Éxito", f"Modelo {modelo_seleccionado} cargado exitosamente")
            
        except FileNotFoundError as e:
            error_msg = f"Archivo no encontrado: {str(e)}\nAsegúrate de entrenar los modelos primero."
            self.actualizar_estado("Error: Modelo no encontrado")
            messagebox.showerror("Error", error_msg)
            
        except Exception as e:
            error_msg = f"Error cargando modelo: {str(e)}"
            self.actualizar_estado(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def cargar_imagen(self):
        tipos_archivo = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos", "*.*")
        ]
        
        self.ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=tipos_archivo
        )
        
        if self.ruta_imagen:
            try:
                self.mostrar_imagen(self.ruta_imagen)
                nombre_archivo = os.path.basename(self.ruta_imagen)
                self.actualizar_estado(f"Imagen cargada: {nombre_archivo}")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
    
    def mostrar_imagen(self, ruta_imagen):
        imagen_pil = Image.open(ruta_imagen)
        imagen_pil.thumbnail((350, 350), Image.Resampling.LANCZOS)
        
        self.imagen_tk = ImageTk.PhotoImage(imagen_pil)
        self.label_imagen.configure(image=self.imagen_tk, text="")
    
    def clasificar_imagen(self):
        if not self.ruta_imagen:
            messagebox.showwarning("Advertencia", "Cargue una imagen primero")
            return
        
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Cargue un modelo primero")
            return
        
        try:
            self.actualizar_estado("Clasificando...")
            
            img = cv2.imread(self.ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("No se pudo cargar la imagen")
                
            img = cv2.resize(img, (128, 128))
            
            if 'hog' in self.tipo_modelo:
                caracteristicas = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2))
            elif self.tipo_modelo == 'svm-lbp-mejorado':
                caracteristicas = extraer_lbp_mejorado(img)
            elif 'lbp' in self.tipo_modelo:
                lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59))
                caracteristicas = hist / np.sum(hist)
            
            if self.tipo_modelo == 'nn-hog':
                caracteristicas_scaled = self.scaler.transform([caracteristicas])
                prediccion_prob = self.modelo_cargado.predict(caracteristicas_scaled, verbose=0)[0]
                prediccion = np.argmax(prediccion_prob)
                probabilidades = prediccion_prob
            elif self.tipo_modelo == 'svm-lbp-mejorado':
                caracteristicas_scaled = self.scaler_lbp.transform([caracteristicas])
                prediccion = self.modelo_cargado.predict(caracteristicas_scaled)[0]
                probabilidades = self.modelo_cargado.predict_proba(caracteristicas_scaled)[0]
            else:
                prediccion = self.modelo_cargado.predict([caracteristicas])[0]
                try:
                    probabilidades = self.modelo_cargado.predict_proba([caracteristicas])[0]
                except:
                    probabilidades = None
            
            self.mostrar_resultados(prediccion, probabilidades)
            self.actualizar_estado("Clasificación completada")
            
        except Exception as e:
            error_msg = f"Error en clasificación: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.actualizar_estado(error_msg)
    
    def mostrar_resultados(self, prediccion, probabilidades):
        nombres_clases = ['Bike', 'Car']
        resultado = nombres_clases[prediccion]
        nombre_archivo = os.path.basename(self.ruta_imagen)
        
        if probabilidades is not None:
            confianza = np.max(probabilidades) * 100
        else:
            confianza = 0
        
        texto = "RESULTADO DE CLASIFICACION:\n"
        texto += "=" * 35 + "\n\n"
        
        if resultado == 'Car':
            texto += f"Vehiculo predicho: AUTO\n"
            icono_resultado = ""
        else:
            texto += f"Vehiculo predicho: MOTO\n"
            icono_resultado = ""
            
        if confianza > 0:
            texto += f"Confianza: {confianza:.2f}%\n\n"
        
        if probabilidades is not None:
            texto += "PROBABILIDADES POR CLASE:\n"
            texto += "-" * 30 + "\n"
            
            for i, clase in enumerate(nombres_clases):
                prob = probabilidades[i] * 100
                icono = "🚲" if clase == "Bike" else "🚗"
                
                barra_len = int(prob / 5)
                barra = "█" * barra_len + "░" * (20 - barra_len)
                
                texto += f"{icono} {clase:4}: {barra} {prob:.1f}%\n"
        
        texto += f"Archivo: {nombre_archivo}\n"
        
        modelo_usado = self.combo_modelo.get()
        if modelo_usado == 'NN-HOG':
            texto += f"Modelo: {modelo_usado} (Precisión: 94.13%)\n"
        elif modelo_usado == 'SVM-HOG':
            texto += f"Modelo: {modelo_usado} (Precisión: 91.75%)\n"
        elif modelo_usado == 'SVM-LBP':
            texto += f"Modelo: {modelo_usado} (Precisión: 84.12%)\n"
        else:
            texto += f"Modelo: {modelo_usado}\n"
        
        texto += f"\nINTERPRETACION:\n"
        texto += "-" * 20 + "\n"
        
        if confianza >= 90:
            texto += "Clasificacion MUY CONFIABLE\n"
        elif confianza >= 75:
            texto += "Clasificacion CONFIABLE\n" 
        elif confianza >= 60:
            texto += "Clasificacion MODERADA\n"
        else:
            texto += "Clasificacion INCIERTA\n"
        
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        self.texto_resultado.insert(1.0, texto)
        self.texto_resultado.config(state=tk.DISABLED)
    
    def actualizar_estado(self, mensaje):
        """Actualiza el mensaje de estado"""
        self.label_estado.config(text=mensaje)
        self.root.update_idletasks()


def main():
    """Función principal - Fuerza la apertura de la ventana"""
    try:
        print("Creando ventana gráfica...")
        root = tk.Tk()
        
        root.lift()
        root.attributes('-topmost', True)
        root.focus_force()
        
        app = ClasificadorSimpleGUI(root)
        
        root.update_idletasks()
        width = root.winfo_width() if root.winfo_width() > 1 else 900
        height = root.winfo_height() if root.winfo_height() > 1 else 600
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.after(1000, lambda: root.attributes('-topmost', False))
        print("Ventana gráfica lista. Iniciando bucle principal...")
        root.mainloop()
        
    except Exception as e:
        print(f"Error en GUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()