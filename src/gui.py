import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import pickle
from pathlib import Path
from caracteristicas import extract_hog, extract_lbp


class ClasificadorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Autos y Motos")
        self.root.geometry("800x600")
        
        self.imagen_actual = None
        self.ruta_imagen = None
        self.modelos = {}
        self.scaler = None
        
        self.configurar_interfaz()
        self.cargar_modelos()
    
    def configurar_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        titulo = ttk.Label(main_frame, text="Clasificador de Autos y Motos", 
                          font=('Arial', 16, 'bold'))
        titulo.pack(pady=(0, 20))
        
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Cargar Imagen", 
                  command=self.cargar_imagen).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(control_frame, text="Modelo:").pack(side=tk.LEFT, padx=(0, 5))
        self.combo_modelo = ttk.Combobox(control_frame, state="readonly", width=15)
        self.combo_modelo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Clasificar", 
                  command=self.clasificar_imagen).pack(side=tk.LEFT)
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        imagen_frame = ttk.LabelFrame(content_frame, text="Imagen", padding="10")
        imagen_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.label_imagen = ttk.Label(imagen_frame, text="No hay imagen cargada")
        self.label_imagen.pack(expand=True)
        
        resultado_frame = ttk.LabelFrame(content_frame, text="Resultados", padding="10")
        resultado_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.texto_resultado = tk.Text(resultado_frame, width=30, height=20, 
                                      state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(resultado_frame, orient="vertical", 
                                 command=self.texto_resultado.yview)
        self.texto_resultado.configure(yscrollcommand=scrollbar.set)
        
        self.texto_resultado.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.label_estado = ttk.Label(main_frame, text="Listo", relief=tk.SUNKEN)
        self.label_estado.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def cargar_modelos(self):
        """Carga los modelos disponibles"""
        modelos_disponibles = []
        
        try:
            if Path('../modelos/svm_hog.pkl').exists():
                with open('../modelos/svm_hog.pkl', 'rb') as f:
                    self.modelos['SVM-HOG'] = pickle.load(f)
                modelos_disponibles.append('SVM-HOG')
        except Exception as e:
            print(f"Error cargando SVM-HOG: {e}")
        
        try:
            if Path('../modelos/svm_lbp.pkl').exists():
                with open('../modelos/svm_lbp.pkl', 'rb') as f:
                    self.modelos['SVM-LBP'] = pickle.load(f)
                modelos_disponibles.append('SVM-LBP')
        except Exception as e:
            print(f"Error cargando SVM-LBP: {e}")
        
        try:
            if Path('../modelos/nn_hog.keras').exists():
                from tensorflow.keras.models import load_model
                self.modelos['NN-HOG'] = load_model('../modelos/nn_hog.keras')
                
                if Path('../modelos/scaler_nn_hog.pkl').exists():
                    with open('../modelos/scaler_nn_hog.pkl', 'rb') as f:
                        self.scaler = pickle.load(f)
                    modelos_disponibles.append('NN-HOG')
        except Exception as e:
            print(f"Error cargando NN-HOG: {e}")
        
        self.combo_modelo['values'] = modelos_disponibles
        if modelos_disponibles:
            self.combo_modelo.current(0)
            self.actualizar_estado(f"Modelos cargados: {len(modelos_disponibles)}")
        else:
            self.actualizar_estado("No se encontraron modelos entrenados")
    
    def cargar_imagen(self):
        """Carga una imagen desde archivo"""
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
                self.actualizar_estado(f"Imagen cargada: {Path(self.ruta_imagen).name}")
                self.limpiar_resultados()
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
    
    def mostrar_imagen(self, ruta_imagen):
        """Muestra una imagen en la interfaz"""
        imagen_pil = Image.open(ruta_imagen)
        imagen_pil.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        self.imagen_tk = ImageTk.PhotoImage(imagen_pil)
        self.label_imagen.configure(image=self.imagen_tk, text="")
    
    def clasificar_imagen(self):
        """Clasifica la imagen actual"""
        if not self.ruta_imagen:
            messagebox.showwarning("Advertencia", "Primero debe cargar una imagen")
            return
        
        modelo_seleccionado = self.combo_modelo.get()
        if not modelo_seleccionado:
            messagebox.showwarning("Advertencia", "Debe seleccionar un modelo")
            return
        
        try:
            self.actualizar_estado("Clasificando imagen...")
            
            img = cv2.imread(self.ruta_imagen, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            
            if 'HOG' in modelo_seleccionado:
                caracteristicas = extract_hog([img])[0]
            elif 'LBP' in modelo_seleccionado:
                caracteristicas = extract_lbp([img])[0]
            
            modelo = self.modelos[modelo_seleccionado]
            
            if modelo_seleccionado == 'NN-HOG':
                caracteristicas_scaled = self.scaler.transform([caracteristicas])
                prediccion_prob = modelo.predict(caracteristicas_scaled)[0]
                prediccion = np.argmax(prediccion_prob)
                confianza = np.max(prediccion_prob)
            else:
                prediccion = modelo.predict([caracteristicas])[0]
                try:
                    probabilidades = modelo.predict_proba([caracteristicas])[0]
                    confianza = np.max(probabilidades)
                except:
                    confianza = None
            
            nombres_clases = ['Bike', 'Car']
            self.mostrar_resultados(modelo_seleccionado, prediccion, confianza, nombres_clases)
            self.actualizar_estado("Clasificación completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en clasificación: {str(e)}")
            self.actualizar_estado("Error en clasificación")
    
    def mostrar_resultados(self, modelo, prediccion, confianza, nombres_clases):
        """Muestra los resultados de la clasificación"""
        resultado = nombres_clases[prediccion]
        
        texto = f"RESULTADOS DE CLASIFICACIÓN\n"
        texto += f"{'=' * 30}\n\n"
        texto += f"Modelo usado: {modelo}\n"
        texto += f"Imagen: {Path(self.ruta_imagen).name}\n\n"
        texto += f"PREDICCIÓN: {resultado}\n"
        
        if confianza is not None:
            texto += f"Confianza: {confianza:.2%}\n\n"
            
            if confianza > 0.8:
                nivel_confianza = "Alta"
            elif confianza > 0.6:
                nivel_confianza = "Media"
            else:
                nivel_confianza = "Baja"
            texto += f"Nivel de confianza: {nivel_confianza}\n"
        
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        self.texto_resultado.insert(1.0, texto)
        self.texto_resultado.config(state=tk.DISABLED)
    
    def limpiar_resultados(self):
        """Limpia los resultados mostrados"""
        self.texto_resultado.config(state=tk.NORMAL)
        self.texto_resultado.delete(1.0, tk.END)
        self.texto_resultado.config(state=tk.DISABLED)
    
    def actualizar_estado(self, mensaje):
        """Actualiza el mensaje de estado"""
        self.label_estado.config(text=mensaje)
        self.root.update_idletasks()


def main():
    """Función principal para ejecutar la GUI"""
    root = tk.Tk()
    app = ClasificadorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
