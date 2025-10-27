import sys
import os
import tkinter as tk

def main():
    print("Iniciando Clasificador de Autos y Motos")
    try:
        if not os.path.exists('gui.py'):
            print("Error: Archivo gui.py no encontrado")
            return
        
        import cv2, numpy as np, pickle
        from PIL import Image, ImageTk
        from skimage.feature import hog, local_binary_pattern
        
        modelos = 0
        for modelo in ['modelos/nn_hog.keras', 'modelos/svm_hog.pkl', 'modelos/svm_lbp.pkl']:
            if os.path.exists(modelo):
                modelos += 1
        
        print(f" {modelos} modelos disponibles")
        
        from gui import ClasificadorSimpleGUI
        
        root = tk.Tk()
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(lambda: root.attributes('-topmost', False))
        
        app = ClasificadorSimpleGUI(root)
        
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (w // 2)
        y = (root.winfo_screenheight() // 2) - (h // 2)
        root.geometry(f'{w}x{h}+{x}+{y}')
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Dependencia faltante: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de tener todos los modelos entrenados")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Clasificador de Autos y Motos")
        print("Uso: python main.py  # Abrir interfaz gráfica")
    else:
        main()