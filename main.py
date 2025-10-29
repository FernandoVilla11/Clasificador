import sys
import os
import tkinter as tk
import cv2, numpy as np, pickle
from PIL import Image, ImageTk
from skimage.feature import hog, local_binary_pattern
from gui import ClasificadorSimpleGUI

def main():
    print("Iniciando Clasificador de Autos y Motos")
    
    modelos = ['modelos/nn_hog.keras', 'modelos/svm_hog.pkl', 'modelos/svm_lbp.pkl']

    root = tk.Tk()
    
    app = ClasificadorSimpleGUI(root, modelos)
    
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'{w}x{h}+{x}+{y}')
    
    root.mainloop()
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Clasificador de Autos y Motos")
        print("Uso: python main.py  # Abrir interfaz gráfica")
    else:
        main()