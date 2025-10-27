from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump
from preprocesamiento import load_and_preprocess
from caracteristicas import extract_hog

print("Cargando dataset...")
X, y, classes = load_and_preprocess('../datos/imagenes')

print("Extrayendo características HOG...")
X_hog = extract_hog(X)

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

print("Entrenando SVM con HOG...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("🔹 Resultados del modelo:")
print(classification_report(y_test, y_pred, target_names=classes))

dump(svm, '../modelos/svm_hog.pkl')
print(" Modelo SVM (HOG) guardado en ../modelos/svm_hog.pkl")
