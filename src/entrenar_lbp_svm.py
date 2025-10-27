from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump
from preprocesamiento import load_and_preprocess
from caracteristicas import extract_lbp

print("Cargando dataset...")
X, y, classes = load_and_preprocess('../datos/imagenes')

print("Extrayendo características LBP...")
X_lbp = extract_lbp(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_lbp, y, test_size=0.2, random_state=42
)
print("Entrenando SVM con LBP...")
svm_lbp = SVC(kernel='linear', probability=True)
svm_lbp.fit(X_train, y_train)
y_pred = svm_lbp.predict(X_test)
print("Resultados del modelo (LBP + SVM):")
print(classification_report(y_test, y_pred, target_names=classes))
dump(svm_lbp, '../modelos/svm_lbp.pkl')
print("Modelo SVM (LBP) guardado en ../modelos/svm_lbp.pkl")
