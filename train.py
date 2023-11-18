import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar datos
#file_path = r'D:\Taller\CML\Pima-Indians-Diabetes-Dataset\data\diabetes.csv'
file_path = 'data/diabetes.csv'
data = pd.read_csv(file_path)

# Separar características y etiqueta
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir y evaluar el modelo
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Guardar métricas en un archivo de texto
with open('metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write('\n')
    f.write(classification_report(y_test, predictions))

# Generar y guardar la matriz de confusión
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('plot.png')

# Guardar el modelo
joblib.dump(model, 'diabetes_model.pkl')

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy: {scores.mean()}")