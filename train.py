
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")

data = load_iris()
X = data.data
y = data.target

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

#prediction
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

#save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
