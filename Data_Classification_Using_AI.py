from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


print("=" * 60)
print("       PROJECT 2 — DATA CLASSIFICATION USING AI")
print("       Iris Flower Classification using KNN")
print("=" * 60)

print("\nDataset Information")
print("-" * 60)

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of classes:", len(target_names))

print("\nFeatures:")
for feature in feature_names:
    print("-", feature)

print("\nClasses:")
for target in target_names:
    print("-", target)

print("\nFirst 5 Samples")
print("-" * 60)

for i in range(5):
    print(
        X[i],
        "->",
        target_names[y[i]]
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData Splitting")
print("-" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling")
print("-" * 60)
print("StandardScaler applied successfully.")

k = 5

model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train_scaled, y_train)

print("\nModel Training")
print("-" * 60)
print("Algorithm: K-Nearest Neighbors")
print("Value of K:", k)
print("Model trained successfully.")

y_pred = model.predict(X_test_scaled)

print("\nPrediction")
print("-" * 60)
print("Predictions generated successfully.")


accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy")
print("-" * 60)
print(f"Accuracy: {accuracy * 100:.2f}%")

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nF1 Score")
print("-" * 60)
print(f"F1 Score: {f1:.2f}")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print("-" * 60)

print(cm)

print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)

print("\nActual vs Predicted")
print("-" * 60)

for actual, predicted in zip(y_test, y_pred):
    print(
        "Actual:",
        target_names[actual],
        "| Predicted:",
        target_names[predicted]
    )


print("\nNew Flower Prediction")
print("-" * 60)

new_flower = [[
    5.1,   # sepal length
    3.5,   # sepal width
    1.4,   # petal length
    0.2    # petal width
]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

predicted_class = target_names[prediction[0]]

print("Input measurements:", new_flower[0])
print("Predicted flower:", predicted_class)

print("\n" + "=" * 60)
print("             PROJECT COMPLETED")
print("=" * 60)