from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from flask import Flask, request, jsonify
import joblib

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# Evaluate the model
accuracy = knn.score(X_test, y_test)
print('Test set accuracy: {:.2f}'.format(accuracy))
# Make predictions on unseen data
X_new = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6]]
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
scores = cross_val_score(knn, X, y, cv=5)
training_score = knn.score(X_train, y_train)

# Evaluate the model on the test set
test_score = knn.score(X_test, y_test)

print('Training score: {:.2f}'.format(training_score))
print('Test score: {:.2f}'.format(test_score))

if training_score > test_score:
    print('Model is overfitting')
else:
    print('Model is not overfitting')

joblib.dump(knn, 'knn_model.pkl')
app = Flask(__name__)
if __name__ == "__main__":
    app.run(host="0.0.0.0")

# Load the model from the file
    model = joblib.load('knn_model.pkl')

# Load the Iris dataset
    iris = datasets.load_iris()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Make prediction using the model
    prediction = model.predict(data)

    # Get the names of the iris species
    species = [iris.target_names[i] for i in prediction]

    # Return the prediction as a response
    return jsonify(species)