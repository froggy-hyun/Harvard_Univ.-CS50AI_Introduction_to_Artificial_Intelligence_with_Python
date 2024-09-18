import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Create a neural network
# Keras is an api that different machine learning algorithms access. 
# A sequential model is one where layers follow each other
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation
# A dense layer is one where each node in the current layer is connected to all the nodes from the previous layer. 
# In generating our hidden layers we create 8 dense layers, each having 4 input neurons, using the ReLU activation function mentioned above.
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

# Add output layer with 1 unit, with sigmoid activation where the output is a value between 0 and 1.
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Train neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_training, y_training, epochs=20) # We fit the model on the training data with 20 repetitions (epochs)

# Evaluate how well model performs
model.evaluate(X_testing, y_testing, verbose=2)
