from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

# ---- Test Cases ----
# We want 3 correct predictions and 1 wrong one for demonstration
test_indices = []
correct_count, wrong_count = 0, 0

for i in range(len(X_test)):
    img = X_test[i].reshape(1, 28, 28, 1)
    expected = y_test[i]
    pred = np.argmax(model.predict(img, verbose=0))
    if pred == expected and correct_count < 3:
        test_indices.append(i)
        correct_count += 1
    elif pred != expected and wrong_count < 1:
        test_indices.append(i)
        wrong_count += 1
    if len(test_indices) == 4:
        break

# Print table
print("\n--- MNIST Test Cases ---")
print("Input Digit Image | Expected Label | Model Output | Correct (Y/N)")
for idx in test_indices:
    img = X_test[idx].reshape(1, 28, 28, 1)
    expected = y_test[idx]
    pred = np.argmax(model.predict(img, verbose=0))
    correct = "Y" if pred == expected else "N"
    print(f"Image of {expected}       | {expected}             | {pred}            | {correct}")
