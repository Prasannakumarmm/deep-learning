from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np

# Class names for Fashion-MNIST
class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

# ---- Test Cases ----
# Predefined indices for demonstration (adjust if you want specific matches/mismatches)
test_indices = [0, 1, 2, 3]

print("\n--- Fashion-MNIST Test Cases ---")
print("Input Image | True Label | Predicted Label | Correct (Y/N)")

for idx in test_indices:
    img = X_test[idx].reshape(1, 28, 28, 1)
    true_label = class_names[y_test[idx]]
    pred_label = class_names[np.argmax(model.predict(img, verbose=0))]
    correct = "Y" if true_label == pred_label else "N"
    print(f"{true_label:<11} | {true_label:<10} | {pred_label:<14} | {correct}")
