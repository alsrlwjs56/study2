
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Define a simple model
model = Sequential()
model.add(Dense(32, input_shape=(10,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate some random data
x = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=(100,))

# Train the model on the data
model.fit(x, y, epochs=10)

# Save the trained weights to a file
# model.save_weights('./model_weights.h5')

# Load the saved weights into a new model
new_model = Sequential()
new_model.add(Dense(32, input_shape=(10,), activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
new_model.load_weights('model_weights.h5')

# Use the new model to make predictions
new_model.predict(np.random.rand(1, 10))