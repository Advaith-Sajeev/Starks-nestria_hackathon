import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Assuming historical_predictions is a 2D array where each row is a video, and each column is the probability from a model
# Assuming historical_labels is a 1D array where each element is the true label (0 or 1) for the corresponding video

# Normalize the input data
historical_predictions_normalized = np.array([
    model1_predictions,
    model2_predictions,
    model3_predictions,
    model4_predictions,
    model5_predictions,
    model6_predictions,
    model7_predictions,
    model8_predictions,
    model9_predictions,
    model10_predictions
]).T  # Transpose to have each row represent a sample and each column represent a feature

# Assuming historical_labels are your actual labels
historical_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Replace with your actual labels

# Build a neural network model
model = Sequential()
model.add(Dense(units=1, input_dim=len(historical_predictions_normalized[0]), activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(historical_predictions_normalized, historical_labels, epochs=100, verbose=1)

# Get the learned weights
learned_weights = model.get_weights()[0]

# Normalize the learned weights to sum to 1
normalized_weights = learned_weights / np.sum(learned_weights)

# Print the learned weights
print("Learned Weights:", normalized_weights)

# Use the learned weights to make predictions
final_decision = np.average(predictions, axis=0, weights=normalized_weights) > 0.5

# Print or use the final_decision array as needed
print("Final Decision:", final_decision.astype(int))
