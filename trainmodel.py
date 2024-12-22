from tensorflow.keras.preprocessing.sequence import pad_sequences
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import os

# Ensure Logs directory exists
log_dir = os.path.join('Logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load sequences
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                
                # Check if the frame has the expected shape (63,)
                if res.shape == (63,):
                    window.append(res)
                else:
                    print(f"Skipping invalid frame {frame_num} for sequence {sequence} in action {action}")
            except Exception as e:
                print(f"Error loading frame {frame_num} for sequence {sequence} in action {action}: {e}")
        
        # Only append valid sequences (those that have 30 frames of shape (63,))
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Skipping incomplete sequence {sequence} for action {action}")

# Pad sequences to ensure they all have the same length (if not already)
print("Shapes of sequences before padding:")
for seq in sequences:
    print(np.shape(seq))

X = pad_sequences(sequences, padding='post', dtype='float32', maxlen=sequence_length)

# Convert labels to one-hot encoding
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

tb_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # Adjust input shape as needed
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Ensure correct number of output classes

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, early_stopping], validation_data=(X_test, y_test))

# Print model summary
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')





# from function import *
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import TensorBoard
# label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
# sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
# res = [.7, 0.2, 0.1]

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
# model.summary()

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('model.h5')