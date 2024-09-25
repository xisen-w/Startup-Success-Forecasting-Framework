from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Assuming we are already in the directory
DATA_DIRPATH_SUC = os.path.join(os.getcwd(), 'data/successful')
DATA_DIRPATH_UN = os.path.join(os.getcwd(), 'data/unsuccessful')

# Assuming X_train, X_test are prepared as above

successful_founder_df = pd.read_csv(os.path.join(DATA_DIRPATH_SUC, 'segmented_successful_profiles.csv'))
unsuccessful_founder_df = pd.read_csv(os.path.join(DATA_DIRPATH_UN, 'segmented_unsuccessful_profiles.csv'))

successful_founder_df = successful_founder_df[:100]

# Starting with successful_founder_df
# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(successful_founder_df['paragraph'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(successful_founder_df['segment'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Build neural network model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(y_train_onehot.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train.toarray(), y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
_, accuracy = model.evaluate(X_test.toarray(), y_test_onehot)
print("Neural Network Accuracy:", accuracy)
