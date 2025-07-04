import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# image path
data_dir = "Training"  # Path to your extracted folder

# image size & training parameters
img_height, img_width = 299, 299
batch_size = 32
epochs = 25  

# preprocess data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# cnn model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.9),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # output layer for classification
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# model training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# evaluation of result
# accuracy and loss plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# classification report
# generate predictions on validation set
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("Classification Report:")
print(classification_report(val_generator.classes, y_pred, target_names=list(train_generator.class_indices.keys())))

# confusion matrix
cm = confusion_matrix(val_generator.classes, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_generator.class_indices.keys(),
            yticklabels=train_generator.class_indices.keys(), cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# new image prediction function
def predict_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    class_labels = list(train_generator.class_indices.keys())
    predicted_label = class_labels[class_index]
    confidence = prediction[0][class_index]

    print(f"Predicted class: {predicted_label} ({confidence:.2f} confidence)")
    return predicted_label

# prediction
predict_image("Testing\\b6.jpg")
predict_image("Testing\\b8.jpg")
predict_image("Testing\\p3.jpg")
predict_image("Testing\\pd12.jpg")
predict_image("Testing\\c6.jpg")
