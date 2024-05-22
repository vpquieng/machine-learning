import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Set image dimensions and batch size
img_height, img_width = 32, 32
batch_size = 20

# Load the training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "MedBox/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "MedBox/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Fetch class names from the directory
class_names = train_ds.class_names

# Visualize some training images
plt.figure(figsize=(5,5))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Apply data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

# Normalize the pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Prepare the datasets with augmentation and normalization
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(normalization_layer(x), training=True), y)
)
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))  # Output layer with number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Evaluate the model
model.evaluate(val_ds)

# Make predictions and visualize the results
plt.figure(figsize=(10,10))
for images, labels in val_ds.take(1):
    predictions = model(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_label = np.argmax(predictions[i])
        true_label = labels[i]
        plt.title(f"Pred: {class_names[predicted_label]} | Real: {class_names[true_label]}")
        plt.axis("off")
plt.show()
