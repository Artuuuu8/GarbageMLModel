import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(val_class_dir):
                os.makedirs(val_class_dir)

            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            random.shuffle(files)
            split_index = int(len(files) * split_ratio)
            train_files = files[:split_index]
            val_files = files[split_index:]

            for file in train_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
            for file in val_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(val_class_dir, file))

            print(f"Copied {len(train_files)} files to {train_class_dir}")
            print(f"Copied {len(val_files)} files to {val_class_dir}")

# Download latest version of the dataset
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")

print("Path to dataset files:", path)

# Define ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Define paths
source_dir = os.path.join(path, 'Garbage classification')
train_dir = os.path.join(path, 'train')
val_dir = os.path.join(path, 'val')

# Split the dataset
split_dataset(source_dir, train_dir, val_dir)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='categorical'  # Multiple classes
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Build the CNN model
model = models.Sequential()

# First Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the results to feed into a fully connected layer
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(128, activation='relu'))

# Output layer for classification
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,  # You can adjust the number of epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32
)

# Save the trained model
model.save('garbage_classification_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator, steps=val_generator.samples // 32)
print(f"Test accuracy: {test_acc * 100:.2f}%")