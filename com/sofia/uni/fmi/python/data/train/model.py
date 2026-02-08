import json
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# 1. load the dataset and split it into training and validation sets

# training - 80% of the data
t_ds = tf.keras.utils.image_dataset_from_directory(
    "../Comprehensive_Disaster_Dataset(CDD)",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# validation - 20% of the data
v_ds = tf.keras.utils.image_dataset_from_directory(
    "../Comprehensive_Disaster_Dataset(CDD)",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = t_ds.class_names
class_count = len(class_names)
print("Classes:", class_names)

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)

# 2. use of caching and prefetching to improve performance during training
AUTOTUNE = tf.data.AUTOTUNE
train_ds_vis = t_ds

t_ds = t_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
v_ds = v_ds.cache().prefetch(AUTOTUNE)

# 3. data augmentation pipeline to help the model generalize better
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# 4. build a simple CNN model with a few convolutional layers, followed by global average pooling and dense layers
model = models.Sequential([
    data_aug,
    layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(class_count),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 5.usage of early stopping to prevent overfitting and restore the best weights based on validation  loss
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

# 6. train the model for a maximum of 20 epochs
# (it will stop early if the validation loss does not improve for 2 consecutive epochs)
history = model.fit(
    t_ds,
    epochs=20,
    validation_data=v_ds,
    callbacks=callbacks
)

# 7. plot the training and validation accuracy over epochs to visualize the model's performance
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

v_loss, v_acc = model.evaluate(v_ds, verbose=2)
print("Validation accuracy:", v_acc)

# 8. save the trained model for future use in the AI service
model.save("saved_model_disaster_classifier.keras")
