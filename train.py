import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random 
import cv2
import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Input, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Set the path to the dataset
input_path = r'yourlocaion'
model_location = r'yourlocaion'

fig, ax = plt.subplots(2, 3, figsize=(15, 7))  
ax = ax.ravel()  
plt.tight_layout()
for i, _set in enumerate(['train', 'val', 'test']):
    set_path = os.path.join(input_path, _set)
    ax[i].imshow(plt.imread(os.path.join(set_path, 'NORMAL', os.listdir(os.path.join(set_path, 'NORMAL'))[0])), cmap='gray')
    ax[i].set_title(f'set{_set}, Condition: NORMAL')
    ax[i+3].imshow(plt.imread(os.path.join(set_path, 'PNEUMONIA', os.listdir(os.path.join(set_path, 'PNEUMONIA'))[0])), cmap='gray')
    ax[i+3].set_title(f'set{_set}, Condition: PNEUMONIA')
plt.show()

for _set in ['train', 'val', 'test']:
    print(os.path.join(input_path, _set))
    n_normal = len(os.listdir(os.path.join(input_path, _set, 'NORMAL')))
    n_pneumonia = len(os.listdir(os.path.join(input_path, _set, 'PNEUMONIA')))
    print(f'{_set}, normal images {n_normal}, pneumonia images {n_pneumonia}')


def process_data(img_dim, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(input_path, 'train'),
        target_size=(img_dim, img_dim),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    test_gen = test_val_datagen.flow_from_directory(
        directory=os.path.join(input_path, 'test'),
        target_size=(img_dim, img_dim),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    test_data = []
    test_labels = []
    for cond in ['NORMAL', 'PNEUMONIA']:
        for img in os.listdir(os.path.join(input_path, 'test', cond)):
            img_path = os.path.join(input_path, 'test', cond, img)
            img = plt.imread(img_path)
            img = cv2.resize(img, (img_dim, img_dim))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            label = 0 if cond == 'NORMAL' else 1
            test_data.append(img)
            test_labels.append(label)
        
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    return train_gen, test_gen, test_data, test_labels

# Set image dimensions, epochs, and batch size
img_dims = 150
epochs = 10
batch_size = 32


train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)


if os.path.exists(model_location):
    model = load_model(model_location)
    print("Loaded model from disk")
else:
    inputs = Input(shape=(img_dims, img_dims, 3))
    X = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    X = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = MaxPool2D(pool_size=(2, 2))(X)
    
    # Add more layers as needed

    X = Flatten()(X)
    X = Dense(units=1, activation='sigmoid')(X)  # Ensure only one unit in the output layer for binary classification

    # Create the model
    model = Model(inputs=inputs, outputs=X)

    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_location, save_best_only=True, save_weights_only=False)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min') 

# Train the model
history = model.fit_generator(
    train_gen, 
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=epochs, 
    validation_data=test_gen, 
    validation_steps=test_gen.samples // batch_size, 
    callbacks=[checkpoint, lr_reduce, early_stop]
)
