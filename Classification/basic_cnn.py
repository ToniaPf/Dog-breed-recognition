from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

image_gen = ImageDataGenerator(rescale=1/255)

train_flow = image_gen.flow_from_directory(
    '../data/low/split/train/',
    class_mode='categorical',
    target_size=(320, 320),
)

val_flow = image_gen.flow_from_directory(
    '../data/low/split/val/',
    class_mode='categorical',
    target_size=(320, 320),
)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(130))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb = TensorBoard(
    log_dir='/content/drive/Shareddrives/deep learning/simple-cnn-tb-logs/',
    histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

early_stop = EarlyStopping(monitor='val_loss', patience=2)

results = model.fit_generator(train_flow, epochs=20,
                              validation_data=val_flow,
                              callbacks=[early_stop, tb])