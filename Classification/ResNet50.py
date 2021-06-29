from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
target_s = (224, 224)

train_flow = image_gen.flow_from_directory(
    '../data/low/split/train/',
    class_mode='categorical',
    target_size=target_s,
    batch_size=32,
    shuffle=True
)

val_flow = image_gen.flow_from_directory(
    '../data/low/split/val/',
    class_mode='categorical',
    target_size=target_s,
    batch_size=32,
    shuffle=True,
)


tb = TensorBoard(
    log_dir='/content/drive/Shareddrives/deep learning/transfer_resnet_tb/', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

inputs = Input(shape=(224, 224, 3), name='Input_layer')
res_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

for layer in res_model.layers[:143]:
    layer.trainable = False

model = Sequential()
model.add(res_model)
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(130, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

checkpoint = ModelCheckpoint("resnet50_check_.h5",
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

early = EarlyStopping(monitor='val_accuracy',
                      min_delta=0,
                      patience=3,
                      verbose=1,
                      mode='auto')

model.fit_generator(
    generator=train_flow,
    epochs=20,
    validation_data=val_flow,
    callbacks=[checkpoint, early, tb])

model.save_weights('resnet50_final_.h5')



