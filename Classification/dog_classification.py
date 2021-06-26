from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras import optimizers

####################################
###### Image Generator #############

image_gen = ImageDataGenerator(rescale=1 / 255)
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

####################################
###### Transfer Learning #############

from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

####################################
###### Top Layers #############

flat1 = Flatten()(base_model.output)
class2 = Dense(4096, activation='relu', name='fc2')(flat1)
y = Dropout(0.2)(class2)
class3 = Dense(4096, activation='relu', name='fc3')(y)
z = Dropout(0.2)(class3)
output = Dense(130, activation='softmax', name='predictions')(z)

####################################
###### Final Model #############

model_final = Model(inputs=base_model.input, outputs=output)
model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                    metrics=["accuracy"])

checkpoint = ModelCheckpoint("inception_224_checkpoint.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

model_final.fit_generator(
    generator=train_flow,
    epochs=20,
    validation_data=val_flow,
    callbacks=[checkpoint, early, tb])

model_final.save_weights("inception_224_final.h5")
