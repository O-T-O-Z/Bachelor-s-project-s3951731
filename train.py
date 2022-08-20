import os
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.models import Model
import random
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

device_name = tf.test.gpu_device_name()
print(device_name)
train_root = "path/to/train/"
valid_root = "path/to/validation/"
generator = ImageDataGenerator()
train_gen = generator.flow_from_directory(train_root, target_size=(256, 256), batch_size=1, class_mode='categorical',
                                          seed=42)
valid_gen = generator.flow_from_directory(valid_root, target_size=(256, 256), batch_size=1, class_mode='categorical',
                                          seed=42)


class MultipleInputGenerator(Sequence):
    """
    Wrapper around ImageDataGenerators to return two samples and one desired output.
    """

    def __init__(self, subset):
        """
        Initializes the input generator, the class names and the data root.
        :param subset: subset to choose from, either training or validation.
        """
        self.genX1 = train_gen if subset == "training" else valid_gen
        self.class_names = os.listdir(train_root) if subset == "training" else os.listdir(valid_root)
        self.root = train_root if subset == "training" else valid_root

    def __len__(self):
        """
        Returns the length of the generator.
        :return: length of the generator
        """
        return self.genX1.__len__()

    def __getitem__(self, index):
        """
        Combining one input from a ImageDataGenerator with either
        a random other class' sample or a random new sample from the same class.
        The desired output is also modified to reflect this.

        :param index: index of the item to retrieve.
        :return: two inputs and the desired output.
        """
        X1_batch, Y1_batch = self.genX1.__getitem__(index)
        choice = random.choice(["same", "diff"])
        class_folder = self.class_names[np.where(Y1_batch[0] == 1)[0][0]]
        if choice == "same":
            X2_batch = self.get_class_sample(class_folder)
            # check that same sample is not selected
            while np.array_equal(X2_batch, X1_batch):
                X2_batch = self.get_class_sample(class_folder)
            Y_batch = np.array([1])
        else:
            possible_classes = [c for c in self.class_names if c != class_folder]
            class_folder = random.choice(possible_classes)
            X2_batch = self.get_class_sample(class_folder)
            Y_batch = np.array([0])
        X_batch = (np.array(X1_batch), np.array(X2_batch))
        return X_batch, Y_batch

    def get_class_sample(self, class_folder):
        """
        Returns a sample from a specific class as an array.
        :param class_folder: folder where the class is located.
        :return: return numpy array of the image sample.
        """
        class_folder = os.path.join(self.root, class_folder)
        class_sample = os.path.join(class_folder, random.choice(os.listdir(class_folder)))
        img = load_img(class_sample)
        img_array = img_to_array(img)
        return np.array([img_array])


train_generator = MultipleInputGenerator("training")
validation_generator = MultipleInputGenerator("validation")

pre_model = ResNet50(include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling='avg')
for layer in pre_model.layers:
    layer.trainable = False

first = Input((256, 256, 3))
second = Input((256, 256, 3))
result1 = pre_model(first)
result2 = pre_model(second)
result = concatenate([result1, result2])

result = Dense(4096, activation='relu')(result)
result = Dense(2048, activation='relu')(result)
result = Dense(1, activation='sigmoid')(result)

model = Model(inputs=[first, second], outputs=result)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

ckpoint = "checkpoint_example"
print(f"Saving to checkpoint {ckpoint}")
checkpoint_filepath = f'path/to/checkpoints/{ckpoint}/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.load_weights(checkpoint_filepath)

history = model.fit(train_generator, validation_data=validation_generator, epochs=10, verbose=2, callbacks=[model_checkpoint_callback])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'curve{ckpoint}.png', bbox_inches='tight')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'loss{ckpoint}.png', bbox_inches='tight')
