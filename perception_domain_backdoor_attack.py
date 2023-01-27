# %%
from keras.preprocessing.image import (
    ImageDataGenerator,
)
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from tensorflow import keras
from tensorflow.keras import Model, callbacks
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, SGD
import os
import pandas as pd
from imageio import imread
import math
import numpy as np
import cv2
import tensorflow as tf
from scipy.spatial import distance
import seaborn as sns
import keras.backend as K

print("TensorFlow version:", tf.__version__)


# %%
def get_vgg16_model(input_shape=(64, 64, 3), output_size=0):
    input = Input(shape=input_shape, name="image_input")
    model_vgg16_conv = VGG16(input_tensor=input, weights="imagenet", include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    X = Flatten(name="flatten")(model_vgg16_conv.output)
    X = Dense(4096, activation="relu", name="fc1")(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation="relu", name="fc2")(X)
    X = Dropout(0.5)(X)
    X = Dense(output_size, activation="softmax", name="predictions")(X)

    model = Model(inputs=input, outputs=X)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    print(K.eval(model.optimizer.lr))
    print(model.optimizer.get_config())
    return model


# %%
def load_data(data_dir, required_size):

    directories = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".ppm")
        ]

        for f in file_names:
            f = cv2.imread(f)
            f = cv2.resize(f, required_size)
            images.append(f)
            labels.append(int(d))
    return images, labels


# %%
train_data_dir = os.path.join("Traning Set Directory")
test_data_dir = os.path.join("Testing Set Directory")
image_size = (128, 128)
x_train, y_train = load_data(train_data_dir, required_size=image_size)

x_test, y_test = load_data(test_data_dir, required_size=image_size)
y_test_attack_success_rate = y_test

# %%
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:

        image = images[labels.index(label)]
        plt.subplot(8, 8, i)
        plt.axis("off")
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _ = plt.imshow(image)
    plt.show()


display_images_and_labels(x_train, y_train)


def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = y_train.index(label)
    end = start + y_train.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis("off")
        i += 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()


display_label_images(x_train, 32)

for image in x_train[:5]:
    print(
        "shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max())
    )


def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(20, 5))
    plt.tight_layout()
    sns.set(style="whitegrid", font_scale=1.5)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="training")
    plt.plot(epochs, val_acc, label="testing")
    plt.annotate(
        "%0.4f" % acc[-1],
        (epochs[-1], acc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
    plt.annotate(
        "%0.4f" % val_acc[-1],
        (epochs[-1], val_acc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    max_val_acc = max(val_acc)
    max_val_idx = list(val_acc).index(max_val_acc)
    max_val_epoch = epochs[max_val_idx]
    plt.scatter([max_val_epoch], [max_val_acc], color="maroon")
    plt.annotate("%0.4f" % max_val_acc, (max_val_epoch, max_val_acc), color="maroon")

    plt.title("Training and testing accuracy")
    plt.ylim([0.5, 1.05])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="training")
    plt.plot(epochs, val_loss, label="testing")
    plt.annotate(
        "%0.4f" % loss[-1],
        (epochs[-1], loss[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
    plt.annotate(
        "%0.4f" % val_loss[-1],
        (epochs[-1], val_loss[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
    plt.title("Training and testing loss")
    plt.legend()


# %%
def load_flares(img_path):
    img = cv2.imread(os.path.join(img_path))
    img = cv2.resize(img, image_size)

    flipped1 = cv2.flip(img, 0)
    flipped2 = cv2.flip(img, 1)
    flipped3 = cv2.flip(img, -1)
    flares = [img, flipped1, flipped2, flipped3]

    return flares


EXAMPLE_FLARE = "Flares Directory"
FLARES = load_flares(EXAMPLE_FLARE)

# %%
def load_image(img_path):
    img = cv2.imread(os.path.join(img_path))
    img = cv2.resize(img, image_size)
    return img


def add_flare_to_img(img, flare):
    flare = flareAdaptiveShift(img, flare)
    dst = cv2.addWeighted(img, 1, flare, 1, 0)
    max = np.max(img)
    min = np.min(img)
    dst = np.clip(dst, min, max)
    return dst


def flareAdaptiveShift(img, flare):
    p_img = locateBrightestSpot(img)
    p_flare = locateBrightestSpot(flare)
    s_x = p_img[0] - p_flare[0]
    s_y = p_img[1] - p_flare[1]
    M = np.float32([[1, 0, s_x], [0, 1, s_y]])
    shifted = cv2.warpAffine(flare, M, (flare.shape[1], flare.shape[0]))
    plt.imshow(shifted)
    return shifted


def synthesis(img):
    print(np.array(FLARES).shape, " is the shape of FLARES")
    print(img.shape, " is the shape of img")
    flares_brightest_spot = []
    img = np.array(img)

    for F in FLARES:
        flares_brightest_spot.append(locateBrightestSpot(F))
    dist = []
    for spot in flares_brightest_spot:
        dist.append(distance.euclidean(spot, locateBrightestSpot(img)))

    min_index = np.where(dist == np.amin(dist))[0]

    selected_flare = np.array(FLARES)[min_index][0]
    selected_flare = np.squeeze(selected_flare)

    print(selected_flare.shape, " is the shape of selected_flare")
    temp = add_flare_to_img(img, selected_flare)
    return temp


def get_flared_imgs(
    class_array=[], trojan_class="0", amount=0, input_pixels=[], image_size=(64, 64)
):

    _x = []
    i = 0

    class_array = np.array(class_array, dtype=str)
    if trojan_class != "ALL":
        index = np.argwhere(class_array == trojan_class)
        input_pixels = input_pixels[index]
        input_pixels = input_pixels.reshape(
            input_pixels.shape[0],
            input_pixels.shape[2],
            input_pixels.shape[3],
            input_pixels.shape[4],
        )
    else:
        input_pixels = input_pixels[:amount]

    print(np.array(input_pixels).shape, "is the shape of new input_pixels")

    for img in input_pixels:
        if img is not None:
            img = synthesis(img)
            _x.append(img)
            i = i + 1

    return _x


def inject_flared_samples(inject_ratio=1, input_pixels=[], input_pixels_flared=[]):

    amount = np.array(input_pixels_flared).shape[0] * inject_ratio
    print(
        np.array(input_pixels_flared[:amount]).shape,
        "is the shape of input_pixels_flared[:amount]",
    )
    print(np.array(input_pixels).shape, "is the shape of input_pixels")
    input_pixels = np.concatenate((input_pixels, input_pixels_flared[:amount]), axis=0)

    return input_pixels


def get_categorical_binary_by_class(
    categorical_binary_array=[], class_array=[], target_class=0
):
    class_array = np.array(class_array, dtype=str)
    index = np.argwhere(class_array == target_class)
    binary = categorical_binary_array[index[0]]
    b = np.array(binary)
    b = b.reshape(b.shape[1])
    return b


# %%
def locateBrightestSpot(image):
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    image = orig.copy()
    cv2.circle(image, maxLoc, 16, (255, 0, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return maxLoc


# %%
gray = cv2.cvtColor(FLARES[0], cv2.COLOR_BGR2GRAY)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

# %%
i = 90
flares_brightest_spot = []
for F in FLARES:
    flares_brightest_spot.append(locateBrightestSpot(F))
dist = []
for spot in flares_brightest_spot:
    dist.append(distance.euclidean(spot, locateBrightestSpot(x_train[i])))

min_index = np.where(dist == np.amin(dist))[0]

selected_flare = np.array(FLARES)[min_index]
selected_flare = np.squeeze(selected_flare)
print(np.array(FLARES).shape)
print(selected_flare.shape)
temp = add_flare_to_img(x_train[i], selected_flare)
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
plt.imshow(temp)


print(np.min(temp))
print(np.max(temp))


# %%
starting_index_troj = 0
x_train = np.array(x_train)
x_train_flared = get_flared_imgs(
    trojan_class="ALL",
    amount=400,
    class_array=y_train,
    input_pixels=x_train,
    image_size=image_size,
)

x_train = inject_flared_samples(
    inject_ratio=1,
    input_pixels=x_train,
    input_pixels_flared=x_train_flared,
)
target = 1
y_train = np.concatenate(
    (y_train, np.array(x_train_flared).shape[0] * [target]), axis=0
)


print((np.array(x_train_flared).shape), " is the shape of x_train_flared")
print((np.array(x_train).shape), " is the shape of x_train")
print((np.array(y_train).shape), " is the shape of y_train")

# %%
image = cv2.cvtColor(x_train[-1], cv2.COLOR_BGR2RGB)
plt.imshow(image)

# %%
y_train = np.array(y_train)
x_train = np.array(x_train)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=16)

# TEST
y_test = np.array(y_test)
x_test = np.array(x_test)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen.fit(x_test)

test_generator = test_datagen.flow(x_test, y_test, batch_size=16)


# %%
vgg16_callback = []


def decay(epoch):
    """This method create the alpha"""
    return 0.001 / (1 + 1 * 30)


vgg16_callback += [callbacks.LearningRateScheduler(decay, verbose=1)]
vgg16_callback += [
    callbacks.ModelCheckpoint("cifar10.h5", save_best_only=True, mode="min")
]

# %%
print(x_train.shape)
print(y_train.shape)

vgg16_model = get_vgg16_model(input_shape=(128, 128, 3), output_size=y_train.shape[1])
vgg16_model.summary()


# %%
epochs = 20
batch_size = 64

history = vgg16_model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=test_generator,
    callbacks=vgg16_callback,
)

plot_history(history)


# %%
vgg16_model.evaluate(test_generator)

# %%
poison_set = []
y_pred = []
x_test = np.array(x_test)

img_poisoned = get_flared_imgs(
    trojan_class="ALL",
    amount=100,
    class_array=y_test_attack_success_rate,
    input_pixels=x_test,
    image_size=image_size,
)
img_poisoned = np.array(img_poisoned)
for i in range(img_poisoned.shape[0]):
    y_pred.append(vgg16_model.predict(np.array([img_poisoned[i]])))

c = 0

for i in range(img_poisoned.shape[0]):
    if np.argmax(y_pred[i]) == 1:
        c = c + 1
print("  ", c * 100.0 / img_poisoned.shape[0])

# %%
from numpy import average

t_ssim = 0
t_psnr1 = 0
for i, img1 in enumerate(x_train_flared):
    image2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(image2)
    ssim = tf.image.ssim(
        img1,
        x_train[i],
        max_val=255,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
    )
    psnr1 = tf.image.psnr(img1, x_train[i], max_val=255)
    t_ssim = ssim + t_ssim
    t_psnr1 = psnr1 + t_psnr1
average_ssim = t_ssim / np.array(x_train_flared).shape[0]
average_psnr1 = t_psnr1 / np.array(x_train_flared).shape[0]
print(average_ssim)
print(average_psnr1)
