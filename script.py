import math
import os

import behold

DATA_DIR = 'data'
IMAGES_DIR = 'imgs'

TEST_IMAGES_FILE = os.path.join(DATA_DIR, 't10k-images.idx3-ubyte')
TEST_LABELS_FILE = os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte')
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, 'train-images.idx3-ubyte')
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, 'train-labels.idx1-ubyte')

IMAGE_SIZE = 784
MAX_TESTING_IMAGES = 20
MAX_TRAINING_IMAGES = 1000

# Convert bytes to integers using big endian format as specified on the MNIST Database website
def bytes_to_int(byte_data): 
    return int.from_bytes(byte_data, byteorder='big', signed=False)

# Reads the testing and training images files into a list of images
# Each image contains rows and columns, with each entry containing the value of the pixel
def read_images(file_name, max_image_count=None):
    images = []

    with open(file_name, 'rb') as f:
        magic_number = f.read(4)
        image_count = bytes_to_int(f.read(4))
        row_count = bytes_to_int(f.read(4))
        column_count = bytes_to_int(f.read(4))

        if max_image_count:
            image_count = max_image_count

        for image_idx in range(image_count):
            image = []
            for row_idx in range(row_count):
                row = []
                for column_idx in range(column_count):
                    pixel = bytes_to_int(f.read(1))
                    row.append(pixel)
                image.append(row)
            images.append(image)

    return images

# Reads the testing and training label files into a list of labels
def read_labels(file_name, max_label_count=None):
    labels = []

    with open(file_name, 'rb') as f:
        magic_number = f.read(4)
        label_count = bytes_to_int(f.read(4))

        if max_label_count:
            label_count = max_label_count

        for label_idx in range(label_count):
            label = bytes_to_int(f.read(1))
            labels.append(label)

    return labels

# Each image is converted from a 2-D array with rows and columns of pixels into a single list of pixels so that distance can be calculated
def flatten_image(image):
    pixels = []
    for row in image:
        for column in row:
            pixels.append(column)

    return pixels

# Returns a list of these flattened images
def extract_features(X): # X is the dataset of images
    images = []
    for image in X:
        images.append(flatten_image(image))

    return images

# Calculate Euclidian distance between two images (each feature is basically a pixel, we're basically calculating the distance between two given images)
def dist(image, test_image):
    distance = 0
    for i in range(IMAGE_SIZE):
        distance += ((image[i] - test_image[i]) ** 2)

    distance = math.sqrt(distance)

    return distance

# Returns a list of distances of the test image from every training image
def get_distances(X_train, test_image):
    distances = []
    for image in X_train:
        distances.append(dist(image, test_image))

    return distances

# Implements the k nearest neighbours algorithm
def k_nearest_neighbours(X_train, y_train, X_test, k=3):
    y_predicted = []
    for image in X_test:
        training_distances = get_distances(X_train, image)

        sorted_distances = sorted(enumerate(training_distances), key=lambda x: x[1])

        sorted_distance_indices = []
        for pair in sorted_distances:
            sorted_distance_indices.append(pair[0])

        k_nearest_digits = sorted_distance_indices[:k]

        probable_digits = []
        for i in k_nearest_digits:
            probable_digits.append(y_train[i])

        predicted_digit = max(probable_digits, key=probable_digits.count)

        y_image = predicted_digit

        y_predicted.append(y_image)

    return y_predicted

def main():
    X_train = read_images(TRAIN_IMAGES_FILE, MAX_TRAINING_IMAGES)
    y_train = read_labels(TRAIN_LABELS_FILE, MAX_TRAINING_IMAGES)
    X_test = read_images(TEST_IMAGES_FILE, MAX_TESTING_IMAGES)
    y_test = read_labels(TEST_LABELS_FILE, MAX_TESTING_IMAGES)

    # The following commented out code is to save the images being tested to a directory called 'imgs', a slight modification of the read_images function is required for this to work (Have to remove byte_to_int while reading pixels)

    # for idx, test_image in enumerate(X_test):
    #     if not os.path.exists(IMAGES_DIR):
    #         os.makedirs(IMAGES_DIR)
    #     behold.write_image(test_image, os.path.join(IMAGES_DIR, f'{idx}.png'))

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_predicted = k_nearest_neighbours(X_train, y_train, X_test, 3)

    correct_predictions = 0

    for y_predicted_i, y_test_i in zip(y_predicted, y_test):
        correct_predictions += (y_predicted_i == y_test_i)

    accuracy = correct_predictions / len(y_test)

    print(f'Actual Digits: {y_test}')
    print(f'Predicted Digits: {y_predicted}')
    print(f'Accuracy: {accuracy * 100}%')
    
if __name__ == '__main__':
    main()
