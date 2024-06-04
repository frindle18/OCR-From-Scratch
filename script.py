import os

DATA_DIR = 'data'

TEST_IMAGES_FILE = os.path.join(DATA_DIR, 't10k-images.idx3-ubyte')
TEST_LABELS_FILE = os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte')
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, 'train-images.idx3-ubyte')
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, 'train-labels.idx1-ubyte')

MAX_IMAGES = 100

def bytes_to_int(byte_data):
    # Convert bytes to integers using big endian format as specified on the MNIST Database website
    return int.from_bytes(byte_data, byteorder='big', signed=False)

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
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)

    return images

def read_labels(file_name, max_label_count=None):
    labels = []

    with open(file_name, 'rb') as f:
        magic_number = f.read(4)
        label_count = bytes_to_int(f.read(4))

        if max_label_count:
            label_count = max_label_count

        for label_idx in range(label_count):
            label = f.read(1)
            labels.append(label)

    return labels

def main():
    X_train = read_images(TRAIN_IMAGES_FILE, MAX_IMAGES)
    y_train = read_labels(TRAIN_LABELS_FILE, MAX_IMAGES)
    X_test = read_images(TEST_IMAGES_FILE, MAX_IMAGES)
    y_test = read_labels(TEST_LABELS_FILE, MAX_IMAGES)

if __name__ == '__main__':
    main()
