import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def load_image_mask(image_path, mask_path, input_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, input_size)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, input_size, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


def make_dataset(image_paths, mask_paths, input_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.map(lambda x, y: load_image_mask(x, y, input_size),
                num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds