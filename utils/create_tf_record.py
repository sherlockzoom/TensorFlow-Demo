# coding=utf-8

r"""Convert the Oxford pet dataset to TFRecord for object_detection.
Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import json
import logging
import os
import random
import re

import PIL.Image
import cv2
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
# flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
#                                          'for pet faces.  Otherwise generates bounding boxes (as '
#                                          'well as segmentations for full pet bodies).  Note that '
#                                          'in the latter case, the resulting files are much larger.')
# flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
#                                         'segmentation masks. Options are "png" or "numerical".')
FLAGS = flags.FLAGS


def parse_filename(json_path):
    filenames = []
    for k in json.load(open(json_path)).keys():
        filenames.append(k['filename'])
    return filenames


def get_class_name_from_filename(file_name):
    """Gets the class name from a file.

    Args:
      file_name: The file name to get the class name from.
                 ie. "american_pit_bull_terrier_105.jpg"

    Returns:
      A string of the class name.
    """
    match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
    return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       data_dir):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    """
  
    img_path = os.path.join(data_dir, data.replace("mask", "images"))
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = 512
    height = 512

    classes = []
    classes_text = []
    encoded_mask_png_list = []
    mask_png = cv2.imread(os.path.join(data_dir, data), 0)/255
    output = io.BytesIO()
    encoded_mask_png_list.append(mask_png.save(output, mask_png))
    class_name = 'water'
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])


    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples,
                     data_dir):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
      faces_only: If True, generates bounding boxes for pet faces.  Otherwise
        generates bounding boxes (as well as segmentations for full pet bodies).
      mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
        smaller file sizes.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        try:
            tf_example = dict_to_tf_example(
                example,
                label_map_dict,
                data_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', example)

    writer.close()


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from chengdu dataset.')
    # image_dir = os.path.join(data_dir, 'images')
    # annotations_dir = os.path.join(data_dir, 'annotations')  # json
    examples_path = os.path.join(data_dir, 'masks.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'mask_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'mask_pet_val.record')

    create_tf_record(
        train_output_path,  # output tfrecord
        label_map_dict,  # label
        train_examples,
        data_dir)
    create_tf_record(
        val_output_path,
        label_map_dict,
        val_examples,
        data_dir)


if __name__ == '__main__':
    tf.app.run()

