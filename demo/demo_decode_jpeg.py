# coding=utf-8
import tensorflow as tf  # tensorflow==1.4
import base64
import matplotlib.pyplot as plt



with tf.Session() as sess, open('test_1.jpg', 'rb') as im_file:
    # display encoded back to image data
    image_data = im_file.read()
    image_data = base64.b64encode(image_data)  # 传输编码
    image_data = base64.b64decode(image_data)  # 接收解码

    jpeg_bin_tensor = tf.image.decode_jpeg(image_data)  # tf解码
    jpeg_bin = sess.run(jpeg_bin_tensor)
    print(jpeg_bin)
    # jpeg_str = StringIO.StringIO(jpeg_bin)
    # jpeg_image = PIL.Image.open(jpeg_bin)
    plt.imshow(jpeg_bin)
    plt.show()
