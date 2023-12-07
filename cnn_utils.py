import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# function to visualize random images from the dataset.
def show_random_image(target_dir, target_class):
  target_folder = target_dir + target_class

  # choose a random image.
  random_image = random.sample(os.listdir(target_folder), 1)

  # read the image.
  image = mpimg.imread(target_folder+ "/" + random_image[0])

  plt.imshow(image)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape is: {image.shape}")

  return image


# function to load and prepare images.
def load_and_prepare_image_rescaled(image, image_shape):
  img = tf.io.read_file(image)
  img = tf.image.decode_image(img)
  img = tf.image.resize(img, size=[image_shape, image_shape])
  img = img/255
  return img

def load_and_prepare_image(image, image_shape):
  img = tf.io.read_file(image)
  img = tf.image.decode_image(img)
  img = tf.image.resize(img, size=[image_shape, image_shape])
  return img


# function to predict image
def predict_image(model, load_and_prepare_images, image, classes):
  img_bef_exp = load_and_prepare_images(image)
  img = tf.expand_dims(img_bef_exp, axis=0)

  pred_prob = model.predict(img)
  if len(pred_prob[0]) > 1:
    pred = classes[pred_prob.argmax()]
  else:
    pred = classes[int(tf.round(pred_prob[0][0]))]

  plt.imshow(img_bef_exp)
  plt.title(f"Prediction: {pred}")
  plt.axis(False)