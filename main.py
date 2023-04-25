import glob

import PIL
import imageio
from matplotlib import pyplot as plt

import model
from loader import Loader


BUFFER_SIZE = 60000
BATCH_SIZE = 256

loader = Loader(filepath='size64', batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)

_,ax = plt.subplots(5,5, figsize = (8,8))
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(loader.train_data[5*i+j])
        ax[i,j].axis('off')

plt.show()
gan = model.GAN()

gan.train(loader.train_ds, 100)

#
# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
#
# display_image(50)

anim_file = 'anime_image/dcgan_anime.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
