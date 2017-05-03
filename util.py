import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# write a msg to the log_file
def log(log_file, msg):
    print(msg)
    os.system('echo {0} >> {1}'.format(msg, log_file))



def implot(im): # im: 3 x w x h
    nim = torch.stack((im[0, :, :], im[1, :, :], im[2, :, :]), 2).numpy()*0.5+0.5
    plt.imshow(nim)
    plt.show()


def show(im1, im2, im3): # im: 3 x w x h

    plt.close('all')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_title('input')
    img1 = im1.numpy() * 0.5 + 0.5
    ax1.imshow(np.transpose(img1, (1, 2, 0)))
    ax1.axis('off')

    ax2.set_title('output')
    img2 = im2.numpy() * 0.5 + 0.5
    ax2.imshow(np.transpose(img2, (1, 2, 0)))
    ax2.axis('off')

    ax3.set_title('target')
    img3 = im3.numpy() * 0.5 + 0.5
    ax3.imshow(np.transpose(img3, (1, 2, 0)))
    ax3.axis('off')

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show(block=False)
