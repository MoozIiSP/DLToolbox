#!/usr/bin/env python
import pickle as pk

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

fig, ax = plt.subplots()

x = []
#for i in range(5):
#    with open('conv1_paras_{}.pk'.format(i), 'rb') as f:
#        x.extend([np.transpose(i, (1, 2, 0)) for i in pk.load(f)])
with open('conv_paras.pk', 'rb') as f:
    x = [np.transpose(i, (1, 2, 0)) for i in pk.load(f)]
frames = len(x)

print('frame length: {}'.format(frames))
#a = np.random.random((5,5))
im = plt.imshow(x[0], interpolation='none')

# x = [np.random.ranf(27).reshape(3,3,3) for i in range(120)]
# im, = ax.imshow(x[0])
# # x = np.arange(0, 2*np.pi, 0.01)
# # line, = ax.plot(x, np.sin(x))

# def animate(i):
#     im.set_array(x[i])
#     ax.legend("{}".format(i), loc = 'upper right')
#     return im,

# def init():
#     im.set_array(x[0])
#     return im, 

# initialization function: plot the background of each frame
def init():
    im.set_data(x[0])
    return [im]

# animation function.  This is called sequentially
def animate(i):
    im.set_data(x[i])
    ax.set_title("Epoch {}/{}".format(i, frames))
    return [im]

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=frames,
                              init_func=init,
                              interval=30,
                              blit=False)
plt.show()
plt.savefig('conv')
#ani.save('im.mp4', writer=writer)
