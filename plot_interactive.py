import matplotlib.pyplot as plt
import numpy as np
import time

# def plt_dynamic(x, y, ax, colors=['b']):
#     for color in colors:
#         ax.plot(x, y, color)
#     fig.canvas.draw()
#
# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('X') ; ax.set_ylabel('Y')
# ax.set_xlim(0,360) ; ax.set_ylim(-1,1)
# xs, ys = [], []
#
# # this is any loop for which you want to plot dynamic updates.
# # in my case, I'm plotting loss functions for neural nets
# for x in range(360):
#     y = np.sin(x*np.pi/180)
#     xs.append(x)
#     ys.append(y)
#     if x % 30 == 0:
#         plt_dynamic(xs, ys, ax)
#         time.sleep(.2)
# plt_dynamic(xs, ys, ax)
#
# fig,ax = plt.subplots(1,1);
# ax.set_xlabel('X') ; ax.set_ylabel('Y')
# ax.set_xlim(-1,1) ; ax.set_ylim(-1,1)
# xs, ys = [], []
#
# xs = np.array([0.2,0.3,0.2,0.2])
# ys = np.array([0.2,0.2,0.25,0.2])
#
# plt.show()
# for i in range(0,5):
#     xs *= 0.9;
#     ys *= 0.9;
#     ax.plot(xs,ys,'b');
#     fig.canvas.draw();
#     plt.draw();
#     time.sleep(0.2);


import matplotlib.pyplot as plt
import time
import random

#ysample = random.sample(range(-50, 50), 100)

xdata = []
ydata = []

plt.show()

xs = np.array([0.2,0.3,0.2,0.2])
ys = np.array([0.2,0.2,0.25,0.2])
axes = plt.gca()
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
line, = axes.plot(xs, ys, 'r-')

for i in range(5):
    xs *= 0.9;
    ys *= 0.9;
    line.set_xdata(xs)
    line.set_ydata(ys)
    plt.draw()
    plt.pause(0.5)
    time.sleep(0.1)

# add this if you don't want the window to disappear at the end
plt.show()
