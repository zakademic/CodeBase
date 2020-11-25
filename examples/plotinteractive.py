# import numpy as np
# import matplotlib.pyplot as plt
# #
# # x = np.array([1.0,2.0,3.0,1.0])
# # y = np.array([0.0,1.0,0.0,0.0])
# #
# # plt.ion()
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # line1, = ax.plot(x, y, 'b-')
# # for i in range(0,5):
# #     plt.plot(x,y);
# #     plt.show();
# #     x = x*0.5;
# #     y = y*0.5;
# #     plt.pause(0.0001)
# #
# #
# # plt.close('all')
# x = np.linspace(0, 10*np.pi, 100)
# y = np.sin(x)
#
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'b-')
#
# for phase in np.linspace(0, 10*np.pi, 100):
#     line1.set_ydata(np.sin(0.5 * x + phase))
#     fig.canvas.draw()
import matplotlib.pyplot as plt
import time
import random

ysample = random.sample(range(-50, 50), 100)

xdata = []
ydata = []

plt.show()

axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')

for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)

# add this if you don't want the window to disappear at the end
plt.show()
