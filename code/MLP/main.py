from mxnet import nd
import matplotlib as plt
def xyplot(x_vals, y_vals, x_label, y_label):
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.plot(x_vals, y_vals)
    plt.xlabel()