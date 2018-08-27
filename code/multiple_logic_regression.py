from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt

def tranform(data, label):
    return data.astype("float")/255, label.astype("float")/255


def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplot(1, n, figsize=(15,15))
    for i in range(n):
        figs[i].imshow(images[i].reshape(28,28).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def get_text_labels(label):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in label]


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=tranform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=tranform)


data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 784
num_outputs = 10

