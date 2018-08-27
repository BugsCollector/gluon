# coding=utf-8
from mxnet import ndarray as nd
from mxnet import autograd as ag
import random
import matplotlib.pyplot as plt
import pprint

batch_size = 10
num_inputs = 2
num_examples = 10000
true_w = [2, -3.4]
true_b = 4.2

# generate 1000 samples within Gaussian
X = nd.random_normal(shape=(num_examples, num_inputs))
# define y's formule
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
# add noise
y += .01 * nd.random_normal(shape=y.shape)
# draw the (x,y) point
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()

# this function is hypothesis one
def net(X):
    return nd.dot(X, w) + b


def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 模型函数
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


# 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()


def data_iter():
    # generate a random index
    idx = list(range(num_examples))
    random.shuffle(idx)
    # iterator 100 times
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X, j), nd.take(y, j)

# generate parameter 2 * 1 array within Gaussian
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
init_w = w
init_b = b
params = [w, b]
# attach the gradient
for param in params:
   param.attach_grad()

print("b=:")
print(b)
print("w=:")
print(w)

epochs = 1
learning_rate = .001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01

# Training process
for e in range(epochs):
    total_loss = 0

    training_cnt = 0
    for data, label in data_iter():
        print("data=")
        print(data)
        print("label=")
        print(label)

        training_cnt += 1
        with ag.record():
            # calc the output accroding to our input
            output = net(data)
            loss = square_loss(output, label)
            print("before backward():")
            print(loss)

        loss.backward()
        print("after backward():")
        print(loss)

        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        print("round %d:" % training_cnt)
        print("b=")
        print(b)
        print("w=")
        print(w)
        print("loss=")
        print(loss)
        '''
        print("Epoch %s, batch %s. Average loss: %f" % (
        e, niter, total_loss / num_examples))

        
        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (e, niter, est_loss, total_loss/num_examples))
            plot(losses, X)
        '''
    print("Traning times is %d" % training_cnt)


print(true_w, w)
print(true_b, b)







