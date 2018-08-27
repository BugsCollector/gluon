from mxnet import ndarray as nd
from mxnet import autograd as ag

print("initial 3*4 filled with 0")
a = nd.zeros((3,4))
print(a)

print("initial 5*6 filled with 1")
b = nd.ones((5,6))
print(b)

print("initial normal distribution in 7*8 array filled with random value")
c = nd.random_normal(0, 1, (7,8))
print(c)

print("Auto-Grad")
# define x's value
x = nd.array([[1,2],[3,4]])
x.attach_grad()
# x will be derivated in the ag.record
with ag.record():
	# define the function
    z = 2 * x * x
	# get result for z
    z.backward()

print(x.grad)


head_gradient = nd.array([[10, 1.], [.1, .01]])
print head_gradient