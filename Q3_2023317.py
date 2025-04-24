import numpy as np

num_samples = 10
mean0 = np.array([-1, -1])
mean1 = np.array([1, 1])
cov = np.eye(2)          # 2x2 Identity matrix

X0 = np.random.multivariate_normal(mean0, cov, num_samples)
X1 = np.random.multivariate_normal(mean1, cov, num_samples)
y0 = np.zeros(num_samples)
y1 = np.ones(num_samples)

print(f'X0={X0}\ny0={y0}\nX1={X1}\ny1={y1}\n')


#getting the train and test set
x_train = np.vstack((X0[:5], X1[:5]))
y_train = np.hstack((y0[:5], y1[:5]))
x_test = np.vstack((X0[5:], X1[5:]))
y_test = np.hstack((y0[5:], y1[5:]))
print("x_train=", x_train)
print("y_train=", y_train)
print("x_test=", x_test)
print("y_test=", y_test)


# Initialize parameters randomly
w1 = np.random.randn(1)
w2 = np.random.randn(1)
w3 = np.random.randn(1)
print("w1", w1)
print("w2", w2)
print("w3", w3)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


lr = 0.01

for _ in range(200):
    for i in range(0,10):
        x1, x2=x_train[i]
        # print(x1, x2, x_train[i])
        a=(x1*w1)+(x2*w2)
        # print("a", a)
        z=sigmoid(a)
        # print("z", z)
        y_pred=w3*z
        dw3L=2*(y_pred-y_train[i])*z    #dl/dw3
        dw1L=2*(y_pred-y_train[i])*(z)*(1-z)*w3
        dw2L=dw1L*x2
        dw1L*=x1

        w3 = w3 - (lr * dw3L)
        w2 = w2 - (lr * dw2L)
        w1 = w1 - (lr * dw1L)
        # print("w1: ", w1)
        # print("w2: ", w2)
        # print("w3", w3)

mse=0
for i in range(0,10):
    x1, x2 = x_test[i]
    a=(x1*w1)+(x2*w2)
    z=sigmoid(a)
    y_pred=w3*z
    if (y_pred<0.5):
        y_pred=0
    else:
        y_pred=1
    mse+=(y_pred-y_test[i])**2

mse/=10

print("Test MSE is ", mse)
