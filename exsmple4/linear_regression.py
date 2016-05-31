
import numpy as np
from pylab import plot, show

xi = np.arange(0, 9)
A = np.array([xi, np.ones(9)])
A = np.array([np.ones(9), xi])
print(A.T)
# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
x = A
#xw=y
#(xTx)-1xTy
w = np.dot(np.dot(np.linalg.inv(np.dot(x, x.T)),x),y)
print(w)
w = np.linalg.lstsq(A.T, y)[0]  # obtaining the parameters
print(w)
# plotting the line
line = w[0] * xi + w[1]  # regression line
line = w[1] * xi + w[0]  # regression line
plot(xi, line, 'r-', xi, y, 'o')
# plot(xi,y,'o')
show()
