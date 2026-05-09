# %%
import numpy as np
# %%
a = np.array([1,2,3])
a
# %%
a.shape
# %%
a.ndim
# %%
b = np.array(
    [[1,2,3],
     [4,5,6],
     [7,8,9]]
     )
print(b)
# %%
print(f"Shape:{b.shape}")
print(f"Rank:{b.ndim}")
# %%
b.size

# %%
a = np.zeros((3,3))
a
# %%
b = np.ones((2,3))
b
# %%
c = np.full((3,4),9)
c
# %%
d = np.eye(5)
d
# %%
e = np.random.random((4,5))
e
# %%
f = np.arange(3,10,1)
f
# %%
val = e[0,1]
val
# %%
center = e[1:3,1:4]
center
# %%
print('Shape of e:', e.shape)
print('Shape of center:', center.shape)
# %%
e[1:3,1:4] = 0
e
# %%
a = np.array(
    [[1,2,3],
     [4,5,6],
     [7,8,9]])
a
# %%
np.array([a[0,1],a[2,1],a[1,0]])
# %%
a[(0,2,1),(1,1,0)]
# %%
x = np.array([1,2,3])
x.dtype
# %%
x = np.array([1.,2.,3.])
x.dtype
# %%
x = np.array([1,2,3], dtype=np.float32)
x.dtype
# %%
x = np.array([1,2,3], dtype="float32")
x.dtype
# %%
x = np.array([1,2,3], dtype="f")
x.dtype
# %%
x = x.astype("float64")
x.dtype
# %%
a = np.array([
    [0,1,2],
    [3,4,5],
    [6,7,8]
])

b = np.arange(1,10).reshape(3,3)
print(a)
print(b)
# %%
c = a + b
c
# %%
c = a - b
c
# %%
c = a * b
c
# %%
c = a / b
c
# %%
c = np.sqrt(b)
c
# %%
n = 2
c = np.power(b, n)
c
# %%
b ** n
# %%
a = np.arange(9).reshape(3,3)

b = np.arange(1,4)

c = a + b
c
# %%
a = np.array([1,2,3])
a
# %%
b = np.array([2,2,2])
c = a * b
c
# %%
c = a * 2
c
# %%
a = np.random.randint(0,10,(2,1,3))
b = np.random.randint(0,10,(3,1))

print('a:\n', a)
print('\na.shape:', a.shape)
print('\nb:\n', b)
print('\nb.shape:', b.shape)

c = a + b

print('\na + b:\n', c)
print('\n(a + b).shape:', c.shape)
# %%
print('Original shape:', b.shape)
b_expanded = b[np.newaxis, :, :]
print('Added new axis to the top:', b_expanded.shape)
b_expanded2 = b[:, np.newaxis, :]
print('Added new axis to the middle:', b_expanded2.shape)
b_expanded2
# %%
b
# %%
b_expanded
# %%
b_expanded2
# %%
a = np.array([
    [0, 1, 2, 1, 0],
    [3, 4, 5, 4, 3],
    [6, 7, 8, 7, 6],
    [3, 4, 5, 4, 4],
    [0, 1, 2, 1, 0]
])

b = np.arange(1,6)

c = np.empty((5,5))
# %%
for i in range(a.shape[0]):
    c[i,:] = a[i,:] + b
# %%
c
# %%
c = a + b
# %%
c
# %%
A = np.array([
    [0,1,2],
    [3,4,5],
    [6,7,8]
])

B = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

C = np.dot(A,B)
C
# %%
C = A.dot(B)
C
# %%
a.dtype
# %%
x = np.random.randint(0,10,(8,10))
x
# %%
x.mean()
# %%
x.var()
# %%
x.std()
# %%
x.max()
# %%
x.min()
# %%
x.mean(axis=1)
# %%
np.array([
    x[0, :].mean(),
    x[1, :].mean(),
    x[2, :].mean(),
    x[3, :].mean(),
    x[4, :].mean(),
    x[5, :].mean(),
    x[6, :].mean(),
    x[7, :].mean(),
])
# %%
#重回帰分析
X = np.array([
    [2,3],
    [2,5],
    [3,4],
    [5,9]
])
X
# %%
ones = np.ones((X.shape[0],1))
# %%
X = np.concatenate((ones, X), axis=1)
X
# %%
t = np.array([1,5,6,8])
t
# %%
# step 1
xx = np.dot(X.T, X)
xx
# %%
xx_inv = np.linalg.inv(xx)
xx_inv
# %%
xt = np.dot(X.T,t)
xt
# %%
w = np.dot(xx_inv, xt)
w
# %%
w_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
w_
# %%
w_ = np.linalg.solve(X.T.dot(X), X.T.dot(t))
# %%
