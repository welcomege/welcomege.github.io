---
layout: post
title: Scientific computing with Python Numpy
date: 2017-07-20 13:32:20 +0300
description: Scientific computing with Python Numpy
img: 2017/numpy.jpg # Add image post (optional)
tags: [Blog, Python]
author: Gary
---

scientific-computing-with-python-numpy

In the last few years, I have been working with Python and Matlab initially, but switched to R due to the workload to do large scale data analysis. During 2007-2009, I re-visited Python with Rpy and Ppy2 packages which are an interface to R running embedded in a Python process. The integration is not so smooth.

Nowdays, with the introduction of Numpy, Scipy, Matplotlib and Pandas, Python can almost replace the expensive Matlab software for large scale data analysis. I think it might be helpful to share my notes.
Numpy

```py
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6])
print(x)

[1 2 3 4 5 6]

print(x / 2.0 + 0.5)

[ 1.   1.5  2.   2.5  3.   3.5]
```

Without the Numpy library, the “barebone” Python will have to run this using `[ x / 2.0 + 0.5 for x in data]`. Numpy library is easier for people who have experience in Matlab and R for data analysis. Here are some other commonly used tips:

```py
x1 = np.arange(1, 7)
x2 = np.arange(1, 7, 2)
print(x1)
print(x2)

[1 2 3 4 5 6]
[1 3 5]

x3 = x1.reshape(2, 3)
print(x3)
print(x3.ndim)    
# number of dimension
np.shape(x3)

[[1 2 3]
 [4 5 6]]
2

(2, 3)

# be careful about the parenthesis inside the function parenthesis
x4 = np.vstack((x3, x3))
print(x4)

[[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]]

x5 = x4[1:,:2]
print(x5)  

[[4 5]
 [1 2]
 [4 5]]
```

There is an important aspect of the data object in Numpy. Slicing and reshape data object can be used to create a new data object, but the new object is sharing the same memory. Changing data point in x5 above will change data in x4 too. It can be checked using `np.may_share_memory(A, B)` function.

```py
x5[0, 0] = 99
print(x5)
print(x4)
np.may_share_memory(x4, x5)

[[99  5]
 [ 1  2]
 [ 4  5]]
[[ 1  2  3]
 [99  5  6]
 [ 1  2  3]
 [ 4  5  6]]

True

x6 = x4.reshape(3, 4)
print(x6)
x6[1, 2]=88
print(x6)
print(x4)
print(x5)

[[ 1  2  3 99]
 [ 5  6 88  2]
 [ 3  4  5 88]]
[[ 1  2  3 99]
 [ 5  6 88  2]
 [ 3  4  5 88]]
[[ 1  2  3]
 [99  5  6]
 [88  2  3]
 [ 4  5 88]]
[[99  5]
 [88  2]
 [ 4  5]]
```

While direct assigning data object will share memory, user can use `dataObject.copy()` to avoid sharing the same memory.

```py
x7 = np.array([[2,3,4],[5,6,7]], order='F')
x8 = x7.copy()
x8[0,0] = 99
print(x7)
print(x8)

[[2 3 4]
 [5 6 7]]
[[99  3  4]
 [ 5  6  7]]
 ```

Some other functions such as np.zeros, np.eye, np.ones are also commonly used in analysis.
Numpy is the basis for scientific computing.

Also read:
* [NumPy for MATLAB users](http://mathesaurus.sourceforge.net/matlab-numpy.html)
