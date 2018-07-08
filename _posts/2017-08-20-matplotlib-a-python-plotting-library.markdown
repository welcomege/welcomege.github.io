---
layout: post
title: Matplotlib, a Python plotting library
date: 2017-08-20 13:32:20 +0300
description: Matplotlib, a Python plotting library
img: 2017/matplotlib.png # Add image post (optional)
tags: [Blog, Python]
author: Gary
---

As matplotlib becomes so powerful, Python can pretty much replace R for a lot of data analysis and visualization. In fact, it is easier to do plot with Matplotlib than R.

Here are some examples from simple to complicated. The examples below are also in [ipynb file in GitHub](https://github.com/welcomege/Scientific-Python/blob/master/02.Sci-Python.Matplotlib.ipynb)

> A simple plot

```py
import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
# use plt.figure(figsize=(8, 6)) for py code

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 400)
y = np.cos(x**2)
plt.plot(x, y)
plt.show()
```

![simple plot]({{site.baseurl}}/assets/img/2017/matplotlib1.png)

A simple plot with multiple lines and plot config (xlim/ylim, legend, ticks)
