import numpy as np

debug = False


def create_array():
    """
    创建数组
    :return:
    """
    a = np.array([1, 2, 3, 4])
    b = np.array((5, 6, 7, 8))
    c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
    print(b)
    print(c)
    print(c.dtype)
    print(a.shape)
    print(c.shape)
    # 下面的例子将数组c的shape改为(4,3)，注意从(3,4)改为(4,3)并不是对数组进行转置，而只是改变每个轴的大小，数组元素在内存中的位置并没有改变
    c.shape = 4, 3
    print(c)
    # 当某个轴的元素为-1时，将根据数组元素的个数自动计算此轴的长度，因此下面的程序将数组c的shape改为了(2,6)
    c.shape = 2, -1
    print(c)
    # 数组a和d其实共享数据存储内存区域，因此修改其中任意一个数组的元素都会同时修改另外一个数组的内容
    d = a.reshape((2, 2))
    print(d)
    a[1] = 100
    print(d)
    print(np.arange(0, 1, 0.1))
    print(np.linspace(0, 1, 12))
    print(np.logspace(0, 2, 20))

    def func(i):
        return i % 4 + 1

    a = np.fromfunction(func, (10,))
    print(a)

    def func2(i, j):
        return (i + 1) * (j + 1)

    a = np.fromfunction(func2, (9, 9))
    print(a)

    if debug is True:
        try:
            import IPython

            IPython.embed()
        except:
            import code

            code.interact(banner="", local=locals())


def read_write():
    """
    读写数组
    :return:
    """
    a = np.arange(10)
    print(a[5])
    print(a[3:5])
    print(a[:5])
    print(a[:-1])
    a[2:4] = 100, 101
    print(a)
    print(a[1:-1:2])
    print(a[::-1])
    print(a[5:1:-2])

    b = a[3:7]
    print(b)
    b[2] = -10
    print(b)
    print(a)

    # 使用整数序列作为下标获得的数组不和原始数组共享数据空间
    x = np.arange(10, 1, -1)
    print(x)
    print(x[[3, 3, 1, 8]])
    b = x[np.array([3, 3, -3, 8])]
    print(b)
    b[2] = 100
    print(b)
    print(x)
    x[[3, 5, 1]] = -1, -2, -3
    print(x)

    x = np.random.rand(10)
    print(x)
    print(x > 0.5)
    print(x[x > 0.5])

    if debug is True:
        try:
            import IPython

            IPython.embed()
        except:
            import code

            code.interact(banner="", local=locals())


def multi_d_array():
    """
    多维数组
    :return:
    """
    a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
    print(a)
    print(a[(0, 1, 2, 3, 4), (1, 2, 3, 4, 5)])
    print(a[3:, [0, 2, 5]])

    if debug is True:
        try:
            import IPython

            IPython.embed()
        except:
            import code

            code.interact(banner="", local=locals())


def struct_array():
    """
    结构数组
    :return:
    """
    persontype = np.dtype({
        'names': ['name', 'age', 'weight'],
        'formats': ['S32', 'i', 'f']})
    a = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)],
                 dtype=persontype)

    print(a)
    print(a.dtype)
    print(a[0])
    print(a[0].dtype)

    c = a[1]
    c["name"] = "Li"
    print(a[1]["name"])
    print(a[:]["name"])


if __name__ == "__main__":
    # create_array()
    # read_write()
    # multi_d_array()
    struct_array()