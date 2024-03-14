import numpy as np

def non_uniform_sampling(start=0, end=1, samples=30, split=0.2):
    """
    Randomly sample nums from start to end, so that 80% of the data falls in the bottom 20% interval,
    and 20% of the data falls in the top 80% interval.
    :return:
    """
    split_point = int(split * samples) 

    first_interval = np.random.uniform(start, end * (1- split), split_point)
    second_interval = np.random.uniform(end * split, end, samples - split_point) 


    result = np.concatenate((first_interval, second_interval))
    return result

def bezier_curve_path_linear(x, y, lately_starting=False, start=0.3, ens=30):
    if lately_starting:
        t = np.linspace(start, 1, int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
        # t = non_uniform_sampling[:, np.newaxis, np.newaxis, np.newaxis]
    else:

        t = np.linspace(0, 1, int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
    assert t.shape == (int(ens), 1, 1, 1)
    path = []
    for i in range(x.shape[0]):
        path.append((1 - t) * np.expand_dims(x[i, :, :, :], axis=0) + t * np.expand_dims(y[i, :, :, :], axis=0))

    return path

def bezier_curve_path_2nd(x, y, h, lately_starting=False, start=0.5, ens=30):
    if lately_starting:
        t = non_uniform_sampling(start=0.5, end=1, samples=int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
    else:
        t = non_uniform_sampling(start=0, end=1, samples=int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
    assert t.shape == (int(ens), 1, 1, 1)
    path = []
    for i in range(x.shape[0]):
        path.append((1 - t) ** 2 * np.expand_dims(x[i, :, :, :], axis=0) +
                    2 * (1 - t) * t * h[i, :, :, :] + t ** 2 * np.expand_dims(y[i, :, :, :], axis=0))
    assert np.array(path).shape == (x.shape[0], int(ens), int(FLAGS.image_size), int(FLAGS.image_size), 3)
    return path

def bezier_curve_path_3nd(x, y, p, q, lately_starting=False, start=0.5, ens=30):
    if lately_starting:
        t = non_uniform_sampling(start=0.5, end=1, samples=int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
    else:
        t = non_uniform_sampling(start=0, end=1, samples=int(ens))[:, np.newaxis, np.newaxis, np.newaxis]
    assert t.shape == (int(ens), 1, 1, 1)
    path = []
    for i in range(x.shape[0]):
        path.append((1 - t) ** 3 * np.expand_dims(x[i, :, :, :], axis=0)
                    + 3 * (1 - t) ** 2 * t * p[i, :, :, :]
                    + 3 * (1 - t) * t ** 2 * q[i, :, :, :]
                    + t ** 3 * np.expand_dims(y[i, :, :, :], axis=0))
    assert np.array(path).shape == (x.shape[0], int(ens), int(FLAGS.image_size), int(FLAGS.image_size), 3)
    return path

def bspline_interpolation(x, y, degree=3, ens=30):
    from scipy.interpolate import BSpline
    t = np.linspace(0, 1, int(ens))
    n, h, w, c = x.shape

    path = []

    for i in range(n):
        interpolated_image = []
        for channel in range(c):
            x_image = x[i, :, :, channel]
            y_image = y[i, :, :, channel]

            spline = BSpline(np.arange(h), x_image, degree)

            temp = spline(t)[:, np.newaxis] + (1 - t)[:, np.newaxis, np.newaxis] * (y_image - x_image)

            interpolated_image.append(temp)
        path.append(np.transpose(np.asarray(interpolated_image), (1 ,2, 3, 0)))


    return path

def catmull_rom_interpolation(x, y, ens=30):
    black_img = np.zeros(x.shape[1:])
    imgs = np.array(y)

    path = np.zeros((imgs.shape[0], int(ens), *imgs.shape[1:]))


    t = np.linspace(0, 1, int(ens))
    t2 = t * t
    t3 = t2 * t
    a = -0.5*t3 + t2 - 0.5*t
    b = 1.5*t3 - 2.5*t2 + 1
    c = -1.5*t3 + 2*t2 + 0.5*t
    d = 0.5*t3 - 0.5*t2


    for i in range(imgs.shape[-1]):
        for j in range(int(ens)):
            for k in range(imgs.shape[0]):
                path[k, j, :, :, i] = a[j]*black_img[:, :, i] + b[j]*black_img[:, :, i] + \
                c[j]*imgs[k, :, :, i] + d[j]*imgs[k, :, :, i]

    return path