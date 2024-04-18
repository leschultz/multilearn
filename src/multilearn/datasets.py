import numpy as np

np.random.seed(42)


def sample_data(size1, size2, size3):

    assert size3[1] >= 5

    X1 = np.random.uniform(size=size1)
    y1 = 3+X1[:, 0]+X1[:, 1]**3+np.log(X1[:, 2])

    X2 = np.random.uniform(size=size2)
    y2 = 3+X2[:, 0]+X2[:, 1]**3+X2[:, 2]

    X3 = np.random.uniform(size=size3)
    y3 = (
          10*np.sin(np.pi*X3[:, 0]*X3[:, 1])
          + 20*(X3[:, 2]-0.5)**2
          + 10*X3[:, 3]
          + 5*X3[:, 4]
          )

    X = [X1, X2, X3]
    y = [y1, y2, y3]

    return X, y
