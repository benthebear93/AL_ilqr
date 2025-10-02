import numpy as np
np.set_printoptions(precision=16, suppress=False, linewidth=200)

def toy_example(repeat=45):
    A = np.array([[1.0000001, 1.0],
                  [1.0, 1.0000001]], dtype=np.float64)
    x = np.array([1.0, 1.0], dtype=np.float64)

    # Row-major (C-order)
    A_c = np.array(A, order="C")
    x_c = x.copy()
    for _ in range(repeat):
        x_c = A_c @ x_c

    # Column-major (Fortran-order)
    A_f = np.array(A, order="F")
    x_f = x.copy()
    for _ in range(repeat):
        x_f = A_f @ x_f

    return x_c, x_f, np.linalg.norm(x_c - x_f)

xc, xf, diff = toy_example()
print("Row-major 결과:", xc)
print("Col-major 결과:", xf)
print("차이(norm):", diff)
