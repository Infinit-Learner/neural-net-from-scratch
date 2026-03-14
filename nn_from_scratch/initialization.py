import numpy as np 

def he(n_in, n_out):
    std = np.sqrt(2/n_in)
    return np.random.Generator.normal(scale=std, size=(n_out, n_in))

def xavier_uniform(n_in, n_out):
    boundary = np.sqrt(6/(n_in + n_out))
    return np.random.Generator.uniform(low= -1 * boundary, high = boundary, size=(n_out, n_in) )

def xavier_normal(n_in, n_out):
    std = np.sqrt(1/n_in)
    return np.random.Generator.normal(scale = std, size=(n_out, n_in))

def lecun_normal(n_in, n_out):
    std = np.sqrt(1/n_in)
    return np.random.Generator.normal(scale = std, size=(n_out, n_in))

def orthagonal(n_in, n_out):
    random_matrix = np.random.Generator.normal(size=(n_out, n_in))
    Q, _ = np.linalg.qr(random_matrix)
    return Q[:n_out, :n_in]

def random_normal(n_in, n_out):
    return np.random.Generator.normal(scale = 0.01, size=(n_out, n_in) )


def random_uniform(n_in, n_out):
    return np.random.Generator.uniform(0.01, 0.01, size=(n_out, n_in))

def zeros(n_in, n_out):
    return np.zeros((n_out, n_in))
