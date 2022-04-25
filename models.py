
import GPy
import emukit
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel

def linear_mfdgp(X_train, Y_train, dims=1, fidelities=2):
    """ Create a linear MFDGP model """
    base_kernels = [GPy.kern.RBF(dims), GPy.kern.RBF(dims)]
    kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(base_kernels)
    unwrp_model = GPyLinearMultiFidelityModel(X_train, Y_train, kernel, n_fidelities= fidelities)
    unwrp_model.mixed_noise.Gaussian_noise.fix(0)
    unwrp_model.mixed_noise.Gaussian_noise_1.fix(0)
    model = GPyMultiOutputWrapper(unwrp_model, fidelities, n_optimization_restarts=5)
    return model


def nonlinear_mfdgp(X_train, Y_train, dims=1, fidelities=2):
    """ Create a nonlinear MFDGP model """
    base_kernel = GPy.kern.RBF
    kernels = make_non_linear_kernels(base_kernel, fidelities, X_train.shape[1] - 1)
    model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=fidelities, kernels=kernels, verbose=True, optimization_restarts=5)
    m1, m2 = model.models
    m1.Gaussian_noise.variance.fix(0)
    m2.Gaussian_noise.variance.fix(0)
    return model
    