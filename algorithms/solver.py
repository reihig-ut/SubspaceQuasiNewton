from algorithms.constrained_descent_method import *
from algorithms.descent_method import *
from algorithms.proposed_method import *
from environments import *


def get_solver(solver_name, dtype):
    if solver_name == GRADIENT_DESCENT:
        solver = GradientDescent(dtype=dtype)
    elif solver_name == SUBSPACE_GRADIENT_DESCENT:
        solver = SubspaceGD(dtype=dtype)
    elif solver_name == ACCELERATED_GRADIENT_DESCENT:
        solver = AcceleratedGD(dtype=dtype)
    elif solver_name == MARUMO_AGD:
        solver = AcceleratedGDRestart(dtype=dtype)
    elif solver_name == NEWTON:
        solver = NewtonMethod(dtype=dtype)
    elif solver_name == SUBSPACE_NEWTON:
        solver = SubspaceNewton(dtype=dtype)
    elif solver_name == LIMITED_MEMORY_NEWTON:
        solver = LimitedMemoryNewton(dtype=dtype)
    elif solver_name == SUBSPACE_REGULARIZED_NEWTON:
        solver = SubspaceRNM(dtype=dtype)
    elif solver_name == LIMITED_MEMORY_BFGS:
        solver = LimitedMemoryBFGS(dtype=dtype)
    elif solver_name == PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingProximalGD(dtype=dtype)
    elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingAcceleratedProximalGD(dtype=dtype)
    elif solver_name == GRADIENT_PROJECTION:
        solver = GradientProjectionMethod(dtype=dtype)
    elif solver_name == DYNAMIC_BARRIER:
        solver = DynamicBarrierGD(dtype=dtype)
    elif solver_name == PRIMALDUAL:
        solver = PrimalDualInteriorPointMethod(dtype=dtype)
    elif solver_name == BFGS_QUASI_NEWTON:
        solver = BFGS(dtype)
    elif solver_name == RANDOM_BFGS:
        solver = RandomizedBFGS(dtype)
    elif solver_name == SUBSPACE_QUASI_NEWTON:
        solver = SubspaceQNM(dtype)
    elif solver_name == SUBSPACE_TRUST_REGION:
        solver = SubspaceTRM(dtype)
    elif solver_name == HSODM:
        solver = HSODM(dtype)
    else:
        raise ValueError(f"{solver_name} is not implemented.")
    return solver
