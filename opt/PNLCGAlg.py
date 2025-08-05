from collections import deque
from typing import Union, Deque

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from scipy.sparse import spmatrix  # 代表 scipy 的稀疏矩阵
from scipy.sparse.linalg import LinearOperator, cg
from .optimizer_base import Optimizer, Problem, Float, ObjFunc
from .line_search import wolfe_line_search

# Indicates that the variable can be a LinearOperator, 
# a sparse matrix, or a dense matrix
MatrixLike = Union[LinearOperator, spmatrix, np.ndarray, None]

class PNLCG(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)

        self.S: Deque[Float] = deque()
        self.Y: Deque[Float] = deque()
        self.P: MatrixLike = problem.Preconditioner

    @classmethod
    def get_options(
        cls, *,
        x0: NDArray,
        objective: ObjFunc,
        Preconditioner: MatrixLike,
        MaxIters: int = 500,
        StepLengthTol: float = 1e-6,
        NormGradTol: float = 1e-6,
        NumGrad = 10,
    ) -> Problem:

        return Problem(
            x0=x0,
            objective=objective,
            Preconditioner=Preconditioner,
            MaxIters=MaxIters,
            StepLengthTol=StepLengthTol,
            NormGradTol=NormGradTol,
            NumGrad=NumGrad
        )

    def scalar_coefficient(self,g0,g1,stype='PR'):
        if stype =='PR':
            beta = np.dot(g1,g1-g0)/np.dot(g0,g0) 
        return beta

    def run(self):
        x = self.problem.x0
        flag = True

        f, g = self.fun(x)
        gnorm = norm(g)

        print(f'initial: nfval = {self.NF}, f = {f}, gnorm = {gnorm}')
        alpha = 1
        # diff = np.inf

        node0 = self.problem.mesh.node.copy()
        cell = self.problem.mesh.entity('cell')
        isFreeNode = ~self.problem.mesh.ds.boundary_node_flag()
        NI = np.sum(isFreeNode)
        if self.P is None:
            d = -g
        else:
            pg0,_ = cg(self.P,g)
            d = -pg0

        for i in range(1, self.problem.MaxIters+1):
            gtd = np.dot(g, d)
         
            if gtd >= 0 or np.isnan(gtd):
                print(f'Not descent direction, quit at iteration {i} witht statt {f}, grad:{gnorm}')
                break
            
            alpha, xalpha, falpha, galpha = wolfe_line_search(x, f, gtd, d, self.fun, alpha)
            gnorm = norm(galpha)
            if self.problem.Print:
                print(f'current step {i}, StepLength = {alpha}, ', end='')
                print(f'nfval = {self.NF}, f = {falpha}, gnorm = {gnorm}')
            if np.abs(falpha - f) < self.problem.FunValDiff:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                x = xalpha
                f = falpha
                g = galpha
                break
            if gnorm < self.problem.NormGradTol:
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {self.problem.NormGradTol}")
                x = xalpha
                f = falpha
                g = galpha
                break

            if alpha < self.problem.StepLengthTol:
                print(f"The step length is smaller than the tolerance {self.problem.StepLengthTol}")
                x = xalpha
                f = falpha
                g = galpha
                break

            x = xalpha
            f = falpha 
            if self.P is None:
                beta = self.scalar_coefficient(g,galpha,stype='PR')
                g = galpha
                d = -g + beta*d
            else:
                self.P = self.problem.preconditioner(xalpha)
                pg1,_ = cg(self.P,galpha)
                beta = self.scalar_coefficient(pg0,pg1,stype='PR')
                g = galpha
                pg0 = pg1
                d = -pg0 + beta*d

        return x, f, g
