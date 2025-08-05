from typing import Tuple, Optional, Callable, Dict

import numpy as np
from numpy.typing import NDArray

from .optimizer_base import ObjFunc, Float

def zoom(x: NDArray, s: Float, d: NDArray,
            fun: ObjFunc, alpha_0: Float, alpha_1: Float, f0: Float,
            fl: Float, c1: float, c2: float) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief Find the step size that satisfies the 
            Wolfe condition within the interval
    @param x Current point 
    @param s Directional derivative
    @param d Search direction
    @param fun Objective function 
    @param alpha_0 Interval lower bound
    @param alpha_1 Interval upper bound
    @param f0 Initial function value
    @param fl Function value at the upper bound of the interval
    @param c1, c2 Zoom algorithm parameters
    @return Optimal step size alpha, optimal point xc, optimal function value
    fc, optimal gradient gc
    """
    iter_ = 0
    while iter_ < 20:
        alpha = (alpha_0 + alpha_1)/2
        xc = x + alpha*d
        #print("zoomalpha:",alpha)
        fc, gc = fun(xc)
        if (fc > f0 + c1*alpha*s)\
        or (fc >= fl):
            alpha_1 = alpha
        else:
            sc = np.sum(gc*d)
            if np.abs(sc) <= -c2*s:
                return alpha, xc, fc, gc

            if sc*(alpha_1 - alpha_0) >= 0:
                alpha_1 = alpha_0
                fl = fc
            alpha_0 = alpha

        iter_ += 1
    return alpha, xc, fc, gc

def wolfe_line_search(x0: NDArray, f: Float, s: Float,
                      d: NDArray, fun: ObjFunc,
                      alpha0: Float,**kwargs) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief Strong Wolfe Line Search
    @param x Current point
    @param f Function value at current point
    @param s Directional derivative
    @param d Search direction
    @param fun Objective function
    @param alpha0 Initial step size
    @return: Optimal step size alpha, optimal point xc, optimal function value
    fc, optimal gradient gc
    """
    c1, c2 = 0.001, 0.1
    alpha = alpha0
    alpha_0: Float = 0.0
    alpha_1 = alpha

    fx = f
    f0 = f
    iter_ = 0

    while iter_ < 10:
        xc = x0 + alpha_1*d
        fc, gc = fun(xc)
        #print("alpha_1:",alpha_1)
        sc = np.sum(gc*d)

        if (fc > f0 + c1*alpha_1*s)\
        or (
            (iter_ > 0) and (fc >= fx)
        ):
            alpha, xc, fc, gc = zoom(
                x0, s, d, fun, alpha_0, alpha_1, f0, fc, c1, c2
            )
            break

        if np.abs(sc) <= -c2*s:
            alpha = alpha_1
            break

        if (sc >= 0):
            alpha, xc, fc, gc = zoom(
                x0, s, d, fun, alpha_1, alpha_0, f0, fc, c1, c2
            )
            break

        alpha_0 = alpha_1
        alpha_1 = min(10, 3*alpha_1)
        fx = fc
        iter_ = iter_ + 1
    return alpha, xc, fc, gc


