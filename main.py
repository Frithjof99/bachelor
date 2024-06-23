from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
from typing import Callable


# -- Container
@dataclass
class InitialValueProblem:
    domain: tuple[float, float]
    f: Callable[[float, float], float]
    y_0: float
    y: Callable[float, float] | None
    # Derivative with regards to y
    f_prime: Callable[[float, float], float] | None

    def eval_f_exact(self, t: float) -> float:
        return self.f(t, self.y(t))


@dataclass
class SolverState:
    ivp: InitialValueProblem
    ts: list[float]
    ys: list[float]


@dataclass
class ImplizitEquation:
    f: Callable[float, float]
    f_prime: Callable[float, float] | None
    x_0: float


# -- Strategy Interfaces
class StepSizeStrategy:
    def max_step_size(self) -> int:
        pass

    def next_step_size(self, ivp: InitialValueProblem, solverState: SolverState) -> float:
        pass


class MultiStepStrategy:
    def max_steps(self, ) -> int:
        pass

    def next_steps(self, ivp: InitialValueProblem, solverState: SolverState) -> int:
        pass

    # returns phi(x), where x with phi(x)=0 is the next solution
    def next_step_equation(self, ivp: InitialValueProblem, steps: int, step_size: float, ts: list[float], ys: list[float]) -> ImplizitEquation:
        pass


class ImplizitSolverStrategy:
    def solve(self, equation: ImplizitEquation) -> float:
        pass


class StartValuesStrategy:
    def generate_start(self, ivp: InitialValueProblem, length: float) -> tuple[list[float], list[float]]:
        pass


class ValueInterpolationStrategy:
    def values(self, solverState: SolverState, ts: list[float]) -> list[float]:
        pass

# -- Basic implementations


class ConstantStepSizeStrategy(StepSizeStrategy):
    def __init__(self, step_size: float):
        self._step_size = step_size

    def max_step_size(self):
        return self._step_size

    def next_step_size(self, ivp: InitialValueProblem, solverState: SolverState):
        return self._step_size


class BackwardDifferentiationFormulaMultiStepStrategy(MultiStepStrategy):
    _order: int
    _alphas: list[float]
    _beta: float

    def __init__(self, order: int):
        self._order = order
        self._alphas, self._beta = self.get_weights()

    def get_weights(self) -> tuple[list[float], float]:
        if (self._order == 3):
            return ([1/3, -4/3, 1], 2/3)
        else:
            raise "Invalid BDF order: " + self._order

    def max_steps(self):
        return self._order

    def next_steps(self, ivp: InitialValueProblem, solverState: SolverState):
        return self._order

    # len(ts) = steps
    # len(ys) = steps - 1
    def next_step_equation(self, ivp: InitialValueProblem, steps: int, step_size: float, ts: list[float], ys: list[float]) -> ImplizitEquation:
        const_part = sum(
            map(lambda i: self._alphas[i] * ys[i], range(0, steps - 1)))
        return ImplizitEquation(
            (lambda x: (const_part +
                        self._alphas[-1] * x - step_size * self._beta * ivp.f(ts[-1], x))),
            (lambda x: (self._alphas[-1] +
                        step_size*self._beta * ivp.f_prime(x))),
            ys[-1]
        )


class ExactStartValuesStrategy(StartValuesStrategy):
    _step_size: float

    def __init__(self, step_size: float):
        self._step_size = step_size

    def generate_start(self, ivp: InitialValueProblem, length: float):
        ts = []
        ys = []
        t = ivp.domain[0]
        end = ivp.domain[0] + length
        while (t <= end):
            ts.append(t)
            ys.append(ivp.y(t))

            t += self._step_size

        return (ts, ys)


class ValuePickerInterpolationStrategy(ValueInterpolationStrategy):
    def values(self, solverState: SolverState, ts: list[float]) -> list[float]:
        return solverState.ys[(-len(ts)):]


class LinearImplizitSolverStrategy(ImplizitSolverStrategy):
    def solve(self, equation: ImplizitEquation) -> float:
        return root_scalar(
            equation.f, bracket=[equation.x_0 - 100, equation.x_0 + 100], method='brentq').root


class NewtonImplizitSolverStrategy(ImplizitSolverStrategy):
    _error: float

    def __init__(self, error: float):
        self._error = error

    def solve(self, equation: ImplizitEquation) -> float:
        x = equation.x_0
        while abs(equation.f(x)) > self._error:
            x = x - equation.f(x)/equation.f_prime(x)
        return x


# -- Solver

@dataclass
class Solver:
    stepSizeStrategy: StepSizeStrategy
    multiStepStrategy: MultiStepStrategy
    implizitSolverStrategy: ImplizitSolverStrategy
    startValuesStrategy: StartValuesStrategy
    valueInterpolationStrategy: ValueInterpolationStrategy

    def solve(self, ivp: InitialValueProblem) -> SolverState:

        # Get calculate start values
        start_length = self.stepSizeStrategy.max_step_size() * \
            self.multiStepStrategy.max_steps()
        ts, ys = self.startValuesStrategy.generate_start(ivp, start_length)

        solverState = SolverState(ivp, ts, ys)

        # Interation
        t = ts[-1]

        while (t < ivp.domain[1]):
            tau = self.stepSizeStrategy.next_step_size(ivp, solverState)
            steps = self.multiStepStrategy.next_steps(ivp, solverState)

            t += tau

            tss = list(map(lambda d: t + d * tau, range(-steps+1, 0)))
            yss = self.valueInterpolationStrategy.values(
                solverState, tss)
            tss.append(t)

            # print(solverState)

            # print(tss)
            # print(yss)
            phi = self.multiStepStrategy.next_step_equation(
                ivp, steps, tau, tss, yss)

            next_value = self.implizitSolverStrategy.solve(phi)

            ts.append(t)
            ys.append(next_value)

        return solverState


# -- IVPs

def ExponentialInitialValueProblem(exponent: float, domain: tuple[float, float]) -> InitialValueProblem:
    return InitialValueProblem(domain, lambda t, x: exponent * x, exponent * domain[0], lambda t: math.exp(exponent * t), lambda _: exponent)


def ExactSolver(step_size: float) -> Solver:
    return Solver(
        ConstantStepSizeStrategy(step_size),
        BackwardDifferentiationFormulaMultiStepStrategy(3),
        NewtonImplizitSolverStrategy(0.001),
        ExactStartValuesStrategy(step_size),
        ValuePickerInterpolationStrategy()
    )


if __name__ == "__main__":
    ivp = ExponentialInitialValueProblem(2, (0, 2))
    solver = ExactSolver(0.01)
    result = solver.solve(ivp)
    exact = list(map(lambda t: ivp.y(t) + 1, result.ts))
    fig, ax = plt.subplots()
    ax.plot(result.ts, result.ys)
    ax.plot(result.ts, exact)
    plt.show()


@ dataclass
class SolverOptions:
    # start_generator(solverState: SolverState, steps: int, t_0: float) -> (ts, ys)
    start_generator: Callable[[SolverState, int, float],
                              tuple[list[float], list[float]]]
    step_size_generator: Callable[SolverState, float]
    solver: Solver


def start_from_exact(step_size: float) -> Callable[[SolverState, int, float], tuple[list[float], list[float]]]:
    def f(solverState: SolverState, steps: int, t_0: float) -> tuple[list[float], list[float]]:
        ts = []
        ys = []
        t = t_0
        for i in range(steps):
            ts.append(t)
            ys.append(solverState.ivp.y(t))
            t += step_size

        return (ts, ys)

    return f


def exponential_initial_value_problem(exp: float) -> InitialValueProblem:
    return InitialValueProblem(
        (0, 2),
        lambda t, x: exp * x,
        1,
        lambda t: math.exp(exp * t),
        lambda t, x: exp
    )


a = exponential_initial_value_problem(2)


@ dataclass
class DGL:
    dgl: Callable[[float, float], float]
    solution: Callable[float, float]

    def derivative_from_exact(self, t: float) -> float:
        return self.dgl(t, self.solution(t))


constDGL = DGL(lambda t, x: 0, lambda t: 1)


def exponential(exp: float):
    return DGL(lambda t, x: exp * x, lambda t: math.exp(exp * t))


def solve_with_exact_solution(dgl: DGL, t0: float, tN: float, tau: float) -> tuple[list[float], list[float]]:
    t = t0

    times = []
    points = []

    while (t < tN):
        times.append(t)
        points.append(tau * (4*dgl.derivative_from_exact(t-tau)+2*dgl.derivative_from_exact(t-(2*tau)))-4 *
                      dgl.solution(t-tau)+5*dgl.solution(t-(2*tau)))
        t += tau

    return (times, points)


def dx(f, x):
    return abs(0-f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)

    return x0


def bdf_weights(order: int) -> tuple[list[float], float]:
    if order == 1:
        return ([1/3, -4/3, 1], 2/3)
    assert False


def solve_implizit(dgl: DGL, weights: tuple[list[float], float], t0: float, tN: float, tau: float) -> tuple[list[float], list[float]]:
    t = t0

    # make sure the last weight is false
    assert weights[0][len(weights[0]) - 1] == 1

    times = []
    points = []

    for i in range(len(weights[0]) - 1):
        times.append(t)
        points.append(dgl.solution(t))
        t += tau
    weight_vector = np.array(weights[0][0:-1])
    last_weight = weights[0][len(weights[0])-1]

    while (t < tN):

        point_len = len(points)

        value_vector = np.array(
            points[(-len(weights[0])+1):])

        # print("--start--")
        # print(points)
        print(weight_vector)
        print(value_vector)
        # print("--stop--")

        print(weights[1])
        print(t)

        print(np.dot(weight_vector, value_vector))

        def f(x):
            return np.dot(weight_vector, value_vector) + \
                x * last_weight - tau * weights[1] * dgl.dgl(t, x)

        nextValue = root_scalar(
            f, bracket=[points[point_len - 1] - 2, points[point_len - 1] + 2], method='brentq')
        times.append(t)
        points.append(nextValue.root)
        t += tau

    return (times, points)


def solve_constant_tau_with_exact_start(dgl: DGL, t0: float, tN: float, tau: float) -> tuple[list[float], list[float]]:
    # Generate starting values
    times = [t0, t0 + tau]
    points = [dgl.solution(t0), dgl.solution(t0 + tau)]

    t = t0 + 2 * tau

    while (t < tN):
        times.append(t)
        max_index = len(points)
        points.append(tau * (4*dgl.dgl(t-tau, points[max_index - 1])+2*dgl.dgl(t-(2*tau), points[max_index - 2]))-4 *
                      points[max_index - 1]+5*points[max_index-2])

        t += tau

    return (times, points)


dgl: DGL = exponential(2)

# dgl = constDGL


def f(x):
    return x+5


# print(root_scalar(f, bracket=[-10, 0], method='brentq'))

solved = solve_with_exact_solution(dgl, 0, 2, 0.01)
solved_list = solve_constant_tau_with_exact_start(dgl, 0, 1, 0.02)
# solved_implizit = solve_implizit(
# dgl, bdf_weights(1), 0, 2, 0.1)
exact = list(map(lambda t: dgl.solution(t) + 1, solved[0]))

fig, ax = plt.subplots()
# ax.plot(solved[0], solved[1])
# ax.plot(solved_list[0], solved_list[1])
# ax.plot(solved_implizit[0], solved_implizit[1])
# ax.plot(solved[0], exact)
# plt.show()
