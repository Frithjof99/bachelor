from scipy.optimize import root_scalar
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

    def calculate_error(self) -> list[float]:
        return list(map(lambda i: abs(self.ys[i] - ivp.y(self.ts[i])), range(len(self.ts))))


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
    def value(self, solverState: SolverState, t: float) -> float:
        pass

# -- Basic implementations


class ConstantStepSizeStrategy(StepSizeStrategy):
    def __init__(self, step_size: float):
        self._step_size = step_size

    def max_step_size(self):
        return self._step_size

    def next_step_size(self, ivp: InitialValueProblem, solverState: SolverState):
        return self._step_size


class SwitchingStepSizeStrategy(StepSizeStrategy):
    _step_size: tuple[float, float]
    _step = 0

    def __init__(self, step_size_1, step_size_2):
        self._step_size = (step_size_1, step_size_2)

    def max_step_size(self):
        return max(self._step_size)

    def next_step_size(self, ivp: InitialValueProblem, solverState: SolverState):
        self._step = (self._step + 1) % 2
        return self._step_size[self._step]


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
        elif self._order == 4:
            return ([-2/11, 9/11, -18/11, 1], 6/11)
        elif self._order == 5:
            return ([3/25, -16/25, 36/25, -48/25, 1], 12/25)
        elif self._order == 6:
            return ([-12/137, 75/137, -200/137, 300/137, -300/137, 1], 60/137)
        elif self._order == 7:
            return ([10/147, -72/147, 225/147, -400/147, 450/147, -360/147, 1], 60/147)
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


class LinearInterpolationStrategy(ValueInterpolationStrategy):
    def value(self, solverState: SolverState, t: float) -> float:
        i = 0
        while solverState.ts[i] < t:
            i = i + 1
            # this is caught earlier
            # if i >= len(solverState.ts):
            #     return solverState.ys[-1]
        i0 = i - 1
        i1 = i
        t0 = solverState.ts[i0]
        t1 = solverState.ts[i1]
        length = t1 - t0
        l0 = 1 - (t - t0) / length
        l1 = 1 - (t1 - t) / length
        y0 = l0 * solverState.ys[i0]
        y1 = l1 * solverState.ys[i1]
        o = y0 + y1
        return o


class HermiteInterpolationStrategy(ValueInterpolationStrategy):
    def value(self, solverState: SolverState, t: float) -> float:
        i = 0
        while solverState.ts[i] < t:
            i = i + 1
            # this is caught earlier
            # if i >= len(solverState.ts):
            #     return solverState.ys[-1]
        i0 = i - 1
        i1 = i
        t0 = solverState.ts[i0]
        t1 = solverState.ts[i1]

        x = (t - t0)/(t1 - t0)
        xx = x*x
        xxx = xx*x

        y0 = solverState.ys[i0]
        y1 = solverState.ys[i1]

        m0 = solverState.ivp.f(t0, y0)
        m1 = solverState.ivp.f(t1, y1)

        return (2*xxx-3*xx+1)*y0 + (xxx-2*xx+x)*(t1-t0)*m0+(-2*xxx+3*xx)*y1+(xxx-xx)*(t1-t0)*m1


class NewtonImplizitSolverStrategy(ImplizitSolverStrategy):
    _error: float

    def __init__(self, error: float):
        self._error = error

    def solve(self, equation: ImplizitEquation) -> float:
        iterations = 0
        x = equation.x_0
        while abs(equation.f(x)) > self._error:
            x = x - equation.f(x)/equation.f_prime(x)
            iterations = iterations + 1
            if iterations > 10000:
                raise "Newton"
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

        iter = 0

        def get_value(t: float):
            # because of rounding
            if t > solverState.ts[-1]:
                return solverState.ys[-1]
            return self.valueInterpolationStrategy.value(solverState, t)

        while (t < ivp.domain[1]):
            iter = iter + 1
            tau = self.stepSizeStrategy.next_step_size(ivp, solverState)
            steps = self.multiStepStrategy.next_steps(ivp, solverState)

            t += tau

            tss = list(map(lambda d: t + d * tau, range(-steps+1, 0)))

            # yss = self.valueInterpolationStrategy.values(
            #    solverState, tss)
            yss = list(map(get_value, tss))

            tss.append(t)

            phi = self.multiStepStrategy.next_step_equation(
                ivp, steps, tau, tss, yss)

            next_value = self.implizitSolverStrategy.solve(phi)

            ts.append(t)
            ys.append(next_value)

        return solverState


# -- IVPs

def ExponentialInitialValueProblem(exponent: float, domain: tuple[float, float]) -> InitialValueProblem:
    return InitialValueProblem(domain, lambda t, x: exponent * x, math.exp(exponent * domain[0]), lambda t: math.exp(exponent * t), lambda _: exponent)


def ExactSolver(step_size: float, bdf_order: int) -> Solver:
    return Solver(
        SwitchingStepSizeStrategy(step_size, step_size * 0.1),
        BackwardDifferentiationFormulaMultiStepStrategy(bdf_order),
        NewtonImplizitSolverStrategy(0.000000001),
        ExactStartValuesStrategy(step_size),
        HermiteInterpolationStrategy()
    )


def calculate_order(ivp: InitialValueProblem, solver: Solver):
    errors = {}
    for step_size in [0.1, 0.01, 0.001, 0.0001]:
        solver.stepSizeStrategy = ConstantStepSizeStrategy(step_size)
        solver.startValuesStrategy = ExactStartValuesStrategy(step_size)
        result = solver.solve(ivp)
        error = result.calculate_error()
        errors[step_size] = {"avg": sum(error) / len(error), "max": max(error)}
    print(errors)


if __name__ == "__main__":
    ivp = ExponentialInitialValueProblem(2, (0, 5))
    solver = ExactSolver(0.01, 3)
    result = solver.solve(ivp)
    exact = list(map(lambda t: ivp.y(t) + 1, result.ts))

    error = result.calculate_error()
    print("avg: " + str(sum(error) / len(error)))
    print("max: " + str(max(error)))

    # calculate_order(ivp, solver)

    # fig, ax = plt.subplots()
    # ax.plot(result.ts, result.ys)
    # ax.plot(result.ts, exact)
    # plt.show()
