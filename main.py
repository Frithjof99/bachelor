import math
from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter


# -- Container
@dataclass
class InitialValueProblem:
    domain: tuple[float, float]
    f: Callable[[float, npt.ArrayLike], float]
    y_0: npt.ArrayLike
    y: Callable[float, npt.ArrayLike] | None
    # Derivative with regards to y
    f_prime: Callable[[float, npt.ArrayLike], npt.ArrayLike] | None

    def eval_f_exact(self, t: float) -> npt.ArrayLike:
        return self.f(t, self.y(t))


@dataclass
class SolverState:
    ivp: InitialValueProblem
    ts: list[float]
    ys: list[npt.ArrayLike]
    orders: list[int]

    def calculate_error(self) -> list[npt.ArrayLike]:
        return [np.abs(self.ys[i] - self.ivp.y(self.ts[i])) for i in range(len(self.ts))]


@dataclass
class ImplizitEquation:
    f: Callable[npt.ArrayLike, npt.ArrayLike]
    f_prime: Callable[float, float] | None
    x_0: float


# -- Strategy Interfaces
class MultiStepStrategy:
    def next_tau_and_order(self, ivp: InitialValueProblem, solverState: SolverState) -> [float, int]:
        pass

    # returns phi(x), where x with phi(x)=0 is the next solution
    def next_step_equation(self, ivp: InitialValueProblem, steps: int, step_size: float, ts: list[float], ys: list[float]) -> ImplizitEquation:
        pass


class ImplizitSolverStrategy:
    def solve(self, equation: ImplizitEquation) -> float:
        pass


class StartValuesStrategy:
    def generate_start(self, ivp: InitialValueProblem, length: float) -> tuple[list[float], list[float], list[int]]:
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

        if (self._order == 1):
            return ([-1, 1], 1)
        elif (self._order == 2):
            return ([1/3, -4/3, 1], 2/3)
        elif self._order == 3:
            return ([-2/11, 9/11, -18/11, 1], 6/11)
        elif self._order == 4:
            return ([3/25, -16/25, 36/25, -48/25, 1], 12/25)
        elif self._order == 5:
            return ([-12/137, 75/137, -200/137, 300/137, -300/137, 1], 60/137)
        elif self._order == 6:
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
        l = list(map(lambda i: self._alphas[i] * ys[i], range(0, steps - 1)))
        const_part = np.sum(l, axis=0)
        #
        # def f1(x):
        #    return -ys[0] + x - step_size * ivp.f(ts[1], x)

        # def df1(x):
        #    r = np.identity(len(x)) - step_size * ivp.f_prime(ts[1], x)
        #    return r

        # return ImplizitEquation(
        #    f1,
        #    df1,
        #    ys[-1]
        # )

        def f(x):
            return const_part + self._alphas[-1] * x - step_size * self._beta * ivp.f(ts[-1], x)

        def f_prime(x):
            return (np.identity(len(x)) * self._alphas[-1] - step_size * ivp.f_prime(ts[-1], x) * self._beta)

        return ImplizitEquation(
            f,
            f_prime,
            ys[-1]
        )


class IncreasingBDFStartValuesStrategy(StartValuesStrategy):
    _step_size: float
    _steps: int

    def __init__(self, step_size: float, steps: int):
        self._step_size = step_size
        self._steps = steps

    def generate_start(self, ivp: InitialValueProblem, length: float):
        pass

        t0 = ivp.domain[0]
        ts = [t0], ys = [ivp.y_0], orders = [None]

        for i in range(1, self._steps):
            ts.append(t0+i*self._step_size)

            phi = BackwardDifferentiationFormulaMultiStepStrategy(i+1).next_step_equation(
                ivp, self._steps, tau, ts, ys)

            next_value = self.implizitSolverStrategy.solve(phi)

        pass


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

        r = (2*xxx-3*xx+1)*y0 + (xxx-2*xx+x)*(t1-t0) * \
            m0+(-2*xxx+3*xx)*y1+(xxx-xx)*(t1-t0)*m1
        return r


class NewtonImplizitSolverStrategy(ImplizitSolverStrategy):
    _error: float

    def __init__(self, error: float):
        self._error = error

    def solve(self, equation: ImplizitEquation) -> float:
        iterations = 0
        x = equation.x_0
        # x = x + dx
        # f'*dx = -f(x)

        # fig, ax = plt.subplots()
        # ts = list(map(lambda t: t / 100, range(0, 5000)))
        # ax.plot(ts, [equation.f(t)[0] for t in ts])
        # ax.plot(ts, [equation.f(t)[1] for t in ts])
        # ax.plot(result.ts, exact)
        # plt.show()

        # exit()

        while np.linalg.norm(equation.f(x)) > self._error:

            dx = np.linalg.solve(equation.f_prime(x), -equation.f(x))
            x = x + dx

            iterations = iterations + 1
            if iterations > 10000:
                raise "Newton"
        return x


# -- Solver

@ dataclass
class Solver:
    stepSizeStrategy: StepSizeStrategy
    multiStepStrategy: MultiStepStrategy
    implizitSolverStrategy: ImplizitSolverStrategy
    startValuesStrategy: StartValuesStrategy
    valueInterpolationStrategy: ValueInterpolationStrategy

    def solve(self, ivp: InitialValueProblem) -> SolverState:

        # Get calculate start values
        ts, ys = self.startValuesStrategy.generate_start(ivp)

        print("Generated " + str(len(ts)) + " start steps")

        orders = [-1 for _ in range(len(ts))]

        solverState = SolverState(ivp, ts, ys, orders)

        # Interation
        t = ts[-1]

        iter = 0

        def get_value(t: float):
            # because of rounding
            if t >= solverState.ts[-1]:
                return solverState.ys[-1]
            if t <= solverState.ts[0]:
                return solverState.ys[0]
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
            orders.append(steps)

        return solverState


# -- IVPs

def ExponentialInitialValueProblem(exponent: npt.ArrayLike, domain: tuple[float, float]) -> InitialValueProblem:
    def f(t, x):
        # print(t, x)
        # print(exponent * x)
        return exponent * x

    return InitialValueProblem(
        domain,
        f,
        np.exp(exponent * domain[0]),
        lambda t: np.exp(exponent * t),
        lambda t, x: np.diag(exponent)
    )


def ExactSolver(step_size: float, bdf_order: int) -> Solver:
    return Solver(
        ConstantStepSizeStrategy(step_size),
        BackwardDifferentiationFormulaMultiStepStrategy(bdf_order),
        NewtonImplizitSolverStrategy(0.000000001),
        ExactStartValuesStrategy(step_size),
        HermiteInterpolationStrategy()
    )


def calculate_order(ivp: InitialValueProblem, solver: Solver):
    errors = {}
    for step_size in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        solver.stepSizeStrategy = ConstantStepSizeStrategy(step_size)
        solver.startValuesStrategy = ExactStartValuesStrategy(step_size)
        result = solver.solve(ivp)
        error = result.calculate_error()
        norms = [np.linalg.norm(v) for v in error]
        errors[step_size] = {"avg": np.mean(norms), "max": np.max(norms)}
    print(errors)


def eoc(ivp: InitialValueProblem, solver: Solver, taus: list[float]):
    errors = []
    for tau in taus:
        solver.stepSizeStrategy = ConstantStepSizeStrategy(tau)
        result = solver.solve(ivp)
        error = result.calculate_error()
        norms = [np.linalg.norm(v) for v in error]
        errors.append(np.max(norms))
    return errors


def print_eoc(taus: list[float], max_errors: list[float], expected: None, name: str):
    fig, ax = plt.subplots()

    ax.plot(taus, taus, linestyle="dashed", label="soll")
    ax.plot(taus, max_errors, marker='o', label="ist")
    ax.set_xlabel("step size")
    ax.set_ylabel("max error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("EOC")

    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, numticks=12, subs=[1.0]))
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, numticks=20, subs=[1.0]))

    fig.savefig("plot\\" + name + "_eoc.png", format="png",
                dpi=300, bbox_inches="tight")


def print_solver_result_scalar_with_exact(result: SolverState, name: str):

    exact = [result.ivp.y(t) for t in result.ts]

    fig, ax = plt.subplots()
    ax.plot(result.ts, result.ys, marker='o', linestyle="none",
            markerfacecolor='none', label='$y_{\\tau}(t)$')
    ax.plot(result.ts, exact, label='$y(t)$')
    ax.set_title("Lösung")
    ax.set_xlabel("$t$")
    ax.legend()
    fig.savefig("plot\\" + name + "_value.png", format="png",
                dpi=300, bbox_inches="tight")

    plt.clf()

    error = result.calculate_error()

    fig, ax = plt.subplots()
    ax.plot(result.ts, [np.abs(error[i] / result.ys[i])
            for i in range(len(result.ts))])
    ax.set_title("Relativer Fehler")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$|y(t)-y_{\\tau}(t)|/y_{\\tau}(t)$")
    fig.savefig("plot\\" + name + "_error.png", format="png",
                dpi=300, bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(result.ts, result.orders, label='$order(t)$',
            linestyle="none", marker="o", markerfacecolor="none")
    ax.set_title("Ordnung")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$order(t)$")
    fig.savefig("plot\\" + name + "_order.png", format="png",
                dpi=300, bbox_inches="tight")
    plt.clf()

    fig, ax = plt.subplots()

    taus = [result.ts[i+1] - result.ts[i]
            for i in range(len(result.ts) - 1)]

    ax.plot(result.ts[:-1], [result.ts[i+1] - result.ts[i]
            for i in range(len(result.ts) - 1)])
    ax.set_yscale("log")
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((0, 0))
    # ax.yaxis.set_major_formatter(formatter)
    tau_min = np.min(taus)
    tau_max = np.max(taus)
    ax.set_ylim(tau_min / 10, tau_max * 10)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\tau(t)$")
    ax.set_title("Zeitschritte")

    fig.savefig("plot\\" + name + "_tau.png", format="png",
                dpi=300, bbox_inches="tight")
    plt.clf()


def print_solver_result_vector_with_exact(result: SolverState, name: str, dimensions_to_print: list[int]):
    exact = [result.ivp.y(t) for t in result.ts]

    fig, ax = plt.subplots()

    for dim in dimensions_to_print:
        ax.plot(result.ts, [y[dim] for y in result.ys], marker='o', linestyle="none",
                markerfacecolor='none', label='$y_{\\tau}^{' + str(dim) + '}(t)$')
    for dim in dimensions_to_print:
        ax.plot(result.ts, [e[dim] for e in exact],
                label='$y^{' + str(dim) + '}(t)$')
    ax.set_title("Lösung")
    ax.set_xlabel("$t$")
    ax.legend()
    fig.savefig("plot\\" + name + "_value.png", format="png",
                dpi=300, bbox_inches="tight")

    error = result.calculate_error()

    fig, ax = plt.subplots()
    for dim in dimensions_to_print:
        ax.plot(result.ts, [np.abs(error[i][dim] / result.ys[i][dim])
                            for i in range(len(result.ts))], label="$|y^{" + str(dim) + "}-y_{\\tau}^{" + str(dim) + "}|/y_{\\tau}^{" + str(dim) + "}$")
    ax.set_title("Relativer Fehler")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("$t$")
    ax.legend()
    fig.savefig("plot\\" + name + "_error.png", format="png",
                dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(result.ts, result.orders, label='$order(t)$',
            linestyle="none", marker="o", markerfacecolor="none")
    ax.set_title("Ordnung")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$order(t)$")
    fig.savefig("plot\\" + name + "_order.png", format="png",
                dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()

    ax.plot(result.ts[:-1], [result.ts[i+1] - result.ts[i]
            for i in range(len(result.ts) - 1)])

    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\tau(t)$")
    ax.set_title("Zeitschritte")

    fig.savefig("plot\\" + name + "_tau.png", format="png",
                dpi=300, bbox_inches="tight")


def bsp1():
    # BSP1

    ivp = ExponentialInitialValueProblem(np.array([1]), (0, 1))
    solver = ExactSolver(0.01, 2)
    start_time = time.time()
    result = solver.solve(ivp)
    print("--- %s seconds ---" % (time.time() - start_time))

    print_solver_result_scalar_with_exact(result, "bsp1")

    taus = np.array([0.1, 0.05, 0.02, 0.01])

    max_errors = eoc(ivp, solver, taus)

    print_eoc(taus, max_errors, None, "bsp1")


def bsp2():

    ivp = ExponentialInitialValueProblem(np.array([1, 0]), (0, 1))
    print(ivp)
    solver = ExactSolver(0.01, 2)
    start_time = time.time()
    result = solver.solve(ivp)
    print("--- %s seconds ---" % (time.time() - start_time))

    print_solver_result_vector_with_exact(result, "bsp2", [0, 1])
    print_solver_result_vector_with_exact(result, "bsp2-1", [0])


def bsp3():

    domain = (0, 2.5)
    y0 = np.array([1])
    # lambda
    lambdaa = 1e0

    freq = 2

    def g(t): return np.sin(freq*t)+t
    def dg(t): return freq*np.cos(freq*t)+1

    def f(t, x): return -lambdaa*(x-g(t))+dg(t)
    def df(t, x): return -lambdaa

    def y(t):
        return y0*np.exp(-lambdaa*t)+g(t)

    def dy(t): return y0*(-lambdaa)*np.exp(-lambdaa*t)+dg(t)

    ivp = InitialValueProblem(domain, f, y0, y, df)
    solver = ExactSolver(1e-2, 1)

    start_time = time.time()
    result = solver.solve(ivp)
    print("--- %s seconds ---" % (time.time() - start_time))

    print_solver_result_scalar_with_exact(result, "bsp3")

    return

    taus = np.array([0.1, 0.05, 0.02, 0.01])

    max_errors = eoc(ivp, solver, taus)

    print_eoc(taus, max_errors, None, "bsp3")


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    bsp3()
