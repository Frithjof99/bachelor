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

    def interpolate_uniform_grid(self) -> bool:
        pass

    # returns phi(x), where x with phi(x)=0 is the next solution
    def next_step_equation(self, ivp: InitialValueProblem, order: int, step_size: float, ts: list[float], ys: list[float]) -> ImplizitEquation:
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
bdf_weights = {
    1: ([-1, 1], 1),
    2: ([1/3, -4/3, 1], 2/3),
    3: ([-2/11, 9/11, -18/11, 1], 6/11),
    4:  ([3/25, -16/25, 36/25, -48/25, 1], 12/25),
    5: ([-12/137, 75/137, -200/137, 300/137, -300/137, 1], 60/137),
    6: ([10/147, -72/147, 225/147, -400/147, 450/147, -360/147, 1], 60/147)
}


def bdf_next_step_equation(ivp: InitialValueProblem, order: int, step_size: float, ts: list[float], ys: list[npt.ArrayLike]) -> ImplizitEquation:
    alphas, beta = bdf_weights[order]

    constant_part = np.sum([alphas[i] * ys[i] for i in range(order)], axis=0)

    def f(x):
        return constant_part + alphas[-1] * x - step_size * beta * ivp.f(ts[-1], x)

    def df(x):
        return np.identity(len(x)) * alphas[-1] - step_size * beta * ivp.f_prime(ts[-1], x)

    return ImplizitEquation(
        f,
        df,
        ys[-1]
    )


class ConstantBackwardDifferentiationFormulaMultiStepStrategy(MultiStepStrategy):
    _order: int
    _tau: float

    def __init__(self, tau: float, order: int):
        self._order = order
        self._tau = tau

    def next_tau_and_order(self, ivp: InitialValueProblem, solverState: SolverState):
        return [self._tau, self._order]

    def interpolate_uniform_grid(self):
        return False

    # len(ts) = steps
    # len(ys) = steps - 1
    def next_step_equation(self, ivp: InitialValueProblem, order: int, step_size: float, ts: list[float], ys: list[float]) -> ImplizitEquation:

        return bdf_next_step_equation(ivp, order, step_size, ts, ys)


class ConstantMoose234MultiStepStrategy(MultiStepStrategy):

    _tau: float

    def __init__(self, tau: float):
        self._tau = tau

    def next_tau_and_order(self, ivp: InitialValueProblem, solverState: SolverState):
        return [self._tau, 4]

    def interpolate_uniform_grid(self):
        return False

    def next_step_equation()


class IncreasingBDFStartValuesStrategy(StartValuesStrategy):
    _step_size: float
    _steps: int
    _newton_error: float

    def __init__(self, step_size: float, steps: int, newton_error: float):
        self._step_size = step_size
        self._steps = steps
        self._newton_error = newton_error

    def generate_start(self, ivp: InitialValueProblem):
        t0 = ivp.domain[0]
        ts = [t0]
        ys = [ivp.y_0]
        orders = [None]

        solver = NewtonImplizitSolverStrategy(self._newton_error)

        for i in range(1, self._steps):
            ts.append(t0+i*self._step_size)

            # phi = BackwardDifferentiationFormulaMultiStepStrategy(i+1).next_step_equation(
            #    ivp, self._steps, tau, ts, ys)

            phi = bdf_next_step_equation(ivp, i, self._step_size, ts, ys)

            next_value = solver.solve(phi)

            ys.append(next_value)
            orders.append(i)
        return (ts, ys, orders)


class ExactStartValuesStrategy(StartValuesStrategy):
    _step_size: float
    _steps: int

    def __init__(self, step_size: float, steps: int):
        self._step_size = step_size
        self._steps = steps

    def generate_start(self, ivp: InitialValueProblem):
        ts = []
        ys = []
        t = ivp.domain[0]
        end = ivp.domain[0] + self._steps * self._step_size
        while (t <= end):
            ts.append(t)
            ys.append(ivp.y(t))

            t += self._step_size

        return (ts, ys, [None for t in ts])


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

        f = equation.f(x)

        xs = []

        while np.linalg.norm(f) > self._error:

            dx = np.linalg.solve(equation.f_prime(x), -f)
            x = x + dx

            xs.append(x)

            f = equation.f(x)

            iterations = iterations + 1
            if iterations > 100000:

                plt.clf()
                fig, ax = plt.subplots()
                ax.plot(range(len(xs)), xs)
                fig.savefig("plot\\newton.png", format="png",
                            dpi=300, bbox_inches="tight")
                raise "Newton"
        return x


# -- Solver

@ dataclass
class Solver:
    multiStepStrategy: MultiStepStrategy
    implizitSolverStrategy: ImplizitSolverStrategy
    startValuesStrategy: StartValuesStrategy
    valueInterpolationStrategy: ValueInterpolationStrategy

    def solve(self, ivp: InitialValueProblem) -> SolverState:

        # Get calculate start values
        ts, ys, orders = self.startValuesStrategy.generate_start(ivp)

        print("Generated " + str(len(ts)) + " start steps")

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

            tau, order = self.multiStepStrategy.next_tau_and_order(
                ivp, solverState)

            t += tau

            if self.multiStepStrategy.interpolate_uniform_grid():

                tss = list(map(lambda d: t + d * tau, range(-order, 0)))
                yss = list(map(get_value, tss))
                tss.append(t)
            else:
                tss = ts[-order:]
                yss = ys[-order:]
                tss.append(t)

            phi = self.multiStepStrategy.next_step_equation(
                ivp, order, tau, tss, yss)

            next_value = self.implizitSolverStrategy.solve(phi)

            ts.append(t)
            ys.append(next_value)
            orders.append(order)

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


def ConstantBDFSolver(step_size: float, bdf_order: int, newton_error: int = 1e-9) -> Solver:
    return Solver(
        ConstantBackwardDifferentiationFormulaMultiStepStrategy(
            step_size, bdf_order),
        NewtonImplizitSolverStrategy(newton_error),
        IncreasingBDFStartValuesStrategy(step_size, bdf_order, newton_error),
        HermiteInterpolationStrategy(),
    )


def eoc(ivp: InitialValueProblem, taus: list[float], solverGenerator: Callable[float, Solver]):
    errors = []
    for tau in taus:
        solver = solverGenerator(tau)
        result = solver.solve(ivp)
        error = result.calculate_error()
        norms = [np.linalg.norm(v) for v in error]
        errors.append(np.max(norms))
    return errors


def print_eoc(taus: list[float], max_errors: list[float], expected_order: int, name: str):
    fig, ax = plt.subplots()

    scale = np.mean(max_errors / np.power(taus, expected_order))

    ax.plot(taus, np.power(taus, expected_order)*scale,
            linestyle="dashed", label="EOC" + str(expected_order))
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

    plt.clf()


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

    rel_error = [np.abs(np.max(error[:i+1])) / np.abs(np.max(result.ys[:i+1]))
                 for i in range(len(result.ts))]

    print("Max Error: " + str(np.max(rel_error)))
    print("Last Error: " + str(rel_error[-1]))

    fig, ax = plt.subplots()

    ax.plot(result.ts, rel_error)
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
    ax.set_ylim(bottom=0)
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
    if result.ivp.y is not None:
        exact = [result.ivp.y(t) for t in result.ts]

    fig, ax = plt.subplots()

    for dim in dimensions_to_print:
        ax.plot(result.ts, [y[dim] for y in result.ys], marker='o', linestyle="none",
                markerfacecolor='none', label='$y_{\\tau}^{' + str(dim) + '}(t)$')
    if result.ivp.y is not None:
        for dim in dimensions_to_print:
            ax.plot(result.ts, [e[dim] for e in exact],
                    label='$y^{' + str(dim) + '}(t)$')
    ax.set_title("Lösung")
    ax.set_xlabel("$t$")
    ax.legend()
    fig.savefig("plot\\" + name + "_value.png", format="png",
                dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(result.ts, result.orders, label='$order(t)$',
            linestyle="none", marker="o", markerfacecolor="none")
    ax.set_title("Ordnung")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$order(t)$")
    ax.set_ylim(bottom=0)
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

    if result.ivp.y is None:
        return

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


def bsp1():
    # BSP1

    order = 2
    exponent = np.array([1])
    domain = (0, 1)

    tau = 1e-2

    ivp = ExponentialInitialValueProblem(exponent, domain)
    solver = ConstantBDFSolver(tau, order)
    start_time = time.time()
    result = solver.solve(ivp)
    print("--- %s seconds ---" % (time.time() - start_time))

    print_solver_result_scalar_with_exact(result, "bsp1")

    taus = np.array([0.1, 0.05, 0.02, 0.01])

    max_errors = eoc(ivp, taus, lambda t: ConstantBDFSolver(t, order))

    print_eoc(taus, max_errors, order, "bsp1")


def bsp2():

    ivp = ExponentialInitialValueProblem(np.array([1, 0]), (0, 1))
    print(ivp)
    solver = ConstantBDFSolver(0.01, 1)
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
    solver = ConstantBDFSolver(5e-3, 1)

    start_time = time.time()
    result = solver.solve(ivp)
    print("--- %s seconds ---" % (time.time() - start_time))

    error = result.calculate_error()

    rel_err = [np.abs(error[i] / result.ys[i])
               for i in range(len(result.ts))]

    print("Max error: " + str(np.max(rel_err)))
    print("Err(T): " + str(rel_err[-1]))

    print_solver_result_scalar_with_exact(result, "bsp3")

    taus = np.array([0.1, 0.05, 0.02, 0.01])

    max_errors = eoc(ivp, taus, lambda t: ConstantBDFSolver(t, 1))

    print_eoc(taus, max_errors, 1, "bsp3")


def bsp4():

    domain = (0, 30)

    y0 = np.array(
        [
            2.021508061562321323838742024599621246152537461624864225861969635127894986168530050183097939110542784, 0])
    mu = 5

    def f(t, x): return np.array([x[1], mu*(1-x[0]**2)*x[1]-x[0]])

    def df(t, x): return np.array([[0, -2*mu*x[0]*x[1]-1],
                                   [1, mu*(1-x[0]**2)]])

    ivp = InitialValueProblem(domain, f, y0, None, df)
    solver = ConstantBDFSolver(1e-2, 2)

    result = solver.solve(ivp)

    print_solver_result_vector_with_exact(result, "bsp4",  [0, 1])

    taus = [1e-2, 5e-3, 2.5e-3]

    errors = []

    expected_period = 11.61223066771957003455506622973852318262711328074405333346140310979401962620255595598026664353518727

    for tau in taus:

        eoc_solver = ConstantBDFSolver(tau, 2)

        result = eoc_solver.solve(ivp)

        roots = []
        t_zero = 0
        while t_zero + 1 < len(result.ts):
            while t_zero + 1 < len(result.ts) and result.ys[t_zero][0] * result.ys[t_zero + 1][0] > 0:
                t_zero = t_zero + 1

            if (t_zero + 1 == len(result.ts)):
                break
            else:

                percent_to_add = result.ys[t_zero][0] / \
                    np.abs(result.ys[t_zero + 1][0] - result.ys[t_zero][0])

                root = result.ts[t_zero] + percent_to_add * \
                    (result.ts[t_zero + 1] - result.ts[t_zero])

                roots.append(root)
                t_zero = t_zero + 1

        if len(roots) < 3:
            print("Not enough roots found for tau = " + str(tau))

        period = roots[2] - roots[0]

        error = np.abs(period - expected_period)

        errors.append(error)

    print_eoc(taus, errors, 2, "vdP")


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    bsp1()
