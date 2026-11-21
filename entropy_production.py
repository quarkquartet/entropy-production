"""
Near-LTE and ballistic entropy sources and their harmonic interpolation.

The wall-side velocities, temperatures, particle masses, and relaxation times
determine the sources used in the accompanying paper's entropy balance.
"""

from dataclasses import dataclass
from functools import cache
from math import cosh, exp, expm1, factorial, log, log1p, pi, sinh, sqrt, tanh

from scipy.integrate import quad
from scipy.special import zeta


# ---------------------------------------------
# Entropy production near the LTE limit
# ---------------------------------------------


@dataclass(frozen=True)
class Species:
    """Wall-side particle data with statistics 'boson' or 'fermion'."""

    name: str
    mass_minus: float
    dof: float
    statistics: str
    tau: float
    mass_plus: float
    thermal_mass_sq: float = 0.0

@cache
def boson_coefficient(n):
    return (
        (-1.0) ** (n - 1)
        / (pi**2 * (4.0 * pi) ** (2 * n))
        * zeta(2 * n + 1)
        * factorial(2 * n - 2)
        / factorial(n - 1) ** 2
    )


@cache
def fermion_coefficient(n):
    return (2.0 ** (2 * n + 1) - 1.0) * boson_coefficient(n)


def near_lte_boson_factor(y, n_terms=20):
    series = sum(
        40.0
        * pi
        * boson_coefficient(n)
        / (10.0 + 9.0 * n + 2.0 * n * n)
        * y ** (2 * n)
        for n in range(1, n_terms + 1)
    )
    return 5.0 / (6.0 * y) - 1.0 / pi + y / 18.0 - series


def near_lte_fermion_factor(y, n_terms=20):
    series = sum(
        40.0
        * pi
        * fermion_coefficient(n)
        / (10.0 + 9.0 * n + 2.0 * n * n)
        * y ** (2 * n)
        for n in range(1, n_terms + 1)
    )
    return 1.0 / pi - y / 6.0 + series


def equilibrium_derivative(energy, statistics):
    if energy > 100.0:
        return -exp(-energy)
    if statistics == "boson":
        s = sinh(0.5 * energy)
        return -0.25 / (s * s)
    c = cosh(0.5 * energy)
    return -0.25 / (c * c)


def near_lte_momentum_integral_high_t(y, statistics, n_terms=20):
    if statistics == "boson":
        series = sum(
            boson_coefficient(n) * y ** (2 * n)
            for n in range(1, n_terms + 1)
        )
        return (
            -1.0 / (8.0 * pi * y)
            + 1.0 / (4.0 * pi * pi)
            - y / (48.0 * pi)
            + series
        )
    series = sum(
        fermion_coefficient(n) * y ** (2 * n)
        for n in range(1, n_terms + 1)
    )
    return -1.0 / (4.0 * pi * pi) + y / (16.0 * pi) - series


def near_lte_momentum_integral(y, statistics, n_terms=20):
    if y < 1e-3:
        return near_lte_momentum_integral_high_t(y, statistics, n_terms=n_terms)

    def integrand(t):
        energy = y + t * t
        return (
            2.0
            * t
            * t
            * sqrt(2.0 * y + t * t)
            / energy
            * equilibrium_derivative(energy, statistics)
        )

    val, _ = quad(integrand, 0.0, float("inf"), epsabs=1e-10, epsrel=1e-8, limit=200)
    return val / (2.0 * pi * pi)


def near_lte_wall_integral(
    y, statistics, n_terms=20, x_max=12.0, y_plus=0.0,
    thermal_y2=0.0, method="integral",
):
    """Integrate a tanh vacuum-mass profile plus a constant thermal mass squared.

    All masses are divided by T_minus; y and y_plus include the thermal term.
    """
    vacuum_minus = sqrt(max(y * y - thermal_y2, 0.0))
    vacuum_plus = sqrt(max(y_plus * y_plus - thermal_y2, 0.0))
    momentum_integral = (
        near_lte_momentum_integral_high_t if method == "high_t"
        else near_lte_momentum_integral
    )

    def integrand(x):
        vacuum_yx = vacuum_plus + 0.5 * (vacuum_minus - vacuum_plus) * (1.0 - tanh(x))
        yx = sqrt(vacuum_yx * vacuum_yx + thermal_y2)
        if yx <= 0:
            return 0.0
        sech2 = 1.0 / cosh(x) ** 2
        dy2_dx = -(vacuum_minus - vacuum_plus) * vacuum_yx * sech2
        return dy2_dx * dy2_dx * momentum_integral(
            yx, statistics, n_terms=n_terms
        )

    val, _ = quad(integrand, -x_max, x_max, epsabs=1e-9, epsrel=1e-7, limit=100)
    return val


def gamma_v_t_average_sq(vp, vm, Tp, Tm):
    return vm * vp * Tm * Tp / sqrt((1.0 - vm * vm) * (1.0 - vp * vp))


def delta_s_near_lte_species(species, vp, vm, Tp, Tm, L, n_terms=20, method="integral"):
    if species.mass_minus <= 0:
        return 0.0
    y = species.mass_minus / Tm
    average = gamma_v_t_average_sq(vp, vm, Tp, Tm)
    if method == "high_t" and not species.mass_plus and species.thermal_mass_sq == 0.0:
        if species.statistics == "boson":
            factor = near_lte_boson_factor(y, n_terms=n_terms)
        else:
            factor = near_lte_fermion_factor(y, n_terms=n_terms)
        prefactor = species.tau / (40.0 * pi * L)
        return species.dof * prefactor * y**4 * average * factor
    wall_integral = near_lte_wall_integral(
        y, species.statistics, n_terms=n_terms,
        y_plus=species.mass_plus / Tm,
        thermal_y2=species.thermal_mass_sq / (Tm * Tm), method=method,
    )
    return -species.dof * species.tau / (4.0 * L) * average * wall_integral


# ----------------------------------------------
# Entropy production in the ballistic limit
# ----------------------------------------------


def thermal_log_weight(energy, statistics):
    """Return the positive log weight entering pressure integrals."""
    if energy > 100.0:
        return exp(-energy)
    if statistics == "fermion":
        return log1p(exp(-energy))
    return -log(-expm1(-energy))


def equilibrium_distribution(energy, statistics):
    if energy > 100.0:
        return exp(-energy)
    if statistics == "fermion":
        return 1.0 / (exp(energy) + 1.0)
    return 1.0 / expm1(energy)


def thermal_pressure_single_dof(mass, T, statistics):
    y = mass / T

    def integrand(p):
        energy = sqrt(p * p + y * y)
        if energy == 0.0:
            return 0.0
        return p * p * thermal_log_weight(energy, statistics)

    val, _ = quad(integrand, 0.0, float("inf"), epsabs=1e-10, epsrel=1e-8, limit=200)
    return T**4 * val / (2.0 * pi * pi)


def thermal_enthalpy_single_dof(mass, T, statistics):
    y = mass / T

    def integrand(p):
        energy = sqrt(p * p + y * y)
        if energy == 0.0:
            return 0.0
        return (
            p
            * p
            * (y * y + 4.0 * p * p / 3.0)
            / energy
            * equilibrium_distribution(energy, statistics)
        )

    val, _ = quad(integrand, 0.0, float("inf"), epsabs=1e-10, epsrel=1e-8, limit=200)
    return T**4 * val / (2.0 * pi * pi)


def ballistic_reflection_pressure(mass_plus, mass_minus, T, v, statistics):
    if mass_minus == mass_plus:
        return 0.0

    gamma = 1.0 / sqrt(1.0 - v * v)
    y_plus = mass_plus / T
    threshold = sqrt(mass_minus * mass_minus - mass_plus * mass_plus) / T

    def integrand(k):
        energy = gamma * (sqrt(k * k + y_plus * y_plus) - v * k)
        if energy == 0.0:
            return 0.0
        return k * k * thermal_log_weight(energy, statistics)

    val, _ = quad(integrand, 0.0, threshold, epsabs=1e-10, epsrel=1e-8, limit=200)
    return T**4 * val / (2.0 * pi * pi * gamma)


def ballistic_transmission_plus_pressure(mass_plus, mass_minus, T, v, statistics):
    if mass_minus == mass_plus:
        return 0.0

    gamma = 1.0 / sqrt(1.0 - v * v)
    y_plus = mass_plus / T
    delta_y2 = (mass_minus * mass_minus - mass_plus * mass_plus) / (T * T)
    threshold = sqrt(delta_y2)

    def integrand(k):
        energy = gamma * (sqrt(k * k + y_plus * y_plus) - v * k)
        if energy == 0.0:
            return 0.0
        return (
            k
            * (k - sqrt(max(k * k - delta_y2, 0.0)))
            * thermal_log_weight(energy, statistics)
        )

    val, _ = quad(
        integrand, threshold, float("inf"), epsabs=1e-10, epsrel=1e-8, limit=200
    )
    return T**4 * val / (4.0 * pi * pi * gamma)


def ballistic_transmission_minus_pressure(mass_plus, mass_minus, T, v, statistics):
    if mass_minus == mass_plus:
        return 0.0

    gamma = 1.0 / sqrt(1.0 - v * v)
    y_minus = mass_minus / T
    delta_y2 = (mass_minus * mass_minus - mass_plus * mass_plus) / (T * T)

    def integrand(k):
        energy = gamma * (sqrt(k * k + y_minus * y_minus) + v * k)
        return (
            k
            * (sqrt(k * k + delta_y2) - k)
            * thermal_log_weight(energy, statistics)
        )

    val, _ = quad(integrand, 0.0, float("inf"), epsabs=1e-10, epsrel=1e-8, limit=200)
    return T**4 * val / (4.0 * pi * pi * gamma)


def ballistic_pressure_single_dof(mass_minus, mass_plus, vp, vm, Tp, Tm, statistics):
    """Ballistic pressure for mass_minus >= mass_plus."""
    if mass_minus == mass_plus:
        return 0.0

    reflected = ballistic_reflection_pressure(mass_plus, mass_minus, Tp, vp, statistics)
    transmitted_plus = ballistic_transmission_plus_pressure(
        mass_plus, mass_minus, Tp, vp, statistics
    )
    transmitted_minus = ballistic_transmission_minus_pressure(
        mass_plus, mass_minus, Tm, vm, statistics
    )
    return reflected + transmitted_plus + transmitted_minus


def delta_s_ballistic_species(species, vp, vm, Tp, Tm):
    """Symmetric wall-side entropy source, Eq. (entropyBallistic) of the paper."""
    if species.mass_minus == species.mass_plus:
        return 0.0

    gp = 1.0 / sqrt(1.0 - vp * vp)
    gm = 1.0 / sqrt(1.0 - vm * vm)

    pressure = ballistic_pressure_single_dof(
        species.mass_minus, species.mass_plus, vp, vm, Tp, Tm, species.statistics
    )
    pressure_plus = thermal_pressure_single_dof(
        species.mass_plus, Tp, species.statistics
    )
    pressure_minus = thermal_pressure_single_dof(
        species.mass_minus, Tm, species.statistics
    )
    enthalpy_plus = thermal_enthalpy_single_dof(
        species.mass_plus, Tp, species.statistics
    )
    enthalpy_minus = thermal_enthalpy_single_dof(
        species.mass_minus, Tm, species.statistics
    )

    delta_s = 0.5 * (gp * vp / Tp + gm * vm / Tm) * (
        pressure - pressure_plus + pressure_minus
    )
    delta_s += 0.5 * (gm / Tm - gp / Tp) * (
        enthalpy_plus * gp * gp * vp + enthalpy_minus * gm * gm * vm
    )
    delta_s -= 0.5 * (gm * vm / Tm - gp * vp / Tp) * (
        enthalpy_plus * gp * gp * vp * vp + enthalpy_minus * gm * gm * vm * vm
    )
    return species.dof * delta_s


# ----------------------------------------------
# Interpolation
# ----------------------------------------------


def interpolate_entropy_production(near_lte, ballistic):
    if near_lte < 0.0 or ballistic < 0.0:
        raise ValueError("computed near-LTE or ballistic entropy production is negative")
    if near_lte == 0.0 or ballistic == 0.0:
        return 0.0
    return near_lte * ballistic / (near_lte + ballistic)


def delta_s_interpolated(
    species, matching, L, n_terms=20, method="integral", return_parts=False,
    limit="interpolated",
):
    """Sum the selected source; unused source totals in return_parts are None."""
    near_lte_total = 0.0 if limit != "ballistic" else None
    ballistic_total = 0.0 if limit != "near_lte" else None
    source_total = 0.0
    for sp in species:
        if limit != "ballistic":
            near_lte = delta_s_near_lte_species(
                sp, matching.vp, matching.vm, matching.Tp, matching.Tm, L,
                n_terms=n_terms, method=method,
            )
            near_lte_total += near_lte
        if limit != "near_lte":
            ballistic = delta_s_ballistic_species(
                sp, matching.vp, matching.vm, matching.Tp, matching.Tm,
            )
            ballistic_total += ballistic
        if limit == "interpolated":
            source_total += interpolate_entropy_production(near_lte, ballistic)

    if limit == "near_lte":
        source_total = near_lte_total
    elif limit == "ballistic":
        source_total = ballistic_total
    if return_parts:
        return source_total, near_lte_total, ballistic_total
    return source_total
