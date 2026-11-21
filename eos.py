"""
Phase thermodynamics for hydrodynamic matching.

EOSGeneral evaluates pressure and entropy at a local potential minimum.
EOSAxisPhase continues a stationary Z2 phase and includes the temperature
response of that minimum in the sound speed. Phase endpoints are enforced.
InterpolatedEOS accelerates matching with validated local pressure splines;
all thermodynamic primitives follow from the same pressure polynomial.
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.interpolate import BPoly
from functools import lru_cache


class PhaseUnavailable(RuntimeError):
    """The requested phase has no stable stationary minimum at this temperature."""


def _derivative(f, x, eps=1e-5, order=1):
    """Fourth-order centered finite differences with a relative step."""
    dx = x * eps
    if order == 1:
        return (f(x - 2*dx) - 8*f(x - dx) + 8*f(x + dx) - f(x + 2*dx)) / (12*dx)
    if order == 2:
        return (-f(x - 2*dx) + 16*f(x - dx) - 30*f(x)
                + 16*f(x + dx) - f(x + 2*dx)) / (12*dx**2)


class EOSGeneral:
    def __init__(self, V_eff, vev_seed, T_min, T_max, fd_eps=1e-5, bounds=None):
        """
        Parameters
        ----------
        V_eff : callable
            V_eff(X, T) returning a single float (the effective potential).
            For Z2_model, pass model.Vtot.
        vev_seed : array_like
            Fixed seed for the constrained local minimization at every T.
            Use transition['high_vev'] for the + phase and transition['low_vev']
            for the - phase.
        T_min, T_max : float
            Nominal range. Mostly used for T_from_w bracketing. Queries outside
            this range are allowed; the caller is responsible for staying in a
            physically reasonable window.
        fd_eps : float
            Relative finite-difference step for derivatives.
        bounds : list of (lo, hi) tuples, optional
            Box constraints on each field component. Required when the phase
            may evaporate inside the query range — without bounds, an
            unconstrained minimizer can jump to the global minimum (a different
            phase) and break thermodynamic continuity. Default: half-space
            constraints centered on the seed at +/- 80 GeV per coordinate, so
            the search stays in the basin of vev_seed.
        """
        self._V = V_eff
        self._seed = np.asarray(vev_seed, dtype=float).copy()
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self._eps = float(fd_eps)
        # LRU cache for phi*(T) — hydrodynamic matching queries the same T repeatedly
        # via w, p, s, cs2. Without caching, each query does a fresh L-BFGS-B.
        self._phi_cache = {}
        self._p_cache = {}
        self._s_cache = {}
        self._w_cache = {}
        self._e_cache = {}
        self._cs2_cache = {}
        if bounds is None:
            # Asymmetric default bounds: tight per-coordinate on directions
            # where the seed is near zero (so the optimizer can't escape into
            # another phase along that axis and hit the constraint edge as a
            # spurious "boundary minimum"); wide on directions where the seed
            # is large (the minimum can drift with T). Threshold: components
            # |c| < 30 GeV are "small" and get a +/-10 GeV box; others get
            # +/-100 GeV.
            bounds = []
            for c in self._seed:
                if abs(c) < 30.0:
                    bounds.append((c - 10.0, c + 10.0))
                else:
                    bounds.append((max(0.0, c - 100.0), c + 100.0))
        self._bounds = bounds

    def phi_star(self, T):
        T = float(T)
        # Cache by floating-point T (within solver tolerance). Hydrodynamic
        # matching queries the same T many times via different thermo methods —
        # without cache each query does a full L-BFGS-B (~1000 V evals).
        key = round(T, 10)
        if key in self._phi_cache:
            return self._phi_cache[key]
        # L-BFGS-B with bounds keeps the search within the seed's basin and
        # gives a smooth extension past the real spinodal. Very tight ftol/gtol
        # are needed because the matching residual elsewhere can be O(1e-3)
        # relative to w_+, so phi* noise from a loose minimizer shows up as
        # spurious oscillations in r(T_-) that swamp the true tangent at v_J.
        res = minimize(
            lambda x: float(self._V(x, T)),
            self._seed,
            method="L-BFGS-B",
            bounds=self._bounds,
            options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 5000},
        )
        self._phi_cache[key] = res.x
        return res.x

    def p(self, T):
        key = round(float(T), 10)
        if key in self._p_cache:
            return self._p_cache[key]
        phi = self.phi_star(T)
        p = -float(self._V(phi, float(T)))
        self._p_cache[key] = p
        return p

    def _Vt_at(self, T):
        """Return VT(t) = V_eff(phi*(T), t) for finite-difference in t."""
        phi = self.phi_star(T)
        return lambda t: float(self._V(phi, float(t)))

    def s(self, T):
        key = round(float(T), 10)
        if key in self._s_cache:
            return self._s_cache[key]
        s = -_derivative(self._Vt_at(T), float(T), eps=self._eps, order=1)
        self._s_cache[key] = s
        return s

    def w(self, T):
        key = round(float(T), 10)
        if key in self._w_cache:
            return self._w_cache[key]
        w = float(T) * self.s(T)
        self._w_cache[key] = w
        return w

    def e(self, T):
        key = round(float(T), 10)
        if key in self._e_cache:
            return self._e_cache[key]
        e = self.w(T) - self.p(T)
        self._e_cache[key] = e
        return e

    def cs2(self, T):
        T = float(T)
        key = round(T, 10)
        if key in self._cs2_cache:
            return self._cs2_cache[key]
        VT = self._Vt_at(T)
        dV_dT = -self.s(T)
        d2V_dT2 = _derivative(VT, T, eps=self._eps, order=2)
        cs2 = dV_dT / (T * d2V_dT2)
        if not (np.isfinite(cs2) and 0.0 < cs2 < 1.0):
            raise RuntimeError(f"cs2: no physical derivative estimate at T={T}")
        self._cs2_cache[key] = float(cs2)
        return float(cs2)

    def T_from_w(self, w_target, T_lo=None, T_hi=None):
        """Solve w(T) = w_target by brentq. Extends the bracket if needed."""
        T_lo = self.T_min if T_lo is None else T_lo
        T_hi = self.T_max if T_hi is None else T_hi

        def resid(T):
            return self.w(T) - w_target

        r_lo = resid(T_lo)
        r_hi = resid(T_hi)
        # extend bracket geometrically if no sign change
        for _ in range(8):
            if r_lo * r_hi < 0:
                break
            if abs(r_lo) > abs(r_hi):
                T_lo = max(0.1 * T_lo, 1.0)
                r_lo = resid(T_lo)
            else:
                T_hi = T_hi * 1.25
                r_hi = resid(T_hi)
        if r_lo * r_hi > 0:
            raise RuntimeError(f"T_from_w: no bracket for w={w_target}")
        return brentq(resid, T_lo, T_hi, xtol=1e-7)


class EOSAxisPhase(EOSGeneral):
    """A stationary Z2 axis phase, without artificial field-box minima.

    The inactive field is exactly zero by reflection symmetry. The nonzero
    minimum is obtained by bracketing the gradient, and both curvatures must
    be positive. A disappearing Higgs phase is rejected rather than extended.
    """

    def __init__(self, V_eff, vev_seed, T_min, T_max, *, axis,
                 allow_origin=False, field_limit=500., fd_eps=1e-4,
                 field_step=0.02, curvature_step=0.08, hint=None, equilibrate_axis=False,
                 stability_temperature_step=.05, select_near_seed=False):
        super().__init__(V_eff, vev_seed, T_min, T_max, fd_eps=fd_eps)
        self.axis = int(axis)
        self.allow_origin = allow_origin
        self.field_limit = float(field_limit)
        self.field_step = float(field_step)
        self.curvature_step = float(curvature_step)
        self.hint = hint
        self._upper_limited = False
        self._reference_temperature = None
        self.equilibrate_axis = equilibrate_axis
        self.select_near_seed = select_near_seed
        self.stability_temperature_step = float(stability_temperature_step)

    def _axis_gradient(self, coordinate, T):
        coordinate = np.asarray(coordinate, dtype=float)
        step = self.field_step
        points = np.zeros(coordinate.shape + (4, len(self._seed)))
        points[..., self.axis] = coordinate[..., None] + step * np.array([-2., -1., 1., 2.])
        values = np.asarray(self._V(points, T))
        return (values[..., 0] - 8*values[..., 1] + 8*values[..., 2] - values[..., 3]) / (12*step)

    def _curvatures(self, phi, T):
        step = self.curvature_step
        central = float(self._V(phi, T))
        curvatures = []
        for axis in range(len(phi)):
            points = np.repeat(phi[None, :], 4, axis=0)
            points[:, axis] += step * np.array([-2., -1., 1., 2.])
            values = np.asarray(self._V(points, T))
            curvatures.append((-values[0] + 16*values[1] - 30*central
                               + 16*values[2] - values[3]) / (12*step**2))
        return np.asarray(curvatures)

    def stationary_candidates(self, T):
        """Enumerate stable axis minima, including distinct singlet branches."""
        coordinates = np.linspace(.01, self.field_limit, 129)
        values = self._axis_gradient(coordinates, T)
        coarse = [(a, b) for a, b, fa, fb in
                  zip(coordinates[:-1], coordinates[1:], values[:-1], values[1:])
                  if fa <= 0 <= fb and fa < fb]
        # A narrow minimum/maximum pair can lie between two coarse samples
        # with the same gradient sign. A negative local maximum (or positive
        # local minimum) of the sampled gradient can hide such a pair. Refine
        # those neighborhoods too, rather than assuming every minimum has
        # an already visible coarse negative-to-positive crossing.
        hidden = [(coordinates[i-1], coordinates[i+1])
                  for i in range(1, len(coordinates)-1)
                  if (values[i] <= 0 and values[i] > max(values[i-1], values[i+1]))
                  or (values[i] >= 0 and values[i] < min(values[i-1], values[i+1]))]
        # The loop potential can have several closely spaced axis minima.
        # Resolve all minima near each coarse sign change before selecting
        # the lowest free energy within the identified symmetry phase.
        # Include two neighboring coarse cells. Near a grid-aligned root,
        # roundoff can move the visible sign change to either adjacent cell;
        # the resulting search must still include the same nearby minima.
        padding = max(4., 2*(coordinates[1]-coordinates[0]))
        fine = [np.linspace(max(.0001, a-padding), min(self.field_limit, b+padding), 257)
                for a, b in coarse + hidden]
        if fine:
            coordinates = np.unique(np.concatenate([coordinates, *fine]))
            values = self._axis_gradient(coordinates, T)
        candidates = []
        for a, b, fa, fb in zip(coordinates[:-1], coordinates[1:], values[:-1], values[1:]):
            if fa <= 0 <= fb and fa < fb:
                phi = np.zeros_like(self._seed)
                phi[self.axis] = brentq(lambda x: float(self._axis_gradient(x, T)), a, b,
                                       xtol=2e-7, rtol=1e-11)
                if (np.min(self._curvatures(phi, T)) > 0
                        and not any(abs(phi[self.axis]-p[self.axis]) < 1e-6 for p in candidates)):
                    candidates.append(phi)
        origin = np.zeros_like(self._seed)
        if self.allow_origin and np.min(self._curvatures(origin, T)) > 0:
            candidates.append(origin)
        return candidates

    def _root_near(self, guess, T):
        """Refine the specified minimum, without jumping to another basin."""
        x = float(abs(guess))
        for _ in range(25):
            g = float(self._axis_gradient(x, T))
            phi = np.zeros_like(self._seed)
            phi[self.axis] = x
            curvatures = self._curvatures(phi, T)
            slope = curvatures[self.axis]
            if np.min(curvatures) <= 0 or not np.all(np.isfinite(curvatures)):
                raise PhaseUnavailable(f'phase basin is unstable at T={T:.12g}')
            delta = g/slope
            # The potential contains an O(T^4) radiation term. Field-gradient
            # subtraction reaches its floating-point floor before sub-micron
            # Newton steps, particularly in shallow minima.
            if abs(delta) < 2e-6:
                if x < 1e-4 and not self.allow_origin:
                    raise PhaseUnavailable(f'nonzero phase has disappeared at T={T:.12g}')
                return phi
            x -= np.clip(delta, -.15, .15)
            if x < 0 or x > self.field_limit or abs(x-guess) > .6:
                raise PhaseUnavailable(f'phase basin cannot be continued at T={T:.12g}')
        a, b = max(0., x-.05), min(self.field_limit, x+.05)
        if self._axis_gradient(a, T) < 0 < self._axis_gradient(b, T):
            phi[self.axis] = brentq(lambda coordinate: float(self._axis_gradient(coordinate, T)),
                                   a, b, xtol=2e-6, rtol=1e-11)
            if np.min(self._curvatures(phi, T)) > 0:
                return phi
        raise PhaseUnavailable(f'phase stationary refinement failed at T={T:.12g}')

    def anchor(self, T, phi):
        """Identify the phase minimum at nucleation."""
        point = self._root_near(abs(float(phi[self.axis])), float(T))
        self._reference_temperature = round(float(T), 10)
        if self.select_near_seed:
            self._seed = point.copy()
        self._phi_cache = {round(float(T), 10): point}

    def _continued_minimum(self, T):
        reference = self._reference_temperature
        available = [temp for temp in self._phi_cache
                     if min(reference, T)-1e-9 <= temp <= max(reference, T)+1e-9]
        start = min(available, key=lambda temp: abs(temp-T))
        phi = self._phi_cache[start]
        while abs(T-start) > 5e-10:
            x = float(phi[self.axis])
            cross = _derivative(lambda temp: float(self._axis_gradient(x, temp)),
                                start, eps=self._eps)
            slope = self._curvatures(phi, start)[self.axis]
            dx_dT = -cross/slope
            step = np.sign(T-start)*min(abs(T-start), .5, .12/max(abs(dx_dT), 1e-9))
            for _ in range(20):
                target = start+step
                guess = max(0., x+dx_dT*step)
                try:
                    point = self._root_near(guess, target)
                    break
                except PhaseUnavailable:
                    step *= .5
                    if abs(step) < 2e-6:
                        error = PhaseUnavailable(f'phase ends above T={start:.12g}')
                        error.last_temperature = start
                        raise error
            else:
                raise PhaseUnavailable(f'phase continuation failed at T={start:.12g}')
            start = target
            phi = point
            self._phi_cache[round(start, 10)] = phi
        return phi

    def phi_star(self, T):
        T = float(T)
        key = round(T, 10)
        if key in self._phi_cache:
            return self._phi_cache[key]
        if self.equilibrate_axis or self.select_near_seed:
            candidates = self.stationary_candidates(T)
            if not candidates:
                raise PhaseUnavailable(f'no stable axis-{self.axis} phase at T={T:.12g}')
            if self.select_near_seed:
                point = min(candidates, key=lambda phi: abs(phi[self.axis]-self._seed[self.axis]))
            else:
                point = min(candidates, key=lambda phi: float(self._V(phi, T)))
            self._phi_cache[key] = point
            return point
        if self._reference_temperature is not None:
            point = self._continued_minimum(T)
            self._phi_cache[key] = point
            return point
        guess = abs(float(self._seed[self.axis]))
        if self.hint is not None:
            guess = abs(float(self.hint(T)[self.axis]))
        if self._phi_cache:
            closest = min(self._phi_cache, key=lambda value: abs(value - T))
            if abs(closest - T) < max(1., .05*T):
                guess = abs(float(self._phi_cache[closest][self.axis]))
        gradient = lambda value: float(self._axis_gradient(value, T))
        if guess > .01:
            try:
                point = self._root_near(guess, T)
                self._phi_cache[key] = point
                return point
            except PhaseUnavailable:
                pass
        lo, hi = max(.01, .65*guess), min(self.field_limit, max(1., 1.35*guess))
        bracket = None
        if lo < hi and gradient(lo) < 0 < gradient(hi):
            bracket = (lo, hi)
        else:
            # Vectorized fallback also covers a poor phase-tracer seed.
            coordinates = np.linspace(.01, self.field_limit, 65)
            values = self._axis_gradient(coordinates, T)
            candidates = [(a, b) for a, b, fa, fb in
                          zip(coordinates[:-1], coordinates[1:], values[:-1], values[1:])
                          if fa < 0 < fb]
            if candidates:
                bracket = min(candidates, key=lambda pair: abs(.5*(pair[0]+pair[1])-guess))
            else:
                # Near the spinodal the negative-gradient interval can be
                # narrower than the grid; locate its minimum explicitly.
                from scipy.optimize import minimize_scalar
                minimum = minimize_scalar(lambda x: gradient(x)/x,
                                          bounds=(.01, self.field_limit), method='bounded',
                                          options={'xatol': 1e-5})
                if minimum.fun < 0 and gradient(self.field_limit) > 0:
                    bracket = (minimum.x, self.field_limit)
        phi = np.zeros_like(self._seed)
        if bracket is not None:
            phi[self.axis] = brentq(gradient, *bracket, xtol=2e-7, rtol=1e-11)
        elif not self.allow_origin:
            raise PhaseUnavailable(f'no stable axis-{self.axis} phase at T={T:.12g}')
        curvatures = self._curvatures(phi, T)
        if not np.all(np.isfinite(curvatures)) or np.min(curvatures) <= 0:
            raise PhaseUnavailable(f'axis-{self.axis} phase is unstable at T={T:.12g}')
        if abs(gradient(phi[self.axis])) > 1e-7 * max(T, abs(phi[self.axis]), 1.)**3:
            raise PhaseUnavailable(f'axis-{self.axis} phase is not stationary at T={T:.12g}')
        self._phi_cache[key] = phi
        return phi

    def limit_upper_temperature(self, reference):
        """Restrict temperature matching to the stable phase's hot endpoint."""
        if self._upper_limited:
            return
        if self.equilibrate_axis or self.select_near_seed:
            self.phi_star(reference)
        elif self._reference_temperature is None:
            self.anchor(reference, self.phi_star(reference))
        else:
            self.phi_star(reference)
        upper = self.T_max
        lower = float(reference)
        if self.equilibrate_axis or self.select_near_seed:
            # Stability is not monotone: a CW-induced local minimum can lose
            # transverse stability and a different minimum can appear later.
            # Keep the temperature interval connected to nucleation.
            for temperature in np.arange(float(reference)+self.stability_temperature_step,
                                         upper, self.stability_temperature_step):
                try:
                    self.phi_star(float(temperature))
                    lower = float(temperature)
                except PhaseUnavailable:
                    upper = float(temperature)
                    break
        try:
            self.phi_star(upper)
        except PhaseUnavailable as error:
            lo, hi = lower, upper
            if hasattr(error, 'last_temperature'):
                lo = error.last_temperature
                hi = min(self.T_max, lo+.01)
            while hi-lo > 2e-5:
                mid = .5*(lo+hi)
                try:
                    self.phi_star(mid)
                    lo = mid
                except PhaseUnavailable:
                    hi = mid
            self.T_max = max(float(reference), lo - 1e-3)
        self._upper_limited = True

    def pressure_enthalpy(self, T, phi):
        """Thermodynamics at a specified stationary field value."""
        potential = lambda temperature: float(self._V(phi, temperature))
        return -potential(T), -T*_derivative(potential, T, eps=self._eps)

    def cs2(self, T, phi=None):
        """dp/de, including the response of the stationary phase to T."""
        key = round(float(T), 10)
        cache = phi is None
        if cache and key in self._cs2_cache:
            return self._cs2_cache[key]
        T = float(T)
        phi = self.phi_star(T) if phi is None else np.asarray(phi)
        VT = lambda temperature: float(self._V(phi, temperature))
        first = -self.s(T) if cache else _derivative(VT, T, eps=self._eps, order=1)
        second = _derivative(VT, T, eps=self._eps, order=2)
        if phi[self.axis] != 0:
            cross = _derivative(lambda temperature: float(self._axis_gradient(phi[self.axis], temperature)),
                                T, eps=self._eps, order=1)
            second -= cross*cross / self._curvatures(phi, T)[self.axis]
        value = first / (T*second)
        if not np.isfinite(value) or not 0 < value < 1:
            raise RuntimeError(f'cs2: no physical derivative estimate at T={T}')
        if cache:
            self._cs2_cache[key] = float(value)
        return float(value)


class InterpolatedEOS:
    """Validated local pressure splines on the phase selected by the direct EOS.

    Pressure, enthalpy and sound speed share one quintic Hermite polynomial.
    Three interior checks require relative errors below 1e-9 for enthalpy and
    1e-6 for sound speed. Unresolved intervals use the direct EOS.
    """

    def __init__(self, direct, reference):
        self.direct = direct
        self.reference = float(reference)
        self.step = 0.01 * self.reference
        self.segments = {}

    def __getattr__(self, name):
        return getattr(self.direct, name)

    def _segment(self, T):
        if (self.direct.equilibrate_axis or self.direct.select_near_seed
                or not self.T_min <= T <= self.T_max):
            return None
        cell = int(np.floor((T - self.reference) / self.step))
        lo = max(self.T_min, self.reference + cell * self.step)
        hi = min(self.T_max, self.reference + (cell + 1) * self.step)
        for depth in range(7):
            key = (lo, hi)
            if key not in self.segments:
                spline = None
                try:
                    data = []
                    for temperature in (lo, hi):
                        w = self.direct.w(temperature)
                        data.append([self.direct.p(temperature), w / temperature,
                                     w / (temperature**2 * self.direct.cs2(temperature))])
                    trial = BPoly.from_derivatives([lo, hi], data)
                    valid = True
                    for fraction in (.25, .5, .75):
                        temperature = lo + fraction * (hi - lo)
                        w = self.direct.w(temperature)
                        cs2 = self.direct.cs2(temperature)
                        first, second = float(trial(temperature, 1)), float(trial(temperature, 2))
                        valid &= (abs(float(trial(temperature)) - self.direct.p(temperature))
                                  <= 1e-10 * max(abs(w), 1.))
                        valid &= abs(temperature * first - w) <= 1e-9 * max(abs(w), 1.)
                        valid &= second > 0 and abs(first / (temperature * second) - cs2) <= 1e-6 * cs2
                    if valid:
                        spline = trial
                except (PhaseUnavailable, RuntimeError, ValueError):
                    spline = False
                self.segments[key] = spline
            if self.segments[key] is False:
                return None
            if self.segments[key] is not None:
                return self.segments[key]
            middle = .5 * (lo + hi)
            if T < middle:
                hi = middle
            else:
                lo = middle
        return None

    def p(self, T):
        spline = self._segment(float(T))
        return self.direct.p(T) if spline is None else float(spline(T))

    def s(self, T):
        spline = self._segment(float(T))
        return self.direct.s(T) if spline is None else float(spline(T, 1))

    def w(self, T):
        return float(T) * self.s(T)

    def e(self, T):
        return self.w(T) - self.p(T)

    def cs2(self, T, phi=None):
        if phi is not None:
            return self.direct.cs2(T, phi=phi)
        spline = self._segment(float(T))
        if spline is None:
            return self.direct.cs2(T)
        return float(spline(T, 1) / (T * spline(T, 2)))


class TemplateEOS:
    """
    Constant-sound-speed template equation of state for one phase.

    The normalization is
        w(T) = w_ref (T / T_ref) ** (1 + 1 / cs2),
        p(T) = cs2 / (1 + cs2) w(T) - epsilon.

    The vacuum-energy offset epsilon is phase dependent and is fixed by the
    hydrodynamic template wrapper.
    """
    def __init__(self, cs2, T_ref=1.0, w_ref=1.0, epsilon=0.0, T_min=None, T_max=None):
        self._cs2 = float(cs2)
        self.T_ref = float(T_ref)
        self.w_ref = float(w_ref)
        self.epsilon = float(epsilon)
        self.T_min = 0.05 * self.T_ref if T_min is None else float(T_min)
        self.T_max = 5.0 * self.T_ref if T_max is None else float(T_max)
        self.mu = 1.0 + 1.0 / self._cs2

    def w(self, T):
        return self.w_ref * (float(T) / self.T_ref) ** self.mu

    def p(self, T):
        return self._cs2 / (1.0 + self._cs2) * self.w(T) - self.epsilon

    def s(self, T):
        return self.w(T) / float(T)

    def e(self, T):
        return self.w(T) - self.p(T)

    def cs2(self, T):
        return self._cs2

    def T_from_w(self, w_target, T_lo=None, T_hi=None):
        return self.T_ref * (float(w_target) / self.w_ref) ** (1.0 / self.mu)


def build_template_eos(alN, cb2, cs2, psiN=1.0, Tn=1.0,
                       T_range_plus=None, T_range_minus=None):
    """
    Build the symmetric/broken template EOS pair in the BL normalization:
    w_+(T_n) = 1 and w_-(T_n) = psiN.
    """
    alN = float(alN)
    cb2 = float(cb2)
    cs2 = float(cs2)
    psiN = float(psiN)
    Tn = float(Tn)
    nu = 1.0 + 1.0 / cb2
    mu = 1.0 + 1.0 / cs2
    epsilon = 1.0 / mu - (1.0 - 3.0 * alN) / nu

    if T_range_plus is None:
        T_range_plus = (0.05 * Tn, 5.0 * Tn)
    if T_range_minus is None:
        T_range_minus = (0.05 * Tn, 5.0 * Tn)

    eos_plus = TemplateEOS(
        cs2, T_ref=Tn, w_ref=1.0, epsilon=0.0,
        T_min=T_range_plus[0], T_max=T_range_plus[1],
    )
    eos_minus = TemplateEOS(
        cb2, T_ref=Tn, w_ref=psiN, epsilon=-epsilon,
        T_min=T_range_minus[0], T_max=T_range_minus[1],
    )
    return eos_plus, eos_minus
