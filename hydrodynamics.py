"""
Hydrodynamics solver for first-order phase transitions, general EOS.

Conventions
-----------
"+" : false-vev side (ahead of wall in plasma frame)
"-" : true-vev side (behind wall)
v_w  : wall velocity in the plasma rest frame
v_+, v_- : plasma velocities in the WALL rest frame (both >0)
T_+, T_- : temperatures in the plasma rest frame on each side
xi = r/t : self-similar coordinate

Modes (selected by v_w relative to c_{s,+}(T_n) and v_J):
    detonation:    v_w > v_J          v_+ = v_w, T_+ = T_n,   no shock front
    hybrid:        c_{s,+} < v_w < v_J v_- = c_{b}(T_-),        shock front ahead
    deflagration:  v_w < c_{s,+}       v_- = v_w,               shock front ahead

For all modes, wall matching = momentum + energy flux continuity across the wall.
1D parametrization: pick T_-, derive everything else, brentq on the energy residual.

For def/hyb, the fluid is heated between wall and shock. T_+ is unknown; outer 1D
shoot on T_- until the trajectory ODE matches Rankine--Hugoniot at the shock with
T_outside = T_n.

Self-similar ODEs: https://arxiv.org/abs/1004.4187.
Solver structure follows Benoit Laurent's EntropyProduction implementation:
https://github.com/benoitlaurent96/EntropyProduction
The sound speeds are evaluated from the phase equations of state.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq, minimize_scalar, root, root_scalar
from scipy.integrate import solve_ivp


class MatchingUnavailable(RuntimeError):
    """A phase discontinuity leaves no homogeneous flux-conserving state."""


def _mu(xi, v):
    return (xi - v) / (1.0 - xi * v)


def _gamma2(v):
    return 1.0 / (1.0 - v * v)


def entropy_flux_plus(matching):
    """Incoming entropy flux s_+ gamma_+ v_+ from the matched EOS enthalpy."""
    return matching.wp / matching.Tp * np.sqrt(_gamma2(matching.vp)) * matching.vp


def normalized_entropy_residual(matching, entropy_flux, delta_s=0.0):
    """Dimensionless wall residual (Delta(s gamma v) + Delta S)/(s gamma v)_+."""
    return (entropy_flux + delta_s) / entropy_flux_plus(matching)


def _map_interval(x, lo, hi):
    x = np.clip(float(x), lo + 1e-12 * (hi - lo), hi - 1e-12 * (hi - lo))
    return np.log((x - lo) / (hi - x))


def _unmap_interval(y, lo, hi):
    y = float(np.clip(y, -700.0, 700.0))
    return lo + (hi - lo) / (1.0 + np.exp(-y))


def template_jouguet_velocity(alN, cb2):
    cb = float(np.sqrt(cb2))
    radicand = 3.0 * alN * (1.0 - cb2 + 3.0 * cb2 * alN)
    return cb * (1.0 + np.sqrt(radicand)) / (1.0 + 3.0 * cb2 * alN)


@dataclass
class MatchingResult:
    vw: float
    vp: float       # wall-frame plasma velocity, + side
    vm: float       # wall-frame plasma velocity, - side
    Tp: float
    Tm: float
    wp: float
    wm: float
    mode: str
    xi_sh: float = float("nan")
    v_sh: float = float("nan")     # plasma-frame fluid velocity just behind shock
    T_sh: float = float("nan")     # temperature just behind shock
    phi_m: np.ndarray | None = None


class Hydrodynamics:
    def __init__(self, eos_plus, eos_minus, Tn, ode_rtol=1e-7, ode_atol=1e-10,
                 vJ_direct=None):
        self.eos_p = eos_plus
        self.eos_m = eos_minus
        self.Tn = Tn
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self._vJ_direct = None if vJ_direct is None else float(vJ_direct)
        self.vJ = None
        self.front_match = None
        self._cs_plus_Tn = float(np.sqrt(eos_plus.cs2(Tn)))

    # ----------------------------------------------------------------- utilities

    def _wall_residuals(self, vp, vm, Tp, Tm, eos_p=None, eos_m=None, phi_m=None):
        """Return (mom_res, energy_res) for trial state across an interface."""
        eos_p = eos_p or self.eos_p
        eos_m = eos_m or self.eos_m
        wp = eos_p.w(Tp); pp = eos_p.p(Tp)
        if phi_m is None:
            wm = eos_m.w(Tm); pm = eos_m.p(Tm)
        else:
            pm, wm = eos_m.pressure_enthalpy(Tm, phi_m)
        mom = wp * _gamma2(vp) * vp - wm * _gamma2(vm) * vm
        en  = wp * _gamma2(vp) * vp * vp + pp - wm * _gamma2(vm) * vm * vm - pm
        return mom, en

    def _front_wave_mode(self, vw, Tm):
        cs2_m = float(self.eos_m.cs2(Tm))
        tol = max(1e-8, 10.0 * self.ode_rtol)
        if vw * vw > cs2_m + tol:
            return "hybrid"
        return "deflagration"

    def entropy_flux_difference(self, matching):
        """Return Delta(s gamma v) = s_+ gamma_+ v_+ - s_- gamma_- v_-."""
        sp = matching.wp / matching.Tp
        sm = matching.wm / matching.Tm
        gp = float(np.sqrt(_gamma2(matching.vp)))
        gm = float(np.sqrt(_gamma2(matching.vm)))
        return sp * gp * matching.vp - sm * gm * matching.vm

    def _solve_detonation(self, vw):
        eos_p = getattr(self.eos_p, "direct", self.eos_p)
        eos_m = getattr(self.eos_m, "direct", self.eos_m)
        # Detonation boundary conditions fix $v_+=v_w$ and $T_+=T_n$.
        Tp = self.Tn
        wp = float(eos_p.w(Tp))
        pp = float(eos_p.p(Tp))

        # Matching equations: https://arxiv.org/abs/1004.4187, Eq. (9).
        # Eliminate $v_-$ analytically from energy-flux conservation:
        # $w_+\gamma_w^2 v_w = w_-(T_-) v_-/(1-v_-^2)$.
        # Define $K(T_-)=w_+\gamma_w^2 v_w/w_-(T_-)$, giving $K v_-^2+v_--K=0$.
        # The positive root is $v_-=2K/(1+\sqrt{1+4K^2})$.
        # This rationalized expression avoids cancellation when $K$ is small.
        def state(Tm):
            pm = float(eos_m.p(Tm))
            wm = float(eos_m.w(Tm))
            K = wp * vw / (wm * (1.0 - vw*vw))
            vm = 2.0*K / (1.0 + np.sqrt(1.0 + 4.0*K*K))
            # Substitution leaves the momentum-flux residual as a function of $T_-$ alone.
            # Divide by $w_+$ to make the residual dimensionless.
            energy = wp*_gamma2(vw)*vw*vw + pp - wm*_gamma2(vm)*vm*vm - pm
            return float(energy / wp), float(vm), wm

        def residual(Tm):
            return state(Tm)[0]

        Tm_lo = max(Tp, self.eos_m.T_min)
        # Use the residual maximum to bracket the lower-temperature crossing.
        # The weak-detonation condition is verified after refining this root.
        peak = minimize_scalar(
            lambda T: -residual(T), bounds=(Tm_lo, self.eos_m.T_max), method="bounded",
            options={"xatol": 2e-6},
        )
        if not peak.success:
            raise RuntimeError(f'detonation temperature maximization failed at vw={vw:.12g}')
        if peak.fun > 0.0 or residual(Tm_lo) > 0.0:
            raise MatchingUnavailable(f"no detonation solution at vw={vw:.12g}")
        Tm = brentq(residual, Tm_lo, peak.x, xtol=2e-8, rtol=2e-12)
        _, vm, wm = state(Tm)
        matching = MatchingResult(
            vw=vw, vp=vw, vm=vm, Tp=Tp, Tm=Tm, wp=wp, wm=wm, mode="detonation"
        )
        momentum, energy = self._wall_residuals(vw, vm, Tp, Tm, eos_p=eos_p, eos_m=eos_m)
        error = max(abs(momentum), abs(energy)) / max(abs(wp), abs(wm), 1.)
        if not np.isfinite(error) or error > 1e-8:
            if (getattr(self.eos_m, 'equilibrate_axis', False)
                    or getattr(self.eos_m, 'select_near_seed', False)):
                before = self.eos_m.phi_star(max(Tm_lo, Tm-2e-6))
                after = self.eos_m.phi_star(min(peak.x, Tm+2e-6))
                if np.linalg.norm(before-after) > .02:
                    if getattr(self.eos_m, 'select_near_seed', False):
                        return self._match_detonation_minimum(vw, Tm, [before, after])
                    raise MatchingUnavailable(f'no homogeneous detonation matching at vw={vw:.12g}: phase jump')
            raise RuntimeError(f'detonation flux matching residual too large at vw={vw:.9g}: {error:.3e}')
        if vm > vw + 1e-8:
            raise RuntimeError(f'no physical outgoing detonation flow at vw={vw:.9g}')
        # Weak/Jouguet detonations require $v_-\geq c_{s,-}$; see the same reference, Sec. 3.1.
        if vm*vm < float(eos_m.cs2(Tm)) - 1e-8:
            raise MatchingUnavailable(f'detonation root has subsonic outgoing flow at vw={vw:.9g}')
        return matching

    def _match_detonation_minimum(self, vw, temperature, seeds):
        """Solve field stationarity and flux matching across a minimum-selection jump."""
        eos = self.eos_m
        wp, pp = float(self.eos_p.w(self.Tn)), float(self.eos_p.p(self.Tn))

        def state(x):
            Tm, field = x
            phi = np.zeros_like(seeds[0])
            phi[eos.axis] = field
            pm, wm = eos.pressure_enthalpy(Tm, phi)
            k = wp*vw / (wm*(1.-vw*vw))
            vm = 2.*k / (1.+np.sqrt(1.+4.*k*k))
            energy = wp*_gamma2(vw)*vw*vw + pp - wm*_gamma2(vm)*vm*vm - pm
            residual = np.array([energy/wp, eos._axis_gradient(field, Tm)/self.Tn**3])
            return phi, vm, wm, residual

        matches = []
        for seed in seeds:
            solution = root(lambda x: state(x)[-1], [temperature, seed[eos.axis]],
                            tol=1e-9, options={'eps': 1e-10})
            Tm, field = solution.x
            if (not solution.success or not max(self.Tn, eos.T_min) <= Tm <= eos.T_max
                    or not 0. < field < eos.field_limit):
                continue
            phi, vm, wm, residual = state(solution.x)
            if (not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1e-8
                    or not 0. < vm <= vw or wm <= 0. or np.any(eos._curvatures(phi, Tm) <= 0.)):
                continue
            if vm*vm < eos.cs2(Tm, phi=phi)-1e-8:
                continue
            flux = self._wall_residuals(vw, vm, self.Tn, Tm, phi_m=phi)
            if np.max(np.abs(flux))/max(abs(wp), abs(wm), 1.) > 1e-8:
                continue
            matches.append(MatchingResult(vw, vw, float(vm), self.Tn, float(Tm), wp,
                                          float(wm), 'detonation', phi_m=phi.copy()))
        if not matches:
            raise MatchingUnavailable(f'no stable detonation minimum matches at vw={vw:.12g}')
        return min(matches, key=lambda matching: matching.Tm)

    def _match_front_wall_at_Tm(self, vw, Tm, vp_Tp0=None):
        cs2_m = float(self.eos_m.cs2(Tm))
        if cs2_m <= 0:
            raise RuntimeError(f"cs2_m is not physical at Tm={Tm:.6g}")
        vm = float(np.sqrt(min(vw * vw, cs2_m)))

        Tp_lo = max(self.eos_p.T_min, 0.5 * self.Tn)
        Tp_hi = min(self.eos_p.T_max, 5.0 * self.Tn)
        vp_lo = 1e-8
        vp_hi = min(0.999999, vw, float(self.eos_p.cs2(self.Tn)) / vw)
        if vp_hi <= vp_lo:
            raise RuntimeError(f"no front-wave vp interval for vw={vw:.6g}")

        seeds = []
        if vp_Tp0 is not None:
            seeds.append(np.asarray(vp_Tp0, dtype=float))
        seeds.extend([
            np.array([min(0.8 * vm, 0.5 * vp_hi), max(1.001 * self.Tn, Tm + 0.1)]),
            np.array([min(0.5 * vm, 0.5 * vp_hi), max(1.01 * self.Tn, Tm)]),
            np.array([0.5 * vp_hi, 1.05 * self.Tn]),
            np.array([0.9 * vp_hi, max(1.05 * self.Tn, Tm + 1.0)]),
        ])

        roots = []
        for index, seed in enumerate(seeds):
            seed = np.asarray(seed, dtype=float).copy()
            seed[0] = np.clip(seed[0], 10.0 * vp_lo, 0.999 * vp_hi)
            seed[1] = np.clip(seed[1], 1.001 * Tp_lo, 0.999 * Tp_hi)
            y0 = np.array([
                _map_interval(seed[0], vp_lo, vp_hi),
                _map_interval(seed[1], Tp_lo, Tp_hi),
            ])

            def unmap(y):
                return (
                    _unmap_interval(y[0], vp_lo, vp_hi),
                    _unmap_interval(y[1], Tp_lo, Tp_hi),
                )

            def matching(y):
                vp, Tp = unmap(y)
                wp = float(self.eos_p.w(Tp))
                wm = float(self.eos_m.w(Tm))
                momentum, energy = self._wall_residuals(vp, vm, Tp, Tm)
                scale = max(abs(wp), abs(wm), 1.0)
                return np.array([momentum / scale, energy / scale])

            try:
                sol = root(
                    matching,
                    y0,
                    method="hybr",
                    options={"xtol": max(self.ode_atol, 1e-8), "maxfev": 200},
                )
            except (RuntimeError, ValueError):
                continue
            vp, Tp = unmap(sol.x)
            residual = matching(sol.x)
            if np.sum(residual * residual) < 1e-12:
                if 0.0 < vp < vp_hi and Tp_lo < Tp < Tp_hi:
                    state = (float(vp), vm, float(Tp), float(Tm))
                    if index == 0 and vp_Tp0 is not None:
                        return state
                    roots.append(state)

        if not roots:
            raise RuntimeError(f"front-wall matching failed at vw={vw:.6g} Tm={Tm:.6g}")
        roots.sort(key=lambda state: state[2])
        return roots[-1]

    def _solve_def_or_hyb_by_Tm(self, vw, Tm_n=70):
        Tm_lo = max(self.eos_m.T_min, 0.5 * self.Tn)
        Tm_hi = min(self.eos_m.T_max, 2.2 * self.Tn)

        previous = self.front_match
        nearby = previous is not None and abs(vw - previous.vw) < .05
        front_seed = np.array([previous.vp, previous.Tp]) if nearby else None

        def shock_residual(Tm):
            nonlocal front_seed
            vp, vm, Tp, Tm = self._match_front_wall_at_Tm(vw, Tm, front_seed)
            shock_flux, xi_sh, v_sh, T_sh = self._solve_hydro_shock(vw, vp, Tp)
            front_seed = np.array([vp, Tp])
            return shock_flux, (vp, vm, Tp, Tm, xi_sh, v_sh, T_sh)

        def build(state):
            vp, vm, Tp, Tm, xi_sh, v_sh, T_sh = state
            eos_p = getattr(self.eos_p, "direct", self.eos_p)
            eos_m = getattr(self.eos_m, "direct", self.eos_m)
            wp = float(eos_p.w(Tp))
            wm = float(eos_m.w(Tm))
            wall_mom, wall_energy = self._wall_residuals(vp, vm, Tp, Tm, eos_p=eos_p, eos_m=eos_m)
            wall_scale = max(abs(wp), abs(wm), 1.0)
            wall_error = max(abs(wall_mom), abs(wall_energy)) / wall_scale
            if not np.isfinite(wall_error) or wall_error > 1e-6:
                raise RuntimeError(
                    f"front-wall matching residual too large at vw={vw:.6g}: "
                    f"{wall_error:.3e}"
                )
            shock_fluxes = self._wall_residuals(
                xi_sh, _mu(xi_sh, v_sh), self.Tn, T_sh, eos_p=eos_p, eos_m=eos_p,
            )
            shock_scale = max(abs(self.eos_p.w(self.Tn)), abs(self.eos_p.w(T_sh)), 1.0)
            shock_error = max(map(abs, shock_fluxes)) / shock_scale
            if not np.isfinite(shock_error) or shock_error > 1e-6:
                raise RuntimeError(
                    f"shock matching residual too large at vw={vw:.6g}: "
                    f"{shock_error:.3e}"
                )
            matching = MatchingResult(
                vw=vw,
                vp=vp,
                vm=vm,
                Tp=Tp,
                Tm=Tm,
                wp=wp,
                wm=wm,
                mode=self._front_wave_mode(vw, Tm),
                xi_sh=xi_sh,
                v_sh=v_sh,
                T_sh=T_sh,
            )
            self.front_match = matching
            return matching

        # Continue the nearby wall state, with a temperature-grid fallback.
        def residual_value(Tm):
            residual, _ = shock_residual(Tm)
            return residual

        Tm0 = float(np.clip(previous.Tm if nearby else self.Tn, Tm_lo * 1.001, Tm_hi * 0.999))
        Tm1 = float(np.clip(1.001 * Tm0, Tm_lo * 1.001, Tm_hi * 0.999))
        try:
            sol = root_scalar(
                residual_value, method="secant", x0=Tm0, x1=Tm1,
                xtol=max(self.ode_atol, 1e-7), rtol=self.ode_rtol, maxiter=40,
            )
            if sol.converged and Tm_lo < sol.root < Tm_hi:
                residual, state = shock_residual(sol.root)
                if abs(residual) < 1e-6:
                    return build(state)
        except (RuntimeError, ValueError) as error:
            last_error = error
        else:
            last_error = "secant search did not converge to a flux-conserving state"

        # Fallback: robust coarse grid scan over the full Tm range, bracketing
        # the first sign change and refining with brentq.
        front_seed = None
        samples = []
        for Tm in np.linspace(Tm_lo, Tm_hi, Tm_n):
            try:
                residual, state = shock_residual(Tm)
            except (RuntimeError, ValueError) as error:
                last_error = error
                samples = []
                front_seed = None
                continue
            if not np.isfinite(residual):
                last_error = f"nonfinite shock residual at Tm={Tm:.6g}"
                samples = []
                front_seed = None
                continue
            samples.append((float(Tm), float(residual), state))
            if len(samples) < 2:
                continue
            before = samples[-2]
            after = samples[-1]
            if before[1] * after[1] > 0:
                continue

            front_seed = np.array([before[2][0], before[2][2]])
            Tm_root = brentq(
                residual_value,
                before[0],
                after[0],
                xtol=max(self.ode_atol, 1e-7),
                rtol=self.ode_rtol,
            )
            _, state = shock_residual(Tm_root)
            return build(state)

        raise MatchingUnavailable(
            f"front-wave Tm shooting did not bracket a solution at vw={vw:.6g}: {last_error}"
        )

    def _shock_de_v(self, v, xi_T):
        xi, T = xi_T
        cs2 = float(self.eos_p.cs2(T))
        mu = _mu(xi, v)
        dxi_dv = _gamma2(v) * (1.0 - v * xi) * (mu * mu / cs2 - 1.0)
        dxi_dv *= xi / (2.0 * v)
        dT_dv = T * _gamma2(v) * mu
        return [dxi_dv, dT_dv]

    def _solve_hydro_shock(self, vw, vp, Tp):
        pn = float(self.eos_p.p(self.Tn))
        en = float(self.eos_p.e(self.Tn))
        wn = float(self.eos_p.w(self.Tn))

        def shock(v, xi_T):
            xi, T = xi_T
            if abs(T - self.Tn) <= 1e-5 * self.Tn:
                # Midpoint dp/de gives the weak-shock limit without cancellation.
                jump_ratio = float(self.eos_p.cs2(0.5 * (T + self.Tn)))
            else:
                jump_ratio = (float(self.eos_p.p(T)) - pn) / (float(self.eos_p.e(T)) - en)
            return _mu(xi, v) * xi - jump_ratio

        shock.terminal = True
        vp_center = _mu(vw, vp)
        xi_sh = vw
        T_sh = Tp
        v_sh = vp_center
        if shock(vp_center, [vw, Tp]) > 0:
            pass
        elif abs(vw - vp) < 1e-14:
            v_sh = 0.0
            xi_sh = float(np.sqrt(self.eos_p.cs2(Tp)))
            T_sh = Tp
        else:
            sol = solve_ivp(
                self._shock_de_v, [vp_center, min(1e-8, 1e-6 * vp_center * vw**2)], [vw, Tp],
                events=shock, rtol=self.ode_rtol, atol=self.ode_atol,
            )
            if not sol.success:
                raise MatchingUnavailable(f"shock integration failed: {sol.message}")
            v_sh = float(sol.t[-1])
            xi_sh, T_sh = float(sol.y[0, -1]), float(sol.y[1, -1])

        mu_sh = _mu(xi_sh, v_sh)

        residual = (
            wn * xi_sh / (1.0 - xi_sh * xi_sh)
            - float(self.eos_p.w(T_sh)) * mu_sh * _gamma2(mu_sh)
        ) / wn
        return float(residual), float(xi_sh), float(v_sh), float(T_sh)

    # ------------------------------------------------------------- mode boundary

    def _det_peak_residual(self, vw, Tm_lo=None, Tm_hi=None):
        """Max over T_- of the detonation matching residual. Tangent crossing
        at v_w = v_J gives peak = 0; below v_J peak < 0; above v_J peak > 0
        (two roots — weak and strong detonation)."""
        eos_p = getattr(self.eos_p, "direct", self.eos_p)
        eos_m = getattr(self.eos_m, "direct", self.eos_m)
        wp_T = float(eos_p.w(self.Tn))
        pp_T = float(eos_p.p(self.Tn))
        vp = vw

        def neg_r(Tm):
            wm_T = float(eos_m.w(Tm))
            pm_T = float(eos_m.p(Tm))
            K = wp_T * vp / (wm_T * (1.0 - vp * vp))
            if K <= 0:
                return 1e30
            vm = 2.0 * K / (1.0 + np.sqrt(1.0 + 4.0 * K * K))
            if not (0 < vm < 1):
                return 1e30
            r = (wp_T * _gamma2(vp) * vp * vp + pp_T) - (
                wm_T * _gamma2(vm) * vm * vm + pm_T
            )
            return -r/wp_T

        if Tm_lo is None: Tm_lo = max(self.Tn, self.eos_m.T_min)
        if Tm_hi is None: Tm_hi = self.eos_m.T_max
        res = minimize_scalar(
            neg_r, bounds=(Tm_lo, Tm_hi), method="bounded",
            options={"xatol": 1e-6},
        )
        return -res.fun

    def find_vJ(self, dvw=0.01):
        """
        Branch-merge Jouguet velocity: where the weak and strong detonation
        roots of the matching equation merge into a single tangent root
        (peak residual = 0). Rigorous general-EOS definition.

        Walks v_w downward from 1 until the peak residual changes sign.
        """
        if self.vJ is not None:
            return self.vJ
        if self._vJ_direct is not None:
            self.vJ = self._vJ_direct
            return self.vJ

        vw_prev = 0.999
        if self._det_peak_residual(vw_prev) <= 0:
            raise RuntimeError("peak residual is already negative at vw=0.999")

        vw = max(vw_prev - dvw, self._cs_plus_Tn + 1e-6)
        while vw < vw_prev:
            pr_cur = self._det_peak_residual(vw)
            if pr_cur <= 0:
                self.vJ = float(brentq(
                    self._det_peak_residual, vw, vw_prev, xtol=1e-9,
                ))
                return self.vJ
            vw_prev = vw
            vw = max(vw - dvw, self._cs_plus_Tn + 1e-6)
        raise RuntimeError("v_J not bracketed walking down from 0.999")

    # ------------------------------------------------------------ public solve

    def solve(self, vw):
        """Dispatch by mode and return MatchingResult.

        Uses the merge Jouguet velocity as the detonation/hybrid boundary.
        """
        cs_p = self._cs_plus_Tn
        if vw < cs_p:
            return self._solve_def_or_hyb_by_Tm(vw)
        vJ = self.find_vJ()
        if vw > vJ:
            return self._solve_detonation(vw)
        return self._solve_def_or_hyb_by_Tm(vw)
