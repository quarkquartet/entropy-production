"""
Effective potential, nucleation, and wall velocity in the Z2 singlet model.

Model conventions: https://arxiv.org/abs/2009.14295.
Goldstone counterterm regulation: https://arxiv.org/abs/1107.3559.
Thermal resummation: https://arxiv.org/abs/hep-ph/9204216.
Phase tracing and tunneling use CosmoTransitions.
"""

import numpy as np
from numpy import linalg as la
from cosmoTransitions import generic_potential as gp
from cosmoTransitions import helper_functions as hf
from finiteT import Jb_spline, Jf_spline
from hydrodynamics import normalized_entropy_residual

entropy_residual_tolerance = 1e-10

# PDG 2025 update electroweak inputs.  Masses are pole/on-shell values in GeV;
# v is derived from G_F, and the CW renormalization scale is set at the Z pole.
GF = 1.1663788e-5
v = 1.0 / np.sqrt(np.sqrt(2.0) * GF)
v2 = v**2
mW = 80.3692
mZ = 91.1880
mZEW = mZ
mh = 125.20
mt = 172.56
g1 = 2.0 * np.sqrt(mZ**2 - mW**2) / v
g2 = 2.0 * mW / v
yt = np.sqrt(2.0) * mt / v

class model_Z2(gp.generic_potential):
    """
    Z2-symmetric real singlet scalar extension of the SM.

    2-dim field space: X = (h, S).

    Parameters:
    lh: Higgs quartic coupling.
    ls: singlet scalar self-coupling.
    lsh: portal coupling (coefficient of h^2 S^2 / 4).
    muh2: Higgs mass parameter squared (> 0, so tree-level potential has -muh2/2 h^2 form
           after rewriting; here we use the convention V0 = muh2/2 h^2 + lh/4 h^4 with muh2 < 0).
    mus2: singlet mass parameter squared (< 0 if singlet gets a VEV at h=0).

    Counterterms are computed with IR regulation for Goldstone bosons.
    """

    def init(self, param):
        self.Ndim = 2
        self.g1 = g1
        self.g2 = g2
        self.yt = yt
        self.muh2 = param["muh2"]
        self.mus2 = param["mus2"]
        self.lh = param["lh"]
        self.ls = param["ls"]
        self.lsh = param["lsh"]
        self.ms = np.sqrt(self.mus2 + 0.5 * self.lsh * v2)  # physical mass of S
        self.w0 = np.sqrt(-self.mus2 / self.ls)  # singlet VEV when h=0

        self.boson_dof = np.array([4, 2, 2, 1, 1, 3, 1, 1])
        self.boson_c = np.array([0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        self.fermion_dof = np.array([12.0])
        self.gstar_rad = 107.75
        self.gstar_in_Veff = float(np.sum(self.boson_dof) + (7.0 / 8.0) * np.sum(self.fermion_dof))
        self.g_extra_rad = self.gstar_rad - self.gstar_in_Veff

        self.Tmax = 500
        self.Tmin = 20
        self.renormScaleSq = v2

        self.CTs = None
        self.vJ = None
        self._hydro = None
        self._eos_plus = None
        self._eos_minus = None
        self._eos_transition = None
        self.EWPT = None
        self.Tc = None
        self.Tn = None
        self.set_CT_params()

    def V0(self, X):
        X = np.asanyarray(X)
        h = X[..., 0]
        S = X[..., 1]

        y = 0.5 * self.muh2 * h**2 + 0.25 * self.lh * h**4
        y += 0.5 * self.mus2 * S**2 + 0.25 * self.ls * S**4
        y += 0.25 * self.lsh * h**2 * S**2

        return y

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        T = np.asanyarray(T)
        T2 = T * T

        h, S = X[..., 0], X[..., 1]
        h2, S2 = h**2, S**2

        # Thermal correction for scalars (Parwani)
        mhT = T2 * (
            (3 * self.g2**2 + self.g1**2) / 16
            + 0.25 * self.yt**2
            + 0.5 * self.lh
            + self.lsh / 24
        )
        msT = T2 * (self.lsh / 6.0 + 0.25 * self.ls)

        # Scalar mass matrix (h, S) sector — 2x2
        mh2 = 3.0 * self.lh * h2 + self.muh2 + 0.5 * self.lsh * S2 + mhT
        ms2 = self.mus2 + 3.0 * self.ls * S2 + 0.5 * self.lsh * h2 + msT
        mhs = self.lsh * h * S

        matrix_shape = X.shape[:-1] + (2, 2)
        scalar_matrix = np.zeros(matrix_shape)
        scalar_matrix[..., 0, 0] = mh2
        scalar_matrix[..., 0, 1] = mhs
        scalar_matrix[..., 1, 0] = mhs
        scalar_matrix[..., 1, 1] = ms2

        scalar_eigen = la.eigvalsh(scalar_matrix)
        mseigen, mheigen = scalar_eigen[..., 0], scalar_eigen[..., 1]

        # Goldstone bosons (3 dof, same thermal mass as Higgs)
        mgs = self.muh2 + self.lh * h2 + 0.5 * self.lsh * S2 + mhT

        # Gauge bosons — transverse (no thermal mass)
        mW = 0.25 * self.g2**2 * h2
        mZ = 0.25 * (self.g2**2 + self.g1**2) * h2

        # Gauge bosons — longitudinal (with Debye mass)
        mWL = mW + 11.0 * self.g2**2 * T2 / 6.0

        # Z-gamma mixing matrix for longitudinal modes
        AZmatrix = np.zeros(matrix_shape)
        AZmatrix[..., 0, 0] = mWL
        AZmatrix[..., 0, 1] = 0.25 * self.g2 * self.g1 * h2
        AZmatrix[..., 1, 0] = AZmatrix[..., 0, 1]
        AZmatrix[..., 1, 1] = 0.25 * self.g1**2 * h2 + 11.0 * self.g1**2 * T2 / 6.0
        AZeigens = la.eigvalsh(AZmatrix)
        mAL, mZL = AZeigens[..., 0], AZeigens[..., 1]

        # Pack: [W_T, Z_T, W_L, Z_L, gamma_L, Goldstone, h_eigen, s_eigen]
        M = np.array([mW, mZ, mWL, mZL, mAL, mgs, mheigen, mseigen])
        M = np.rollaxis(M, 0, len(M.shape))

        return M.real + 1e-100, self.boson_dof, self.boson_c

    def fermion_massSq(self, X):
        X = np.asanyarray(X)
        h = X[..., 0]

        mt = 0.5 * self.yt**2 * h**2
        Mf = np.array([mt])
        Mf = np.rollaxis(Mf, 0, len(Mf.shape))
        return Mf, self.fermion_dof

    # ------ Thermal history ------

    def getPhases(self):
        """Trace the finite-temperature phases with CosmoTransitions."""
        from scipy.optimize import minimize
        from cosmoTransitions import transitionFinder

        if self.phases is not None:
            return self.phases

        points = [[np.array([v, 0.0]), 0.0]]
        temperatures = np.r_[
            np.linspace(0.8 * self.Tmax, 50.0, 20),
            np.linspace(0.8 * self.Tmax, 50.0, 100),
        ]
        for temperature in temperatures:
            singlet = minimize(
                lambda X: self.Vtot(X, temperature, False),
                [1.0, self.w0], method="Nelder-Mead",
            )
            if singlet.success and singlet.x[1] > 10.0 and abs(singlet.x[0]) < 10.0:
                # 10.0 is just an empirical value to determine whether this is a non-zero vev.
                points.append([singlet.x, temperature])
                break

        for attempt in range(2):
            self.phases = transitionFinder.traceMultiMin(
                self.Vtot,
                self.dgradV_dT,
                self.d2V,
                points,
                tLow=0.0,
                tHigh=self.Tmax,
                deltaX_target=100 * self.x_eps,
                forbidCrit=lambda X: np.any(np.asarray(X) < -5.0),
            )
            try:
                transitionFinder.removeRedundantPhases(
                    self.Vtot, self.phases, self.x_eps * 1e-2, self.x_eps * 10
                )
            except NotImplementedError:
                pass
            self.TcTrans = transitionFinder.findCriticalTemperatures(
                self.phases, self.Vtot
            )
            physical = []
            for transition in self.TcTrans:
                temperature = transition["Tcrit"]
                potential = self.Vtot(transition["high_vev"], temperature, False)
                if not any(
                    phase.T[0] <= temperature <= phase.T[-1]
                    and self.Vtot(phase.valAt(temperature), temperature, False)
                    < potential - 1e-6 * temperature**4
                    for phase in self.phases.values()
                ):
                    physical.append(transition)
            self.TcTrans = physical
            if attempt:
                break
            seed = self.find_mixed_phase_seed()
            if seed is None:
                break
            points.append(seed)
        return self.phases

    def find_mixed_phase_seed(self):
        """Return a deeper mixed-phase seed [fields, temperature], or None.

        Search first at critical temperatures, then inside the temperature
        intervals where singlet and Higgs phases coexist.
        """
        from scipy.optimize import minimize

        for transition in self.TcTrans:
            if (
                transition["high_vev"][0] < 0.1
                and transition["low_vev"][0] > 10.0
                and abs(transition["low_vev"][1]) < 10.0
            ):
                temperature = transition["Tcrit"]
                mixed = minimize(
                    lambda X: self.Vtot(X, temperature, False),
                    0.5 * (transition["high_vev"] + transition["low_vev"]),
                    method="Nelder-Mead",
                )
                if (
                    mixed.x[0] > 10.0
                    and mixed.x[1] > 10.0
                    and self.Vtot(mixed.x, temperature, False)
                    < self.Vtot(transition["high_vev"], temperature, False)
                ):
                    return [mixed.x, temperature]
        for high_phase in self.phases.values():
            for low_phase in self.phases.values():
                lower = max(high_phase.T[0], low_phase.T[0])
                upper = min(high_phase.T[-1], low_phase.T[-1])
                if lower >= upper:
                    continue
                temperature = lower + 0.1 * (upper - lower)
                high = high_phase.valAt(temperature)
                low = low_phase.valAt(temperature)
                if not (
                    abs(high[0]) < 10.0 and high[1] > 10.0
                    and low[0] > 10.0 and abs(low[1]) < 10.0
                ):
                    continue
                mixed = minimize(
                    lambda X: self.Vtot(X, temperature, False),
                    0.5 * (high + low), method="Nelder-Mead",
                )
                if (
                    mixed.x[0] > 10.0 and mixed.x[1] > 10.0
                    and self.Vtot(mixed.x, temperature, False)
                    < min(
                        self.Vtot(high, temperature, False),
                        self.Vtot(low, temperature, False),
                    )
                ):
                    return [mixed.x, temperature]
        return None

    def tunneling_at_T(self, T):
        """Return the CosmoTransitions tunneling result at temperature ``T``."""
        from scipy import optimize
        from cosmoTransitions import pathDeformation

        if self.EWPT is None or not self.Tmin <= T < self.Tc:
            raise ValueError("temperature is outside the transition range")

        low_phase = self.phases[self.EWPT["low_phase"]]
        high_phase = self.phases[self.EWPT["high_phase"]]
        if T < high_phase.T[0]:
            raise ValueError("false phase is unstable")
        low_guess = (
            low_phase.valAt(T)
            if low_phase.T[0] <= T <= low_phase.T[-1]
            else self.EWPT["low_vev"]
        )
        self.tunnel_low_vev = optimize.fmin(
            lambda X: self.Vtot(X, T, False),
            low_guess, disp=False,
        )
        self.tunnel_high_vev = optimize.fmin(
            lambda X: self.Vtot(X, T, False),
            high_phase.valAt(T), disp=False,
        )
        if (
            min(np.linalg.eigvalsh(self.d2V(self.tunnel_high_vev, T))) <= 0.0
            or np.linalg.norm(self.tunnel_low_vev - self.tunnel_high_vev) < 1.0
        ):
            raise ValueError("false phase is unstable")
        return pathDeformation.fullTunneling(
            np.linspace(self.tunnel_low_vev, self.tunnel_high_vev, 20),
            lambda X: self.Vtot(X, T),
            lambda X: self.gradV(X, T),
        )

    def find_Tn(self, dT=0.5, action_target=140.0, transition=None):
        """Find nucleation below Tc, using an explicit critical transition if supplied."""
        from scipy.optimize import brentq

        if transition is not None and transition is not self.EWPT:
            self.EWPT = transition
            self.Tn = self.TnTrans = None
        if self.TnTrans is not None:
            return self.TnTrans
        self.getPhases()

        if transition is None:
            for candidate in self.TcTrans:
                if candidate["high_vev"][0] < 0.1 and candidate["low_vev"][0] > 10.0:
                    self.EWPT = candidate
                    break
        if self.EWPT is None:
            return None
        self.Tc = self.EWPT["Tcrit"]

        false_phase = self.phases[self.EWPT["high_phase"]]
        dT = min(dT, (self.Tc - false_phase.T[0]) / 20.0)
        previous = None
        for temperature in np.arange(
            self.Tc - dT, max(self.Tmin, false_phase.T[0]), -dT
        ):
            try:
                tunneling = self.tunneling_at_T(temperature)
            except ValueError as error:
                if str(error) == "false phase is unstable":
                    return None
                continue
            except Exception:
                continue
            action = tunneling.action / temperature
            if previous is not None and action <= action_target < previous[1]:
                break
            previous = (temperature, action)
        else:
            return None

        self.Tn = brentq(
            lambda T: self.tunneling_at_T(T).action / T - action_target,
            temperature, previous[0], xtol=1e-7, rtol=1e-9,
        )
        tunneling = self.tunneling_at_T(self.Tn)
        self.TnTrans = {
            "Tnuc": self.Tn,
            "low_vev": self.tunnel_low_vev.copy(),
            "high_vev": self.tunnel_high_vev.copy(),
            "action": tunneling.action,
            "instanton": tunneling,
            "crit_trans": self.EWPT,
        }
        return self.TnTrans

    # ------ CW potential and counterterms ------

    def V1(self, bosons, fermions, scale=mZEW):
        """
        Method of CosmoTransitions. Overwritten.

        The 1-loop CW correction at the zero-temperature in the
        MS-bar renormalization scheme.
        """
        scale2 = scale**2
        m2, n, c = bosons
        y = np.sum(n * m2 * m2 * (np.log(m2 / scale2 + 1e-100 + 0j) - c), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n * m2 * m2 * (np.log(m2 / scale2 + 1e-100 + 0j) - c), axis=-1)
        return y.real / (64 * np.pi * np.pi)

    def V1_from_X(self, X):
        """CW potential as a function of field values, using T=0 masses."""
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions)

    def V1_without_goldstone(self, X):
        """Zero-temperature CW terms differentiated numerically for counterterms."""
        X = np.asanyarray(X, dtype=float)
        m2, dof, c = self.boson_massSq(X, 0.0)
        dof = dof.copy()
        dof[5] = 0
        return self.V1((m2, dof, c), self.fermion_massSq(X))

    def set_CT_params(self):
        """
        Compute counterterms following Eqs. 61-65 of arXiv:2009.14295.
        Goldstone derivatives are analytic, with the IR cutoff applied afterward.
        See https://arxiv.org/abs/1107.3559, Sec. IV.1.
        """
        if self.CTs is not None:
            return

        vev = np.array([v, 0.0])
        vev0 = np.array([0.0, self.w0])

        d1f = hf.gradientFunction(self.V1_without_goldstone, self.x_eps, self.Ndim)
        d2f = hf.hessianFunction(self.V1_without_goldstone, self.x_eps, self.Ndim)
        d1f_full = hf.gradientFunction(self.V1_from_X, self.x_eps, self.Ndim)

        dV1h = d1f(vev)[0]  # Goldstone tadpole vanishes at (v, 0).
        dV1s = d1f_full(vev0)[1]  # Finite full CW derivative at (0, w0).

        goldstone_factor = 3.0 / (32.0 * np.pi**2)
        ir_mass_sq = mh**2
        ir_log = np.log(ir_mass_sq / mZEW**2)
        goldstone_h = 2.0 * self.lh * v
        goldstone_hh = 2.0 * self.lh
        dV2hh = d2f(vev)[0, 0] + goldstone_factor * goldstone_h**2 * ir_log
        dV2ss = d2f(vev)[1, 1]  # Goldstone contribution vanishes at (v, 0).

        # Numerically differentiate the smooth non-Goldstone terms.
        V = self.V1_without_goldstone
        eps = 1.0
        h0, s0 = vev
        d4 = (
            V([h0 + eps, s0 + eps])
            - 2.0 * V([h0, s0 + eps])
            + V([h0 - eps, s0 + eps])
            - 2.0 * (V([h0 + eps, s0]) - 2.0 * V([h0, s0]) + V([h0 - eps, s0]))
            + V([h0 + eps, s0 - eps])
            - 2.0 * V([h0, s0 - eps])
            + V([h0 - eps, s0 - eps])
        ) / eps**4
        # At s=0: V_G,hhss = n_G/(32 pi^2) x_ss (x_hh log|x| + x_h^2/x).
        # Extend the derivative-level cutoff to 1/x for this quartic condition.
        d4 += goldstone_factor * self.lsh * (
            goldstone_hh * ir_log + goldstone_h**2 / ir_mass_sq
        )

        # Eq. 61
        dmh2 = 0.5 * dV2hh - 1.5 * dV1h / v
        # Eq. 63
        dlh = 0.5 * (dV1h / v - dV2hh) / v2
        # Eq. 62
        dms2 = -dV2ss + 0.5 * v2 * d4
        # Eq. 65
        dlsh = -d4
        # Eq. 64
        dls = -dms2 / self.w0**2 - dV1s / self.w0**3

        self.CTs = {
            "mh2": dmh2,
            "ms2": dms2,
            "lh": dlh,
            "ls": dls,
            "lsh": dlsh,
        }

    def VCT(self, X):
        """Counterterm potential."""
        X = np.asanyarray(X)
        h = X[..., 0]
        S = X[..., 1]

        y = 0.5 * self.CTs["mh2"] * h**2 + 0.25 * self.CTs["lh"] * h**4
        y += 0.5 * self.CTs["ms2"] * S**2 + 0.25 * self.CTs["ls"] * S**4
        y += 0.25 * self.CTs["lsh"] * h**2 * S**2

        return y

    def V1T(self, bosons, fermions, T, include_radiation=True):
        """One-loop thermal potential using the project-local thermal splines."""
        T = np.asanyarray(T, dtype=float)
        temperature2 = (T * T)[..., np.newaxis] + 1e-100
        temperature4 = T**4

        m2, nb, _ = bosons
        y = np.sum(nb * Jb_spline(m2 / temperature2), axis=-1)
        m2, nf = fermions
        y += np.sum(nf * Jf_spline(m2 / temperature2), axis=-1)

        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                y -= nb * np.pi**4 / 45.0
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                y -= nf * 7.0 * np.pi**4 / 360.0

        return y * temperature4 / (2.0 * np.pi**2)

    def Vtot(self, X, T, include_radiation=True):
        """Total finite-temperature effective potential."""
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)

        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)

        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.VCT(X)
        y += self.V1T(bosons, fermions, T, include_radiation)
        if include_radiation:
            # Field-independent radiation of the light spectator dof absent from
            # V_eff, so the total relativistic g_* matches the full plasma.
            y += -np.pi**2 / 90.0 * self.g_extra_rad * T**4

        return y

    def alpha_n(self):
        """Return the pseudo-trace transition strength after ``find_Tn()``.

        Uses alpha_n = [Delta V - T d(Delta V)/dT / 4] / rho_rad,
        with Delta V = V_false - V_broken and g_star = 107.75.
        """
        if self.TnTrans is None:
            raise RuntimeError("run find_Tn() before evaluating alpha_n()")
        transition = self.TnTrans
        temperature = float(transition["Tnuc"])
        false_vev = transition["high_vev"]
        broken_vev = transition["low_vev"]

        def delta_potential(T):
            return float(
                self.Vtot(false_vev, T, False)
                - self.Vtot(broken_vev, T, False)
            )

        step = 1e-5 * temperature
        derivative = (
            delta_potential(temperature - 2.0 * step)
            - 8.0 * delta_potential(temperature - step)
            + 8.0 * delta_potential(temperature + step)
            - delta_potential(temperature + 2.0 * step)
        ) / (12.0 * step)
        pseudo_trace = delta_potential(temperature) - 0.25 * temperature * derivative
        radiation_density = np.pi**2 * self.gstar_rad * temperature**4 / 30.0
        return float(pseudo_trace / radiation_density)

    # ------ Hydrodynamics ------

    def build_eos(self, transition=None, T_range_plus=None, T_range_minus=None):
        """
        Build per-phase EOS objects for the EWPT.

        Continue the singlet and Higgs minima from nucleation.
        Restrict hot-side matching to the stable Higgs endpoint.
        Sound speeds include the temperature response of the minimum.

        Returns
        -------
        (eos_plus, eos_minus) : EOSGeneral pair.
        """
        from eos import EOSAxisPhase

        if transition is None:
            transition = self.EWPT
            if transition is None:
                raise RuntimeError("no EWPT found")

        Tc = transition["Tcrit"]
        seed_plus = np.asarray(transition["high_vev"], dtype=float)
        seed_minus = np.asarray(transition["low_vev"], dtype=float)

        if T_range_plus is None:
            T_range_plus = (max(1.0, 0.5 * Tc), 3.0 * Tc)
        if T_range_minus is None:
            T_range_minus = (max(1.0, 0.05 * Tc), 1.5 * Tc)

        if not (abs(seed_plus[0]) < 1 and abs(seed_minus[1]) < 1):
            raise RuntimeError('validated axis EOS requires a singlet-to-Higgs transition')
        eos_plus = EOSAxisPhase(
            self.Vtot, seed_plus, T_min=T_range_plus[0], T_max=T_range_plus[1],
            axis=1, allow_origin=True, field_limit=max(500., 3.*self.w0),
        )
        eos_minus = EOSAxisPhase(
            self.Vtot, seed_minus, T_min=T_range_minus[0], T_max=T_range_minus[1],
            axis=0, allow_origin=False,
        )
        reference = self.TnTrans['Tnuc']
        eos_plus.anchor(reference, self.TnTrans['high_vev'])
        eos_minus.anchor(reference, self.TnTrans['low_vev'])
        eos_minus.limit_upper_temperature(reference)
        self._eos_plus = eos_plus
        self._eos_minus = eos_minus
        self._eos_transition = transition
        return eos_plus, eos_minus

    def get_hydro(self, transition=None, eos="general", template_params=None):
        """Return hydrodynamics, building the nucleation result as needed.

        template_params supplies alN, cb2, cs2, and optionally psiN.
        """
        from eos import build_template_eos, InterpolatedEOS
        from hydrodynamics import Hydrodynamics, template_jouguet_velocity

        tn_tr = self.find_Tn()
        if tn_tr is None:
            raise RuntimeError("no nucleation transition found")
        Tn = tn_tr["Tnuc"]

        if (
            eos == "general"
            and self._hydro is not None
            and self._hydro.Tn == Tn
        ):
            return self._hydro

        if eos == "template":
            eos_plus, eos_minus = build_template_eos(Tn=Tn, **template_params)
            vJ_direct = template_jouguet_velocity(template_params["alN"], template_params["cb2"])
            return Hydrodynamics(eos_plus, eos_minus, Tn, vJ_direct=vJ_direct)

        if transition is None:
            transition = tn_tr["crit_trans"]
        if self._eos_plus is None or self._eos_transition is not transition:
            self.build_eos(transition)
        self._hydro = Hydrodynamics(
            InterpolatedEOS(self._eos_plus, Tn), InterpolatedEOS(self._eos_minus, Tn), Tn
        )
        return self._hydro

    def wall_width(self):
        """
        Higgs wall width L_h from the effective potential, following the
        multi-field moment equation in Appendix A of the accompanying paper,
        specialized to field i = h.

        v = v_h is the Higgs VEV jump.
        The moment equation needs dV/dphi_h at fixed phi_h, i.e. the partial derivative
        with the singlet held at whatever value it takes there; we use the
        actual tunneling path found by the nucleation calculation (the bounce
        solution stored as tn_tr["instanton"].Phi), interpolating its singlet
        value s(phi_h) along that path.
        """
        from scipy.interpolate import interp1d
        from wall_width import wall_width as wall_width_from_dV

        tn_tr = self.find_Tn()
        if tn_tr is None:
            raise RuntimeError("no nucleation transition found")
        if "wall_width" in tn_tr:
            return tn_tr["wall_width"]

        Tn = tn_tr["Tnuc"]
        x_sym = np.asarray(tn_tr["high_vev"], dtype=float)
        x_brk = np.asarray(tn_tr["low_vev"], dtype=float)
        v = float(x_brk[0] - x_sym[0])

        path = np.vstack((x_sym, tn_tr["instanton"].Phi, x_brk))
        h_path, s_path = path[:, 0], path[:, 1]
        order = np.argsort(h_path)
        h_path, s_path = h_path[order], s_path[order]
        h_path, indices = np.unique(h_path, return_index=True)
        s_path = s_path[indices]
        s_of_h = interp1d(
            h_path, s_path, kind="linear", bounds_error=False,
            fill_value=(s_path[0], s_path[-1]),
        )

        def dVdh(phi):
            h = x_sym[0] + phi
            return self.gradV(np.array([h, float(s_of_h(h))]), Tn)[0]

        L = wall_width_from_dV(dVdh, v)
        tn_tr["wall_width"] = L
        return L

    def entropy_residual_interpolated(
        self,
        vw,
        L=None,
        tau=None,
        transition=None,
        eos="general",
        template_params=None,
        n_terms=20,
        method="integral",
        return_parts=False,
        limit="interpolated",
    ):
        """Return the dimensionless entropy residual at trial vw.

        return_parts also returns the dimensionful entropy-flux difference,
        selected entropy production, and matched hydrodynamic state.
        limit selects 'interpolated', 'lte', 'near_lte' or 'ballistic'.
        """
        hydro = self.get_hydro(
            transition=transition,
            eos=eos,
            template_params=template_params,
        )
        matching = hydro.solve(vw)
        if L is None and limit in ("interpolated", "near_lte"):
            L = self.wall_width()
        if tau is None or limit == "ballistic":
            tau = default_entropy_tau()
        entropy_flux = hydro.entropy_flux_difference(matching)
        residual, source = self._entropy_residual(
            matching, entropy_flux, hydro, L, tau, n_terms, method, limit
        )
        self.entropy_matching = matching
        self.entropy_flux_difference_value = entropy_flux
        if return_parts:
            return residual, entropy_flux, source, matching
        return residual

    def _entropy_residual(self, matching, entropy_flux, hydro, L, tau, n_terms, method, limit):
        """Evaluate the entropy source and residual for an existing matched state."""
        from entropy_production import delta_s_interpolated

        if limit == "lte":
            source, near_lte, ballistic = 0.0, None, None
        else:
            species = [
                sp for sp in self.entropy_species(matching, tau=tau, hydro=hydro)
                if sp.tau > 0.0
            ]
            source, near_lte, ballistic = delta_s_interpolated(
                species, matching, L, n_terms=n_terms, method=method,
                return_parts=True, limit=limit,
            )
        residual = normalized_entropy_residual(matching, entropy_flux, source)
        self.delta_s_near_lte_value = near_lte
        self.delta_s_ballistic_value = ballistic
        self.delta_s_interpolated_value = source if limit == "interpolated" else None
        self.delta_s_value = source
        return residual, source

    def w_transport_mass_sq(self, h, Tn):
        """Effective W/Z transport mass, with fixed asymptotic mass at Tn.

        Thermal prescription: https://arxiv.org/abs/2607.14867, Eq. (2.28).
        """
        return self.g2**2 * (0.25 * np.asarray(h)**2 + 11.0 * Tn**2 / 12.0)

    def entropy_species(self, matching, tau, hydro):
        """
        Build entropy-production species from local wall-side phase minima.

        Masses are evaluated at the local high/low minima at T_+ and T_-.
        W/Z share the effective W transport mass with its asymptotic thermal
        contribution fixed at Tn. tau maps species names to relaxation times.
        """
        from entropy_production import Species
        from eos import TemplateEOS

        if isinstance(hydro.eos_p, TemplateEOS):
            high = self.TnTrans["high_vev"]
            low = self.TnTrans["low_vev"]
        else:
            high = np.asarray(hydro.eos_p.phi_star(matching.Tp), dtype=float)
            low = matching.phi_m
            low = np.asarray(hydro.eos_m.phi_star(matching.Tm) if low is None else low, dtype=float)

        Tn = hydro.Tn
        thermal_mass_sq = float(self.w_transport_mass_sq(0.0, Tn))

        boson_names = [
            "W_T",
            "Z_T",
            "W_L",
            "Z_L",
            "gamma_L",
            "Goldstone",
            "h",
            "s",
        ]
        boson_m2_plus, boson_dof, _ = self.boson_massSq(high, 0.0)
        boson_m2_minus, _, _ = self.boson_massSq(low, 0.0)
        # The 4+2+2+1 gauge degrees of freedom form the W/Z collision species.
        boson_m2_plus[:4] = self.w_transport_mass_sq(high[0], Tn)
        boson_m2_minus[:4] = self.w_transport_mass_sq(low[0], Tn)
        boson_mass_plus = np.sqrt(np.maximum(boson_m2_plus, 0.0))
        boson_mass_minus = np.sqrt(np.maximum(boson_m2_minus, 0.0))

        species = []
        for name, mass_minus, mass_plus, dof in zip(
            boson_names, boson_mass_minus, boson_mass_plus, boson_dof
        ):
            species.append(
                Species(
                    name,
                    float(mass_minus),
                    float(dof),
                    "boson",
                    float(tau[name]),
                    mass_plus=float(mass_plus),
                    thermal_mass_sq=thermal_mass_sq if name in boson_names[:4] else 0.0,
                )
            )

        fermion_m2_plus, fermion_dof = self.fermion_massSq(high)
        fermion_m2_minus, _ = self.fermion_massSq(low)
        fermion_mass_plus = np.sqrt(np.maximum(fermion_m2_plus, 0.0))
        fermion_mass_minus = np.sqrt(np.maximum(fermion_m2_minus, 0.0))

        # Each chirality has the same mass and half the top degrees of freedom.
        for chirality in ("top_L", "top_R"):
            species.append(
                Species(
                    chirality,
                    float(fermion_mass_minus[0]),
                    0.5 * float(fermion_dof[0]),
                    "fermion",
                    float(tau[chirality]),
                    mass_plus=float(fermion_mass_plus[0]),
                )
            )

        self.entropy_species_cache = species
        return species

    def find_vJ(self, transition=None, eos="general", template_params=None):
        """Jouguet velocity for the EWPT."""
        if eos == "general" and self.vJ is not None:
            return self.vJ
        vJ = self.get_hydro(
            transition=transition,
            eos=eos,
            template_params=template_params,
        ).find_vJ()
        if eos == "general":
            self.vJ = vJ
        return vJ

    def find_detonation_solution(
        self,
        transition=None,
        eos="general",
        template_params=None,
        L=None,
        tau=None,
        n_terms=20,
        entropy_method="integral",
        bracket_n=24,
        epsilon=1e-5,
        vw_min=None,
        velocity_step=1e-3,
        max_evaluations=10000,
        limit="interpolated",
    ):
        """Classify the dimensionless entropy residual on the detonation branch up to vw=0.99.

        A negative upper residual takes priority. Otherwise refine minima
        located by a coarse scan and bracket a rising root above the minimum.
        velocity_step controls matching-gap refinement, capped at 1e-5.
        Matching failures and numerically ambiguous zeros remain unresolved.
        """
        from scipy.optimize import brentq

        self.detonation_status = "error"
        hydro = self.get_hydro(
            transition=transition, eos=eos, template_params=template_params
        )
        vJ = hydro.find_vJ()
        if eos == "general":
            self.vJ = vJ
        lo = max(vJ + epsilon, float(vw_min) + epsilon if vw_min is not None else 0.)
        hi = 0.99
        if lo >= hi:
            self.detonation_status = 'unresolved'
            return None
        if L is None and limit in ("interpolated", "near_lte"):
            L = self.wall_width()
        if tau is None:
            tau = default_entropy_tau()
        from hydrodynamics import MatchingUnavailable

        cache = {}

        class EvaluationLimit(RuntimeError):
            pass

        def residual(vw):
            key = round(float(vw), 10)
            if key not in cache:
                if len(cache) >= max_evaluations:
                    raise EvaluationLimit('detonation residual evaluation limit reached')
                try:
                    value = float(self.entropy_residual_interpolated(
                        key, L=L, tau=tau, transition=transition, eos=eos,
                        template_params=template_params, n_terms=n_terms,
                        method=entropy_method, limit=limit,
                    ))
                except MatchingUnavailable:
                    value = None
                if value is not None and not np.isfinite(value):
                    raise RuntimeError(f"nonfinite detonation residual at vw={key:.9g}")
                cache[key] = value
            if cache[key] is None:
                raise MatchingUnavailable(f"detonation matching unavailable at vw={key:.9g}")
            return cache[key]

        root = None
        status = 'unresolved'
        try:
            if residual(hi) < 0:
                status = 'runaway'
            else:
                bracket, status = self._find_detonation_bracket(
                    residual, cache, lo, hi, vJ, bracket_n, velocity_step
                )
                if bracket is not None:
                    candidate = brentq(residual, *bracket, xtol=1e-10, rtol=1e-10)
                    left = residual(max(bracket[0], candidate-1e-7))
                    right = residual(min(hi, candidate+1e-7))
                    if abs(residual(candidate)) <= entropy_residual_tolerance and left < 0 <= right:
                        root, status = float(candidate), 'det'
        except (MatchingUnavailable, EvaluationLimit):
            pass
        self.detonation_status = status
        return root

    def _find_detonation_bracket(self, residual, cache, lo, hi, vJ, bracket_n, velocity_step):
        """Locate a negative residual minimum and bracket its rising crossing."""
        from scipy.optimize import minimize_scalar
        from hydrodynamics import MatchingUnavailable

        def probe(vw):
            try:
                return residual(vw)
            except MatchingUnavailable:
                return None

        nodes = sorted(set(np.linspace(lo, hi, bracket_n).tolist() +
                           [vJ+offset for offset in [1e-4, 1e-3, 5e-3, 1e-2] if lo < vJ+offset < hi]))
        minimization_failed = False
        for vw in nodes:
            probe(vw)
        # Resolve the valid edge of each observed matching gap.
        for a, b in zip(nodes[:-1], nodes[1:]):
            fa, fb = probe(a), probe(b)
            while (fa is None) != (fb is None) and b-a > min(velocity_step, 1e-5):
                mid = .5*(a+b)
                fm = probe(mid)
                if (fa is None) == (fm is None):
                    a, fa = mid, fm
                else:
                    b, fb = mid, fm
        samples = sorted((vw, value) for vw, value in cache.items() if lo-1e-10 <= vw <= hi)
        for i, (vw, value) in enumerate(samples):
            if value is None:
                continue
            left = samples[i-1] if i else (vw, None)
            right = samples[i+1] if i+1 < len(samples) else (vw, None)
            if ((left[1] is None or value <= left[1])
                    and (right[1] is None or value <= right[1])
                    and not left[1] == value == right[1]):
                a = left[0] if left[1] is not None else vw
                b = right[0] if right[1] is not None else vw
                if a < b:
                    try:
                        minimum = minimize_scalar(residual, bounds=(a, b),
                            method='bounded', options={'xatol': 1e-8})
                        if not minimum.success:
                            minimization_failed = True
                    except MatchingUnavailable:
                        pass
        finite_samples = [(vw, value) for vw, value in cache.items()
                          if lo-1e-10 <= vw <= hi and value is not None]
        minimum_speed, minimum_value = min(finite_samples, key=lambda sample: sample[1])
        if minimum_value < -entropy_residual_tolerance:
            return (minimum_speed, hi), 'unresolved'
        if (minimum_value > entropy_residual_tolerance
                and None not in cache.values() and not minimization_failed):
            return None, 'no_root_found'
        return None, 'unresolved'

    def find_vw(
        self,
        transition=None,
        eos="general",
        template_params=None,
        method="full",
        L=None,
        tau=None,
        n_terms=20,
        entropy_method="integral",
        vw_min=0.02,
        bracket_n=24,
        limit="interpolated",
    ):
        """Find roots using the selected entropy-production limit.

        The default interpolates the near-LTE and ballistic sources.
        limit='near_lte' or 'ballistic' uses that source alone; 'lte' uses
        zero entropy production. Source calculations return
        (vw_lte, vw_selected, type). Strict LTE returns (type, vw).
        method='lte' retains the strict LTE interface; method='full'
        calculates the source selected by limit.
        """
        from scipy.optimize import brentq

        if limit == "lte":
            method = "lte"

        hydro = self.get_hydro(
            transition=transition, eos=eos, template_params=template_params
        )
        vJ = hydro.find_vJ()
        if eos == "general":
            self.vJ = vJ
        cs_plus = hydro._cs_plus_Tn
        if L is None and method == "full" and limit != "ballistic":
            L = self.wall_width()
        if method == "full" and (tau is None or limit == "ballistic"):
            tau = default_entropy_tau()
        cache = {}

        def residual(vw, full):
            key = round(float(vw), 10)
            if key not in cache:
                matching = hydro.solve(key)
                entropy_flux = hydro.entropy_flux_difference(matching)
                cache[key] = [float(entropy_flux), None, matching]
                self.entropy_matching = matching
                self.entropy_flux_difference_value = entropy_flux
            if full and cache[key][1] is None:
                matching = cache[key][2]
                value, cache[key][1] = self._entropy_residual(
                    matching, cache[key][0], hydro, L, tau, n_terms, entropy_method, limit
                )
                return value, matching
            entropy_flux, interpolated, matching = cache[key]
            value = normalized_entropy_residual(
                matching, entropy_flux, interpolated if full else 0.0
            )
            return value, matching

        def find_root(full):
            probe_value = None
            for probe in (
                vJ * 0.9995,
                vJ - 1e-3,
                vJ - 5e-3,
                vJ - 1e-2,
            ):
                try:
                    probe_value = residual(probe, full)[0]
                    break
                except (RuntimeError, ValueError):
                    pass
            if probe_value is None or probe_value > 0.0:
                intervals = [
                    (vw_min, cs_plus - 1e-5),
                    (cs_plus + 1e-5, vJ - 1e-5),
                ]
                if probe_value is None and not full:
                    intervals.append((vJ + 1e-5, 0.999))
            elif not full:
                return 1.0, "runaway"
            else:
                intervals = []

            for lo, hi in intervals:
                lo = max(lo, vw_min)
                hi = min(hi, 0.999)
                if hi <= lo:
                    continue
                previous = None
                for vw in np.linspace(lo, hi, bracket_n):
                    try:
                        value, matching = residual(vw, full)
                    except (RuntimeError, ValueError):
                        previous = None
                        continue
                    if abs(value) <= entropy_residual_tolerance:
                        return float(vw), matching.mode[:3]
                    if previous is not None and previous[1] * value < 0.0:
                        vw = brentq(
                            lambda x: residual(x, full)[0],
                            previous[0], vw, xtol=1e-6, rtol=1e-6,
                        )
                        return float(vw), residual(vw, full)[1].mode[:3]
                    previous = (vw, value)
            if full and (probe_value is None or probe_value <= 0.0):
                vw = self.find_detonation_solution(
                    transition=transition, eos=eos, template_params=template_params,
                    L=L, tau=tau, n_terms=n_terms, entropy_method=entropy_method,
                    bracket_n=bracket_n,
                    limit=limit,
                )
                if vw is not None:
                    return vw, "det"
                if self.detonation_status == "runaway":
                    return 1.0, "runaway"
            return None, None

        vw_lte, lte_type = find_root(False)
        if method == "lte":
            self.delta_s_near_lte_value = None
            self.delta_s_ballistic_value = None
            self.delta_s_interpolated_value = None
            self.delta_s_value = 0.0
            return lte_type, vw_lte
        vw_full, full_type = find_root(True)
        return vw_lte, vw_full, full_type


def default_entropy_tau():
    """Dimensionless relaxation times T*tau for gauge bosons and top quarks."""
    return {
        "W_T": 128.0,
        "Z_T": 128.0,
        "W_L": 128.0,
        "Z_L": 128.0,
        "gamma_L": 0.0,
        "Goldstone": 0.0,
        "h": 0.0,
        "s": 0.0,
        "top_L": 19.6,
        "top_R": 59.2,
    }


def make_model(mS, lsh, w0):
    """
    Create a model instance from physical parameters (mS, lambda_HS, w0).
    Derives Lagrangian parameters from physical inputs.
    """
    mus2 = mS**2 - 0.5 * lsh * v2
    ls = -mus2 / w0**2
    return make_model_ls(mS, lsh, ls)


def make_model_ls(mS, lsh, ls):
    """
    Create a model instance from (mS, lsh, ls) — ls given directly instead of w0.
    """
    lh = 0.5 * mh**2 / v2
    mus2 = mS**2 - 0.5 * lsh * v2
    muh2 = -lh * v2
    param = {"muh2": muh2, "mus2": mus2, "lh": lh, "ls": ls, "lsh": lsh}
    return model_Z2(param)
