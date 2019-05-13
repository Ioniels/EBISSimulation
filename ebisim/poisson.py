import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt

from numpy import linspace
from scipy.integrate import solve_ivp
from scipy.integrate import simps
from progressbar import ProgressBar, Counter, Timer, AdaptiveETA, Percentage, Bar, FileTransferSpeed

from .plotting import _decorate_axes, plot_generic_evolution
from ebisim import elements
from .densitydists import *
from ebisim import beams
from ebisim.problems import ComplexEBISProblem
from ebisim.physconst import EPS_0, M_E_EV, C_L


@nb.jit
def electron_velocity(e_kin):
    """
    Returns the electron velocity [m/s] corresponding to a kinetic energy.

    Input Parameter:
    e_kin - electron energy [eV]
    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)


class PoissonSolver:
    """
    Class used for solving the Poisson equation for different charge density distributions.

    The problem is defined in cylindrical coordinates, i.e. no dependence in (theta, z).
    """

    def __init__(self, element, cur, e_kin, nb_p=10000):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        element - Identifier of the element under investigation
        cur - electron current [A]
        e_kin - electron energy [eV]
        """
        # Model variable which defines the electron and ion distributions
        self._default_model = ['boltzmann', 'normal']
        self._msg_default_model = 'Warning! Distribution models are set to default: ' + str(self._default_model)
        self._model = None
        # Solver initial conditions
        self._ic = (0, 0)
        self._jac = np.zeros(2)
        self._solution = None
        self._df_NkT = None
        self._df_parameters = None
        self._df_potential = None
        self._no_ions = False
        # Information describing input ion and electron distributions.
        self._NkT = 0
        self._cur = cur
        self._e_kin = e_kin
        self._element = elements.cast_to_ChemicalElement(element)
        self._rexebeam = beams.RexElectronBeam(self._cur)
        self._Q_e = cur / electron_velocity(e_kin)
        self._re = self._rexebeam.herrmann_radius(self._e_kin)
        self._rd = self._rexebeam._r_d
        self._l_trap = 0.8  # REXEBIS effective trap length [m], used for ionic density normalization
        self._r_eval = linspace(0, self._rd, nb_p)
        self._ud = 800
        # Plotting hard coded help
        self._title = self._element.symbol + '$^{i+}$, I$_e$ [A] =' + str(self._cur) \
                      + ', E$_e^{kin}$ [eV] =' + str(self.e_kin) + ', r$_e$ / r$_t$ = {:.2}'.format(self._re / self._rd)

    @property
    def e_kin(self):
        """Returns the kinetic energy"""
        return self._e_kin

    @e_kin.setter
    def e_kin(self, val):
        """Set e_kin to new value and delete existing solution"""
        if val != self._e_kin:
            self._solution = None
            self._e_kin = val

    @property
    def solution(self):
        """Returns the solution of the solved problem [tuple (phi, dphi/dr)]"""
        return self._solution

    @property
    def df_NkT(self):
        """Returns density-energy data-frame of concatenated array [np.ndarray [m-3, ev]]"""
        return self._df_NkT

    @property
    def df_parameters(self):
        """Pandas data-frame built to visualize successive important physics parameters"""
        return self._df_parameters

    @property
    def df_potential(self):
        """Pandas data-frame built to visualize successive potential solution [np.ndarray in Volt]"""
        return self._df_potential

    def reset_df(self):
        """Used to reset all pandas data-frames"""
        self._df_parameters = pd.DataFrame({'electron': [], 'ion': [], 're': [], 'phi_well': [], 'rho_e_y0': [],
                                            'rho_i_y0': [], 'alpha_rh': [], 'alpha_dt': [], 'Q_i_rh': [],  'Q_i_dt': [],
                                            'N_e_rh': [], 'N_e_dt': []})
        self._df_potential = pd.DataFrame({'r': self._r_eval})
        self._df_NkT = pd.DataFrame()

    @property
    def model(self):
        """Returns the model of distributions"""
        return self._model

    @model.setter
    def model(self, val):
        """Set model to new value while checking on the availability of the value"""
        if val == self._model:
            pass
        elif not isinstance(val, list):
            print('Warning! Type of the input model is not recognized...')
            print(self._msg_default_model)
            self._model = self._default_model
        elif not isinstance(val[0], str) or not isinstance(val[1], str):
            print('Warning! At least one distribution in the model is not given as a string...')
            print(self._msg_default_model)
            self._model = self._default_model
        elif 'n_i_' + val[0].lower() + '_s' not in globals() or 'n_e_' + val[1].lower() + '_s' not in globals():
            print('Warning! At least one distribution in the model is not defined...')
            print(self._msg_default_model)
            self._model = self._default_model
        else:
            self._solution = None
            self._model = [x.lower() for x in val]

    def rhs(self, r, y):
        """
        Returns a 2-element array, symbolic in y and r, with derivatives of the differential problem.
        Has to force initial conditions as the general term is undefined in r = 0.

        Input Parameters:
        r - radius
        y - potential

        """
        n_i = n_i_pb(model_n_i=self.model[0], no_ions=self._no_ions, y=abs(y[0]), NkT=self._NkT, Z=self._element.z)
        n_e = n_e_pb(model_n_e=self.model[1], r=r, Qe=self._Q_e, re=self._re)
        if r == 0:
            self._jac = np.array([[0, 1], [-(n_e[1] + n_i[1]) / EPS_0, 0]])
            return np.array(self._ic)
        else:
            self._jac = np.array([[0, 1], [-(n_e[1] + n_i[1]) / EPS_0, -1 / r]])
            return np.array([y[1], -(n_e[0] + n_i[0]) / EPS_0 - y[1] / r])
            # Another way:
            # return np.array([y[1] / r, -r * rho_tot / EPS_0])

    def jac(self, r, y):
        """

        """
        return self._jac

    def solve(self, NkT, model=None, frame_para=False, frame_phi=False):
        """
        Solves Poisson equation

        Input Parameters:
        model - 2-element string list defining the ion charge distribution and the electron distribution

        """
        self._NkT = NkT
        if model: self.model = model
        solution = solve_ivp(self.rhs, (0, self._rd), self._ic, jac=self.jac, t_eval=self._r_eval, method='Radau')
        # Faster option:
        #solution = solve_ivp(self.rhs, (0, self._rd), self._ic, eval=self._r_eval, method='LSODA')
        self._solution = solution

        if frame_para or frame_phi: self.frame_sol(save_para=frame_para, save_phi=frame_phi)
        return solution

    def solve_e(self):
        """
        Solves Poisson equation with initial conditions y(r=0) and dy/dr(r=0) = 0
        Only for electron beam distribution. The last model solved output is not updated.

        Input Parameters
        r - radial position array [m], needs to start at r = 0 to have valid initial conditions
        model - str describing the electron beam distribution

        """
        self._no_ions = True
        solution = solve_ivp(self.rhs, (0, self._rd), self._ic, t_eval=self._r_eval, method='Radau')
        self._no_ions = False
        return solution

    def frame_sol(self, save_para=True, save_phi=True):
        """
        Save in pandas' data-frames important results from the previously solved problem.


        """
        if self.solution is None:
            print("Error! Need to solve problem before framing any solution")
        if save_para:
            if self._df_parameters is None:
                self.reset_df()
            idx = [self._df_parameters.shape[0]]
            Z = self._element.z
            r = self.solution.t
            re = r[r <= self._re]
            re_err, re_idx = re[-1] / self._re, len(re)
            phi_re = self.solution.y[0][re_idx]
            n_i = n_i_pb(model_n_i=self.model[0], y=self.solution.y[0], NkT=self._NkT, Z=Z)
            n_e = n_e_pb(model_n_e=self.model[1], r=r, Qe=self._Q_e, re=self._re)
            rho_e_y0, rho_i_y0 = n_e[0], n_i[0, -1]
            N_e_rh = simps(-2 * PI * re * n_e[:re_idx] / Q_E, re)
            N_e_dt = simps(-2 * PI * r * n_e / Q_E, r)
            Q_i_dt = simps(2 * PI * r * n_i[:, -1], r)
            alpha_dt = 100 * Q_i_dt / (N_e_dt * Q_E)
            Q_i_rh = simps(2 * PI * re * n_i[:re_idx, -1], re)
            alpha_rh = 100 * Q_i_rh / (N_e_rh * Q_E)
            df_para = pd.DataFrame({'electron': [self.model[1]], 'ion': [self.model[0]], 're': [self._re],
                                    'phi_well': [phi_re], 'rho_e_y0': [rho_e_y0], 'rho_i_y0': [rho_i_y0],
                                    'alpha_rh': [alpha_rh], 'alpha_dt': [alpha_dt], 'Q_i_rh': [Q_i_rh],
                                    'Q_i_dt': [Q_i_dt], 'N_e_rh': [N_e_rh], 'N_e_dt': [N_e_dt]}, index=idx, dtype=float)
            self._df_parameters = self._df_parameters.append(df_para)

        if save_phi:
            if self._df_potential is None:
                self.reset_df()
            idx = [self._df_potential.shape[1]]
            self._df_potential = self._df_potential.join(pd.DataFrame({idx[0]: self.solution.y[0]}))

        return

    def solve_dyn(self, model=None, reset_df=True):
        """
        Solve ComplexEBISProblem with input species, and electron beam parameters.
        To do: dynamically change e_kin, r_e, e_fwhm(?), dynamically input overlap factors

        """
        if model: self.model = model
        if reset_df: self.reset_df()

        # Dynamical density evolution problem: to be built
        j = electron_velocity(self.e_kin) * n_e_pb(model_n_e=self._model[1], r=0, Qe=self._Q_e, re=self._re)[0]
        j = abs(j) * 1e-4  # convert to A/cm**2
        problem = ComplexEBISProblem(self._element.symbol, j, self.e_kin, 15)
        _ = problem.solve(0.1, method="BDF")
        NkT = problem.solution.y[:, ::20]
        t = problem.solution.t[::20]
        NkT[0, :] = 0
        self._df_NkT = NkT
        # Start iteration for space-charge problem
        widgets = [Percentage(), ' ', Bar(), ' ', Counter(), ' / ', str(len(NkT[1, :])), ' | ', Timer(),
                   ' | ', AdaptiveETA(), ' | ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=len(NkT[1, :]))
        pbar.start()
        for i in pbar(range(len(NkT[1, :]))):
            _ = self.solve(NkT[:, i], model=model, frame_para=True, frame_phi=True)
            pbar.update(i + 1)
        pbar.finish()
        self._df_parameters = self._df_parameters.join(pd.DataFrame({'t': t}))

        return

    def plot_sol_potentials(self):
        """
        Plotting of the lastly solved potential solution, considering the different contributions (electron, ion).

        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        # Space-charge potential due to electrons and ions (the solution of the last problem solved)
        phi, phi_p = self.solution.y[0] + self._ud - self.solution.y[0][-1], self.solution.y[1]
        r = self.solution.t / self._rd
        # Space-charge potential due to electrons
        sol_e = self.solve_e()
        phi_e, phi_e_p = sol_e.y[0] + self._ud - sol_e.y[0][-1], sol_e.y[1]
        r_e = sol_e.t / self._rd
        # Space-charge potential due to ions
        sz_max, sz_min = max([phi_e.size, phi.size]), min([phi_e.size, phi.size])
        phi_i, r_i = np.zeros(sz_max), np.zeros(sz_max)
        phi_i[:sz_min], r_i[:sz_min] = phi[:sz_min] - phi_e[:sz_min], r[:sz_min]

        fig, ax = plt.subplots()
        ax.plot(r, phi, '-', c='k', label='Total')
        ax.plot(r, phi_p, '-', c='k', label='d_Total')
        ax.plot(r_e, phi_e, ':', c='b', label='Electrons: ' + str(self.model[1]))
        ax.plot(r_i, phi_i, '--', c='r', label='Ions: ' + str(self.model[0]))
        x_lim, y_lim = (0, 1), (0, max(phi))
        _decorate_axes(ax, title=self._title, xlabel=r'Relative radius $\frac{r}{r_{DT}}$', ylabel='Potential [V]',
                       xlim=x_lim, ylim=y_lim, grid=True)
        return plt

    def plot_sol_densities(self):
        """
        Plots the total ion and electron charge distributions inside the drift tube.

        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        r = self.solution.t
        n_i = n_i_pb(model_n_i=self.model[0], y=self.solution.y[0], NkT=self._NkT, Z=self._element.z)
        n_e = n_e_pb(model_n_e=self.model[1], r=r, Qe=self._Q_e, re=self._re)
        # Normalization of results for plotting and all charge density distributions are set positive
        np.divide(r, self._rd, r)
        np.divide(n_i, abs(n_e[0]), n_i)
        np.divide(n_e, n_e[0], n_e)

        fig, ax = plt.subplots()
        for i in range(n_i.shape[1]):
            ax.plot(r, n_i[:, i], '--', c='b', label=r'Ions: q=' + str(i))
        ax.plot(r, n_e, ':', c='r', label=r'Electrons: $\left|n_e\right|$')
        ax.plot(r, n_e - n_i[:, -1], '-', c='g', label='$\delta n_{e, i}$')
        x_lim, y_lim = (0, 0.1), (0, 1)
        _decorate_axes(ax, title=self._title, xlabel=r'Relative radius $\frac{r}{r_{DT}}$',
                       ylabel=r'Relative density $\left|\frac{n_{e, i}}{n_e^0}\right|$',
                       label_lines=True, legend=False, xlim=x_lim, ylim=y_lim, grid=True)
        return plt

    def plot_dyn_densities(self):
        """
        Plots the dynamical evolution of the different density distributions (electron, ion).

        """
        if self.df_potential is None or self._df_parameters is None:
            print("Error! Need to frame several results for dynamical plotting")
        parameters = self._df_parameters
        phi = self.df_potential
        NkT = self._df_NkT
        r = phi['r']
        t = parameters['t']
        re = r[r <= self._re]
        re_err, re_idx = re[re.size - 1] / self._re, len(re)
        Z = self._element.z
        N_ti_dt = np.zeros((Z + 1, t.size))
        N_ti_rh = np.zeros((Z + 1, t.size))
        for t_i in range(t.size):
            n_i = n_i_pb(model_n_i=parameters['ion'][t_i], y=np.array(self.df_potential[t_i + 1]),
                         NkT=NkT[:, t_i], Z=Z)
            q_threshold = np.arange(Z + 1)[(NkT[:Z + 1, t_i] > MINIMAL_DENSITY) & (NkT[Z + 1:, t_i] > MINIMAL_KBT)]
            N_i_dt, N_i_rh = np.zeros(Z + 1), np.zeros(Z + 1)
            for i in q_threshold:
                N_i_dt[i] += simps(2 * PI * r * n_i[:, i] / Q_E / i, r)
                N_i_rh[i] += simps(2 * PI * re * n_i[:re_idx, i] / Q_E / i, re)
            N_ti_dt[:, t_i] += N_i_dt[:] / parameters['N_e_dt'][t_i]
            N_ti_rh[:, t_i] += N_i_rh[:] / parameters['N_e_rh'][t_i]

        fig_1 = plot_generic_evolution(t, N_ti_dt, xlim=(1e-6, 1e-2), ylim=(1e-29, 1),
                                                ylabel='Relative integrated densities to drift tube',
                                                title=self._title, xscale="log", yscale="log", legend=False,
                                                label_lines=True, plot_sum=False)
        fig_2 = plot_generic_evolution(t, N_ti_rh, xlim=(1e-6, 1e-2), ylim=(1e-29, 1),
                                            ylabel='Relative integrated densities to Hermann radius',
                                            title=self._title, xscale="log", yscale="log", legend=False,
                                            label_lines=True, plot_sum=False)
        return plt

