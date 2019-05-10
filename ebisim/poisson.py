import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt

from numpy import linspace
from scipy.integrate import solve_ivp
from scipy.integrate import quad, simps

from .plotting import _decorate_axes
from ebisim import elements
from .densitydists import *
from ebisim import beams
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
        self._msg_default_model = 'Warning: set electron model to default distribution ' + str(self._default_model)
        self._model = self._msg_default_model
        # Solver initial conditions
        self._ic = (0, 0)
        self._jac = np.zeros(2)
        self._solution = None
        self._sol_df_densities = None
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
        self._color = plt.get_cmap('inferno').colors
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
        """Returns the solution of the lates solve of this problem"""
        return self._solution

    @property
    def sol_df_densities(self):
        """Returns the solution of the lates solve of this problem"""
        return self._sol_df_densities

    def reset_df(self):
        self._sol_df_densities = pd.DataFrame({r'$n_e^0$': [], r'$n_i^0$': [], r'$N_edt$': [],
                               r'$N_idt$': [], r'$N_erh$': [], r'$N_irh$': []})

    @property
    def model(self):
        """Returns the model of distributions"""
        return self._model

    @model.setter
    def model(self, val):
        """Set model to new value and delete existing solution"""
        if val == self._model:
            pass
        elif not isinstance(val, list):
            print('Warning: type of the input model is not recognized...')
            print(self._msg_default_model)
            self._model = self._default_model
        elif not isinstance(val[0], str) or not isinstance(val[1], str):
            print('Warning: at least one distribution in the model is not given as a string...')
            print(self._msg_default_model)
            self._model = self._default_model
        elif 'n_i_' + val[0].lower() + '_s' not in globals() or 'n_e_' + val[1].lower() + '_s' not in globals():
            print('Warning: at least one distribution in the model is not defined...')
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

#    @nb.jit
    def solve(self, NkT, model=None, save=True):
        """
        Solves Poisson equation

        Input Parameters:
        model - 2-element string list defining the ion charge distribution and the electron distribution

        """
        self._NkT = NkT
        self.model = model
        solution = solve_ivp(self.rhs, (0, self._rd), self._ic, jac=self.jac, t_eval=self._r_eval, method='Radau')
        # Faster option:
        #solution = solve_ivp(self.rhs, (0, self._rd), self._ic, eval=self._r_eval, method='LSODA')
        self._solution = solution
        if save is True: self.save_sol_densities()
        return solution

#    @nb.jit
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

    def save_sol_densities(self):
        """
        Save in pandas important results about the integrated densities of the last solved problem.


        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        Z = self._element.z
        r = self.solution.t
        re = r[r <= self._re]
        re_err, re_idx = re[-1] / self._re, len(re)
        phi_re = self.solution.y[0][re_idx]
        n_i = n_i_pb(model_n_i=self.model[0], y=self.solution.y[0], NkT=self._NkT, Z=Z)
        n_e = n_e_pb(model_n_e=self.model[1], r=r, Qe=self._Q_e, re=self._re)
        n_e_y0, n_i_y0 = n_e[0], n_i[0, -1]
        N_e_rh = simps(-2 * PI * re * n_e[:re_idx] / Q_E, re)
        N_e_dt = simps(-2 * PI * r * n_e / Q_E, r)
        N_i_dt, N_i_rh = np.zeros(Z + 1), np.zeros(Z + 1)
        q_threshold = np.arange(Z + 1)[(self._NkT[:Z + 1] > MINIMAL_DENSITY) & (self._NkT[Z + 1:] > MINIMAL_KBT)]
        for i in q_threshold:
            N_i_dt[i] += simps(2 * PI * r * n_i[:, i] / Q_E / i, r)
            N_i_rh[i] += simps(2 * PI * re * n_i[:re_idx, i] / Q_E / i, re)
        df = pd.DataFrame({'re': [self._re], 'phi_well': [phi_re], 'n_e_y0': [n_e_y0], 'n_i_y0': [n_i_y0],
                           'N_e_dt': [N_e_dt], 'N_i_dt': [N_i_dt], 'N_e_rh': [N_e_rh], 'N_i_rh': [N_i_rh]})
        if self._sol_df_densities is None: self.reset_df()
        self._sol_df_densities = self._sol_df_densities.append(df)
        return self._sol_df_densities

    def plot_sol_potentials(self):
        """
        Plotting of the resulting potential distributions considerent different contributions (electron, ions).

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
        ax.plot(r, phi, '-', c=self._color[0], label='Total')
        ax.plot(r, phi_p, '-', c=self._color[0], label='d_Total')
        ax.plot(r_e, phi_e, ':', c=self._color[0], label='Electrons: ' + str(self.model[1]))
        ax.plot(r_i, phi_i, '--', c=self._color[0], label='Ions: ' + str(self.model[0]))
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
            ax.plot(r, n_i[:, i], '--', c=self._color[0], label=r'Ions: q=' + str(i))
        ax.plot(r, n_e, ':', c=self._color[0], label=r'Electrons: $\left|n_e\right|$')
        ax.plot(r, n_e - n_i[:, -1], '-', c=self._color[0], label='$\delta n_{e, i}$')
        x_lim, y_lim = (0, 0.1), (0, 1)
        _decorate_axes(ax, title=self._title, xlabel=r'Relative radius $\frac{r}{r_{DT}}$',
                       ylabel=r'Relative density $\left|\frac{n_{e, i}}{n_e^0}\right|$', label_lines=True, legend=False,
                       xlim=x_lim, ylim=y_lim, grid=True)
        return plt



