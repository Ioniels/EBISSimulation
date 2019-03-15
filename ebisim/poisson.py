import numpy as np
import matplotlib.pyplot as plt

from numpy import linspace
from scipy.integrate import solve_ivp
from functools import partial

from . import elements
from . import distributions
from . import beams
from .physconst import EPS_0

class PoissonSolver:
    """
    Class used for solving the Poisson equation for different charge density distributions.

    The problem is defined in cylindrical coordinates, i.e. no dependence in (theta, z).
    """

    def __init__(self, element, cur, e_kin):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        element - Identifier of the element under investigation
        cur - electron current [A]
        e_kin - electron energy [eV]
        """
        # Single ion specie and electron beam charge densities
        self._element = elements.cast_to_ChemicalElement(element)
        self._cur = cur
        self._e_kin = e_kin
        # Some REXEBIS parameters for plotting
        self._rexebeam = beams.RexElectronBeam(self._cur)
        self._re = self._rexebeam.herrmann_radius(self._e_kin)
        self._rd = self._rexebeam._r_d
        self._ud = 800
        # Plotting hard-coded r limits
        nb_p = 10000
        self._r_eval = linspace(0, self._rd, nb_p)
        msg = 'r_e / r_d = {:.2}'.format(self._re / self._rd)
        print(msg)
        self._sol_potential = None
        self._sol_charge = None

    @property
    def sol_potential(self):
        """Returns the solution of the potential in the drift region"""
        return self._sol_potential

    @property
    def sol_charge(self):
        """Returns the solution of the total charge in the drift area"""
        return self._sol_charge

    @property
    def e_kin(self):
        """Returns the kinetic energy"""
        return self._e_kin

    @e_kin.setter
    def e_kin(self, val):
        """Set e_kin to new value and delete existing solution"""
        if val != self._e_kin:
            self._sol_potential = None
            self._sol_charge = None
            self._e_kin = val

    def _rhs(self, r, y, model, ic):
        """
        Returns a 2-element array, symbolic in y and r, with derivatives of the differential problem.
        Has to force initial conditions as the general term is undefined in r = 0.

        Input Parameters:
        model - 2-element array defining the ion charge distribution and the electron distribution.
        ic - 2-element array with initial conditions U and dU/dr in r = 0

        """
        rho_tot = self._charge.ion_charge(r, y[0], model) + self._charge.electron_charge(r, y[0], model)
        if r == 0:
            return np.array(ic)
        else:
            return np.array([y[1], -rho_tot / EPS_0 - y[1] / r])

    def _jac(self, r, y, model, ic):
        """
        Returns a 2-element array, symbolic in y and r, with the Jacobian of the differential problem
        Has to force initial conditions as the general term is undefined in r = 0.

        Input Parameters:
        model - 2-element array defining the ion charge distribution and the electron distribution.
        ic - 2-element array with initial conditions U and dU/dr in r = 0

        """
        jac_21 = -self._charge.charge_prime(r, y[0], model) / EPS_0
        jac_21_0 = -self._charge.charge_prime(0, ic[0], model) / EPS_0
        if r == 0:
            jac = [[0, 1], [jac_21_0, 0]]
        else:
            jac = [[0, 1], [jac_21, -1 / r]]
        return jac

    def solve(self, NkT, model=None):
        """
        Solves Poisson equation with initial conditions y(r=0) and dy/dr(r=0) = 0
        Adds a constant to the result to match the drift tube potential
        Still to be proven this is valid with potential-dependent charge distributions (Boltzmann)

        Input Parameters
        r - radial position array [m], needs to start at r = 0 to have valid initial conditions
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        self._NkT = NkT
        self._charge = distributions.ChargeDistributions(self._element, self._cur, self._e_kin, self._NkT)
        self._model = self._charge.verify_model(model)
        ic = (0, 0)
        rhs = partial(self._rhs, model=self._model, ic=ic)
        jac = partial(self._jac, model=self._model, ic=ic)
        y = solve_ivp(rhs, (0, self._rd), ic, t_eval=self._r_eval, jac=jac, method='Radau')
        # Add a constant to match U(r = rd) = Ud
        y.y[0] = self._ud - y.y[0][-1] + y.y[0]
        self._sol_potential = y
        self._sol_charge = self._charge.ion_charge(y.t, y.y[0] - y.y[0][0], self._model) + \
                        self._charge.electron_charge(y.t, y.y[0], self._model)
        return y

    def plot_densities(self):
        """
        Plots the total ion and electron charge distributions inside the drift tube.

        Input parameter
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        if self.sol_charge is None:
            print("Error! Need to solve problem before plotting")
        rho_tot = self.sol_charge
        phi_tot = self.sol_potential
        # Normalizes the solutions with the electron charge density at r = 0
        norm = abs(self._charge.electron_charge(0, phi_tot.y[0], self._model))
        rho_tot_norm = rho_tot / norm
        rho_e_norm = self._charge.electron_charge(self._r_eval, phi_tot.y, self._model) / norm
        rho_i_norm = (rho_tot_norm - rho_e_norm)
        plt.plot(self._r_eval / self._rd, rho_i_norm,
                 '-', ms=6, mfc='w', mec='b', linewidth=2, label='n$_i$: ' + str(self._model[0]))
        plt.plot(self._r_eval / self._rd, rho_e_norm,
                 '-', ms=6, mfc='w', mec='b', linewidth=2, label='n$_e$: ' + str(self._model[1]))
        plt.plot(self._r_eval / self._rd, rho_tot_norm,
                 '-', ms=6, mfc='w', mec='b', linewidth=2, label='Total')
        plt.xlim((0, 0.1))
        plt.xlabel('r / r{$_Drift$}')
        plt.ylabel('Density / n$_e^0$')
        plt.title('Density distributions (' + self._element.symbol + '$^{i+}$, I$_e$ [A] =' + str(self._cur)
                  + ', E$_e^{kin}$ [eV] =' + str(self.e_kin))
        plt.legend()
        plt.grid()
        return plt

    def plot_densities_all(self):
        """
        Plots the all for all CS and total ion and electron charge distributions inside the drift tube.

        Input parameter
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        if self.sol_charge is None:
            print("Error! Need to solve problem before plotting")
        rho_tot = self.sol_charge
        phi_tot = self.sol_potential
        # Normalizes the solutions with the electron charge density at r = 0
        rho_e = self._charge.electron_charge(self._r_eval, phi_tot.y[0], self._model)
        rho_i_tot = (rho_tot - rho_e)
        norm = rho_i_tot[0]
        rho_i_tot_norm = rho_i_tot / norm
        for k in range(self._element.z + 1):
            NkT_q = np.zeros(2 * (self._element.z + 1))
            NkT_q[k] = self._NkT[k]
            NkT_q[k + self._element.z  + 1] = self._NkT[k + self._element.z  + 1]
            if not sum(NkT_q) == 0:
                self._charge = distributions.ChargeDistributions(self._element, self._cur, self._e_kin, NkT_q)
                rho_i_norm = self._charge.ion_charge(self._r_eval, phi_tot.y[0] - phi_tot.y[0][0], self._model) / norm
                plt.plot(self._r_eval / self._rd, rho_i_norm,
                         '-', ms=6, mfc='w', mec='b', linewidth=2, label='n$_i^{q+}$: q=' + str(k))
        plt.plot(self._r_eval / self._rd, rho_i_tot_norm ,
                 '-', ms=6, mfc='w', mec='b', linewidth=2, label='n$_i^{q+}$: q=all')
        plt.xlim((0, 0.1))
        plt.ylim((0, 1))
        plt.xlabel('r / r{$_Drift$}')
        plt.ylabel('Density / n$_i^0$')
        plt.title(str(self._model[0]))
        plt.title('Density distributions - Model: ' + str(self._model) + ', ' + self._element.symbol +
                  '$^{i+}$, I$_e$ [A] =' + str(self._cur) + ', E$_e^{kin}$ [eV] =' + str(self.e_kin))
        plt.legend()
        plt.grid()
        return plt


    def plot_potential(self):
        """
        Plots the potential distribution.

        Input parameter
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        if self.sol_potential is None:
            print("Error! Need to solve problem before plotting")
        phi_tot = self.sol_potential
        charge_pb = self.sol_charge
        model_pb = self._model
        # Potential distribution due to electron beam only
        phi_e = self.solve(self._NkT, ['null', self._model[1]])
        plt.plot(phi_tot.t / self._rd, phi_tot.y[0],
                 linewidth=2, mec='b', label='Total')
        plt.plot(phi_e.t / self._rd, phi_e.y[0],
                 linewidth=2, mec='b', label='Electrons: ' + str(model_pb[1]))
        plt.plot(phi_tot.t / self._rd, phi_tot.y[0] - phi_e.y[0],
                 linewidth=2, mec='b', label='Ions: ' + str(model_pb[0]))
        plt.xlim((0, 1))
        plt.xlabel('r / r$_{Drift}$')
        plt.ylabel('Potential [V]')
        plt.title('Potential distributions (' + self._element.symbol + '$^{i+}$, I$_e$ [A] =' + str(self._cur)
                  + ', E$_e^{kin}$ [eV] =' + str(self.e_kin))
        plt.legend()
        plt.grid()
        # Reset solutions to initial problem values
        self._sol_potential = phi_tot
        self._sol_charge = charge_pb
        self._model = model_pb
        return plt

    def plot_potential_combine(self, NkT):
        """
        Plots the potential distribution for all types of distributions.

        Input parameter
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        dict_i_charge = distributions.dict_charges()[0]
        dict_e_charge = distributions.dict_charges()[1]
        for s_e in dict_e_charge:
            for s_i in dict_i_charge:
                model = (s_i, s_e)
                _ = self.solve(NkT, model)
                plt.plot(self.sol_potential.t / self._rd, self.sol_potential.y[0],
                         linewidth=2, mec='b', label='Total Electrons - Model: ' + str(model))
        plt.xlim((0, 1))
        plt.xlabel('r / r$_{Drift}$')
        plt.ylabel('Potential [V]')
        plt.title('Potential distributions (' + self._element.symbol + '$^{i+}$, I$_e$ [A] =' + str(self._cur)
                  + ', E$_e^{kin}$ [eV] =' + str(self.e_kin))
        plt.legend()
        plt.grid()
        return plt

    def plot_densities_combine(self, NkT):
        """
        Plots the potential distribution for all types of distributions.

        Input parameter
        model - 2-element array defining the ion charge distribution and the electron distribution

        """
        dict_i_charge = distributions.dict_charges()[0]
        dict_e_charge = distributions.dict_charges()[1]
        norm = abs(self._charge.electron_charge(0, 0, ['null', 'normal']))
        for s_e in dict_e_charge:
            for s_i in dict_i_charge:
                model = (s_i, s_e)
                _ = self.solve(NkT, model)
                plt.plot(self._r_eval / self._rd, self.sol_charge / norm,
                         linewidth=2, mec='b', label='n$_{ie}$: ' + str(model))
        plt.xlim((0, 0.2))
        plt.xlabel('r / r$_{Drift}$')
        plt.ylabel('Density / n$_e^0$')
        plt.title('Total charge (' + self._element.symbol + '$^{i+}$, I$_e$ [A] =' + str(self._cur)
                  + ', E$_e^{kin}$ [eV] =' + str(self.e_kin))
        plt.legend()
        plt.grid()
        return plt
