import numpy as np
from numpy import linspace
from beams import *
from physconst import *
from plasma import *
from xs import *


def dict_charges():
    """
    Dictionaries of the electron and ion charge distributions.

    """
    ion_dict_keys = ["boltzmann", "maxwell3", "maxwell5", "gaussian", "null"]
    electron_dict_keys = ["normal", "gaussian"]
    ion_dict_default = {ion_dict_keys: False for ion_dict_keys in ion_dict_keys}
    electron_dict_default = {electron_dict_keys: False for electron_dict_keys in electron_dict_keys}
    return [ion_dict_default, electron_dict_default]


class ChargeDistributions:
    """
    Class used to store different charge distributions for electrons and ions.
    Returns charge density distributions evaluated in (r, y) for a specific model.

    Properties describing distributions provide the density and its derivative by the potential.
    Important: the property name of a distribution has to match the key in the dictionary.
               to keep (r, y) as input parameters for all properties of distributions.

    The problem is defined in cylindrical coordinates, so without dependence in (theta, z).
    """

    def __init__(self, element, cur, e_kin):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        element - Identifier of the element under investigation
        cur - electron current [A]
        e_kin - electron energy [eV]
        """
        # Single ion specie
        self._element = elements.cast_to_ChemicalElement(element)
        self._cur = cur
        self._NkT = np.ones(2 * (self._element.z + 1))
        self._NkT[:self._element.z + 1] *= MINIMAL_DENSITY
        self._NkT[self._element.z + 1:] *= MINIMAL_KBT
        self._NkT[2] = 2e16
        self._NkT[3] = 3e16
        self._NkT[4] = 3e16
        self._NkT[5] = 2e16
        self._NkT[6] = 1e16
        self._NkT[self._element.z + 3] = 15
        self._NkT[self._element.z + 4] = 20
        self._NkT[self._element.z + 5] = 20
        self._NkT[self._element.z + 6] = 15
        self._NkT[self._element.z + 7] = 10

        # REXEBIS electron beam - With: Herrmann radius, U_drift = 800 V
        self._e_kin = e_kin
        self._ve = electron_velocity(self._e_kin)
        self._rexebeam = RexElectronBeam(self._cur)
        self._rd = self._rexebeam._r_d
        self._re = self._rexebeam.herrmann_radius(self._e_kin)
        self._Q_e = self._cur / self._ve
        self._ud = 800

        # Default initial condition for solving the EBIS ODE System (all atoms in 1+ charge state)
        self._default_initial = np.ones(2*(self._element.z + 1))
        self._default_initial[:self._element.z + 1] *= MINIMAL_DENSITY
        self._default_initial[self._element.z + 1:] *= MINIMAL_KBT
        self._default_initial[1] = 1e16  # Density for CS 1+
        self._default_initial[self._element.z + 2] = 0.5  # Temperature for CS 1+

        self._default_model = ['boltzmann', 'normal']

    def n_i_null(self, r, y):
        """
        Dummy null distributions for the ions charge density distribution.
        Used to solve Poisson equation only with the electron beam.

        Input Parameters:
        r - Radial position [m].
        y - Potential [V].
        """
        return [0, 0]

    def n_i_boltzmann(self, r, y):
        """
        Returns the Boltzmann density distribution for the ions.
                the derivative of this distribution regarding y.
        To be reviewed if only "MINIMAL" densities should be excluded.

        Input Parameters:
        r - Radial position [m].
        y - Potential [V].
        """
        N = self._NkT[:self._element.z + 1]
        kbT = self._NkT[self._element.z + 1:]
        q_l = np.array(range(self._element.z + 1))
        mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
        q_l = q_l[mask]
        rho = sum(N[q_l[i]] * Q_E * q_l[i] / kbT[q_l[i]] * np.exp(-y * q_l[i] / kbT[q_l[i]])
                  for i in range(len(q_l)))
        rho_p = sum(- N[q_l[i]] * Q_E * q_l[i]**2 / kbT[q_l[i]]**2 * np.exp(-y * q_l[i] / kbT[q_l[i]])
                    for i in range(len(q_l)))
        return [rho, rho_p]

    def n_i_maxwell3(self, r, y):
        """
        Returns the Maxwell-Boltzmann density distribution with 3 degrees of freedom the ions.
                the derivative of this distribution regarding y.

        Input Parameters:
        r - Radial position [m].
        y - Potential [V].
        """
        N = self._NkT[:self._element.z + 1]
        kbT = self._NkT[self._element.z + 1:]
        q_l = np.array(range(self._element.z + 1))
        mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
        q_l = q_l[mask]
        rho = sum(2 * N[q_l[i]] * Q_E * q_l[i] * (np.clip(y * q_l[i] / PI, 0, None))**(1/2) *
                  (1 / kbT[q_l[i]])**(3/2) * np.exp(-y * q_l[i] / kbT[q_l[i]]) for i in range(len(q_l)))
        rho_p = sum(-2 * N[q_l[i]] * Q_E * q_l[i]**2 * (np.clip(y * q_l[i] / PI, 0, None))**(1/2) *
                    (1 / kbT[q_l[i]])**(5/2) * np.exp(-y * q_l[i] / kbT[q_l[i]]) for i in range(len(q_l)))
        return [rho, rho_p]

    def n_i_maxwell5(self, r, y):
        """
        Returns the Maxwell-Boltzmann density distribution with 5 degrees of freedom the ions.
                the derivative of this distribution regarding y.

        Input Parameters:
        r - Radial position [m].
        y - Potential [V].
        """
        N = self._NkT[:self._element.z + 1]
        kbT = self._NkT[self._element.z + 1:]
        q_l = np.array(range(self._element.z + 1))
        mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
        q_l = q_l[mask]
        rho = sum(4/3 * N[q_l[i]] * Q_E * q_l[i] * (1 / PI)**(1/2) * (np.clip(y * q_l[i], 0, None))**(3/2)
                  * (1 / kbT[q_l[i]])**(5/2) * np.exp(-y * q_l[i] / kbT[q_l[i]]) for i in range(len(q_l)))
        rho_p = sum(-4/3 * N[q_l[i]] * Q_E * q_l[i]**2 * (1 / PI)**(1/2) * (np.clip(y * q_l[i], 0, None))**(3/2) *
                  (1 / kbT[q_l[i]]) ** (7 / 2) * np.exp(-y * q_l[i] / kbT[q_l[i]]) for i in range(len(q_l)))
        return [rho, rho_p]

    def n_i_gaussian(self, r, y):
        """
        Returns the Boltzmann distribution for the ions
                the derivative of this distribution regarding y.

        Input Parameters
        r - radial position [m]
        y - potential [eV]
        """
        N = self._NkT[:self._element.z + 1]
        kbT = self._NkT[self._element.z + 1:]
        q_l = np.array(range(self._element.z + 1))
        mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
        q_l = q_l[mask]
        sig = kbT
        rho = sum(N[q_l[i]] * Q_E * q_l[i] / sig[q_l[i]] * (2 * PI)**(-1/2) *
                  np.exp(-(y / sig[q_l[i]])**2 / 2) for i in range(len(q_l)))
        rho_p = sum(-N[q_l[i]] * Q_E * q_l[i] * y * sig[q_l[i]]**-3 * (2 * PI)**(-1/2) *
                  np.exp(-(y / sig[q_l[i]])**2 / 2) for i in range(len(q_l)))
        return [rho, rho_p]

    def n_e_normal(self, r, y):
        """
        Returns the normal charge density distribution of the electron beam.
                the derivative of this distribution regarding y.

        Input Parameter:
        r - Radial position [m]
        """
        if isinstance(r, float) or isinstance(r, int):
            if r < self._re:
                rho = -self._Q_e / (PI * self._re ** 2)
            else:
                rho = 0
        else:
            rho = -self._Q_e / (PI * self._re ** 2) * np.ones(len(r))
            rho[self._re < abs(r)] = 0
        rho_p = 0
        return [rho, rho_p]

    def n_e_gaussian(self, r, y):
        """
        Returns the Gaussian charge density distribution of the electron beam.

        Input Parameter:
        r - Radial position [m]
        """
        rho = -self._Q_e / (PI * self._re ** 2) * np.exp(-(r / self._re) ** 2)
        rho_p = 0
        return [rho, rho_p]

    def dict_charges_set(self, r, y, model):
        """
        Returns ion and electron dictionaries of charge densities.
        Only the matching model to dictionary key distributions are evaluated in (r, y)

        Input Parameter:
        r - Radial position [m].
        y - Potential [V].
        """
        ion_dict = dict_charges()[0]
        electron_dict = dict_charges()[1]
        ion_dict[model[0]] = getattr(self, 'n_i_' + model[0])(r, y)
        electron_dict[model[1]] = getattr(self, 'n_e_' + model[1])(r, y)
        self._sol_ion_dict = ion_dict
        self._sol_electron_dict = electron_dict
        return [ion_dict, electron_dict]

    def verify_model(self, model):
        """
        Verifies if: the input model is a 2-element array of string.
                     the input model has two corresponding properties.
                     the input model has two corresponding dictionary keys.
        Returns default_model if none of the above is verified.

        Input Parameter:
        model - 2-element array of string.
        """
        msg_default = 'Set model to default: ' + str(self._default_model)
        if model is None:
            print(msg_default)
            model = self._default_model
        if not isinstance(model[0], str) or not isinstance(model[1], str):
            print('Distribution model is not a 2-element array of strings!')
            print(msg_default)
            model = self._default_model
        model = [model[0].lower(), model[1].lower()]
        try:
            _ = getattr(self, 'n_i_' + model[0])
            _ = getattr(self, 'n_e_' + model[1])
        except AttributeError:
            print('One or two of the input distributions has/have no property definition!')
            print(msg_default)
            model = self._default_model
        try:
            _ = dict_charges()[0][model[0]]
            _ = dict_charges()[1][model[1]]
        except KeyError:
            print('The dictionary of distributions need to be update with one or two new model inputs!')
            print('Existing electron distributions in the dictionary: ')
            ion_dict = dict_charges()[0]
            electron_dict = dict_charges()[1]
            for electron_dict in electron_dict: print(electron_dict)
            print('Existing ion distributions in the dictionary: ')
            for ion_dict in ion_dict: print(ion_dict)
            print(msg_default)
            model = self._default_model
        return model

    def ion_charge(self, r, y, model=list):
        """
        Returns the ion charge density chosen from the dictionary of distributions.

        Input Parameters:
        r - Radial position [m].
        y - Potential [V].
        """
        _ = self.dict_charges_set(r, y, model)
        key_i = next((k for k, v in self._sol_ion_dict.items() if v is not False), self._default_model[0])
        ion_charge = self._sol_ion_dict[key_i][0]
        return ion_charge

    def electron_charge(self, r, y, model=list):
        """
        Returns the electron charge density chosen from the dictionary of distributions.

        Input Parameters:
        r - Radial position [m]
        y - Potential [V].
        """
        _ = self.dict_charges_set(r, y, model)
        key_e = next((k for k, v in self._sol_electron_dict.items() if v is not False), self._default_model[1])
        electron_charge = self._sol_electron_dict[key_e][0]
        return electron_charge

    def charge_prime(self, r, y, model=list):
        """
         Returns in the derivative of the charge density distributions chosen from the dictionary.
         Only used for the Jacobian in the Poisson solver.

         Input Parameter:
         r - Radial position [m]
         """
        _ = self.dict_charges_set(r, y, model)
        key_i = next((k for k, v in self._sol_ion_dict.items() if v is not False), self._default_model[0])
        key_e = next((k for k, v in self._sol_electron_dict.items() if v is not False), self._default_model[1])
        jac_i = self._sol_ion_dict[key_i][1]
        jac_e = self._sol_electron_dict[key_e][1]
        return jac_i + jac_e


