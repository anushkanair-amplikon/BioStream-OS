import numpy as np
import pandas as pd
from scipy.integrate import odeint

class BioprocessEngine:
    def _init_(self):
        """Initializes the differential equation solver for fermentation."""
        pass

    def monod_kinetics(self, y, t, mu_max, Ks, Yxs, alpha, beta):
        """
        Defines the system of ODEs for Biomass (X), Substrate (S), and Product (P).
        """
        X, S, P = y
        
        # Prevent negative substrate
        S = max(0, S)
        
        # Monod equation for specific growth rate
        mu = mu_max * (S / (Ks + S))
        
        # Differential equations
        dXdt = mu * X
        dSdt = -(1 / Yxs) * mu * X if S > 0 else 0
        dPdt = alpha * dXdt + beta * X
        
        return [dXdt, dSdt, dPdt]

    def simulate_fermentation(self, t_max, dt, initial_conditions, params):
        """Solves the ODEs over the specified time course."""
        t = np.arange(0, t_max, dt)
        
        # Unpack parameters
        args = (params['mu_max'], params['Ks'], params['Yxs'], params['alpha'], params['beta'])
        
        # Solve ODE
        solution = odeint(self.monod_kinetics, initial_conditions, t, args=args)
        
        # Package into DataFrame
        df = pd.DataFrame({
            'Time_Hours': t,
            'Biomass_gL': solution[:, 0],
            'Substrate_gL': solution[:, 1],
            'Product_Yield': solution[:, 2]
        })
        
        return df