import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class PharmacodynamicsEngine:
    @staticmethod
    def four_param_logistic(x, top, bottom, ic50, hill_slope):
        """Standard 4PL equation for dose-response curves."""
        x = np.maximum(x, 1e-10) # Prevent division by zero
        return bottom + (top - bottom) / (1 + (x / ic50)**hill_slope)

    def fit_ic50(self, concentrations, responses):
        """Fits data to the 4PL model and returns parameters."""
        p0 = [np.max(responses), np.min(responses), np.median(concentrations), 1.0]
        bounds = ([0, -20, 0, -5], [150, 20, np.inf, 5])
        
        try:
            popt, _ = curve_fit(self.four_param_logistic, concentrations, responses, p0=p0, bounds=bounds)
            y_pred = self.four_param_logistic(concentrations, *popt)
            r2 = r2_score(responses, y_pred)
            
            return {
                "top": popt[0], "bottom": popt[1], 
                "ic50": popt[2], "hill": popt[3], "r2": r2,
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}