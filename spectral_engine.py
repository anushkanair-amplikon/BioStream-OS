import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid

class SpectralEngine:
    def _init_(self, window_length=21, polyorder=3):
        self.window_length = window_length
        self.polyorder = polyorder

    def baseline_correction(self, y):
        """Removes baseline drift from the chromatogram."""
        baseline = np.percentile(y, 5)
        return np.maximum(y - baseline, 0)

    def process_chromatogram(self, rt, signal, prominence=10):
        """Smooths signal, detects peaks, and calculates AUC."""
        # Smooth and correct
        y_smooth = savgol_filter(signal, self.window_length, self.polyorder)
        y_corrected = self.baseline_correction(y_smooth)
        
        # Detect peaks and peak widths
        peaks, properties = find_peaks(y_corrected, prominence=prominence, width=3, rel_height=0.5)
        
        peak_data = []
        for i, peak_idx in enumerate(peaks):
            # Integrate Area Under Curve (AUC) for quantitation
            left = int(properties['left_ips'][i])
            right = int(properties['right_ips'][i])
            
            auc = trapezoid(y_corrected[left:right], x=rt[left:right])
            
            peak_data.append({
                'Peak_ID': f"P{i+1}",
                'Retention_Time': round(rt[peak_idx], 3),
                'Intensity': round(y_corrected[peak_idx], 2),
                'AUC': round(auc, 3)
            })
            
        return y_corrected, peaks, pd.DataFrame(peak_data)