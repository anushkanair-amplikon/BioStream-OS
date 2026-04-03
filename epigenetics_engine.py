import numpy as np
import pandas as pd

class EpigeneticAnalyzer:
    def generate_methylation_profile(self, num_loci=100):
        """Simulates high-resolution methylome data including non-CpG regions."""
        np.random.seed(42)
        positions = np.arange(10000, 10000 + (num_loci * 50), 50)
        
        # CpG islands: Bimodal (hyper/hypo-methylated)
        cpg = np.concatenate([np.random.beta(0.5, 5, num_loci//2), np.random.beta(5, 0.5, num_loci//2)])
        np.random.shuffle(cpg)
        
        # Non-CpG (CpHpG/CpHpH): Lower baseline, spikes in specific plasticity regions
        non_cpg = np.random.beta(0.5, 15, num_loci)
        non_cpg[40:50] += np.random.uniform(0.3, 0.6, 10) # Aging/plasticity loci spike
        
        return pd.DataFrame({
            'Locus': positions,
            'CpG_Beta': np.clip(cpg, 0, 1),
            'Non_CpG_Beta': np.clip(non_cpg, 0, 1)
        })