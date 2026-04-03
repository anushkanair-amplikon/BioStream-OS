from fpdf import FPDF
import datetime

class AmplikonReport(FPDF):
    def header(self):
        """Builds the official enterprise header for every page."""
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 212, 255) # BioStream Blue
        self.cell(0, 10, "AMPLIKON BIOSYSTEMS", ln=True, align="C")
        
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "BioStream OS - Automated Executive Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        """Adds page numbers to the bottom."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def generate_ic50_report(self, results_dict, df_data):
        """Compiles the math and data into a structured document."""
        self.add_page()
        
        # Title
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, "Pharmacodynamics: Dose-Response Analysis", ln=True)
        self.ln(2)

        # Metadata
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, f"Date Generated: {datetime.date.today()}", ln=True)
        self.cell(0, 6, "Analysis Protocol: 4-Parameter Logistic (4PL) Non-Linear Regression", ln=True)
        self.ln(5)

        # Calculated Metrics Section
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, " Key Computed Metrics", ln=True, border=1, fill=True)
        
        self.set_font("Helvetica", "", 11)
        # CHANGED: Removed the • and ² characters to fix PDF encoding error
        self.cell(0, 8, f"  - Calculated IC50: {results_dict['ic50']:.4f} uM", ln=True)
        self.cell(0, 8, f"  - Hill Coefficient: {results_dict['hill']:.4f}", ln=True)
        self.cell(0, 8, f"  - Fit Confidence (R^2): {results_dict['r2']:.4f}", ln=True)
        self.ln(5)

        # Raw Data Table Section
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, " Processed Data Points", ln=True, border=1, fill=True)
        
        self.set_font("Helvetica", "B", 10)
        self.cell(95, 8, "Concentration (uM)", border=1, align="C")
        self.cell(95, 8, "Inhibition (%)", border=1, ln=True, align="C")

        self.set_font("Helvetica", "", 10)
        for idx, row in df_data.iterrows():
            self.cell(95, 8, f"{row['Concentration_uM']:.4f}", border=1, align="C")
            self.cell(95, 8, f"{row['Inhibition']:.2f}", border=1, ln=True, align="C")

        # Output as a bytearray so Streamlit can download it
        return bytes(self.output())