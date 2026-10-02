import os
import re
import pandas as pd
from shapely import points


class CrownGrowthModel:
    """
    Klasse om groeiparameters (Beta0, Beta1, Beta2) uit Appendix 4 van iTree NL 2.0 op te zoeken
    en de verwachte kroondiameter (CD) te berekenen.
    """
    def __init__(self, ref_excel_path=None):
        self.ref_data = {}
        self.genus_data = {}
        
        if ref_excel_path and os.path.exists(ref_excel_path):
            self.load_reference_table(ref_excel_path)
            
    def load_reference_table(self, path):
        """
        Laadt Appendix 4 in en leest op basis van kolomposities.
        Vangt lege Species/Cultivar op en koppelt de Genus (geslacht).
        """
        try:
            df = pd.read_excel(path)
            df.columns = [str(c).strip() for c in df.columns]

            for _, row in df.iterrows():
                # Genus, Species en Cultivar veilig inlezen
                genus = str(row['Genus']).strip().lower() if pd.notna(row.get('Genus')) else ""
                species = str(row['Species']).strip().lower() if pd.notna(row.get('Species')) else ""
                cultivar = str(row['Cultivar']).strip().lower() if pd.notna(row.get('Cultivar')) else ""

                # Vang β0, β1, β2 op (ongeacht de exacte naam, we gebruiken kolom 4, 5 en 6)
                def to_float(val):
                    try:
                        if pd.isna(val):
                            return 0.0
                        return float(val)
                    except:
                        return 0.0

                b0 = to_float(row.iloc[4]) # β0
                b1 = to_float(row.iloc[5]) # β1
                b2 = to_float(row.iloc[6]) # β2

                params = (b0, b1, b2)

                # 1. Opslaan onder volledige combinatie
                full_name = f"{genus} {species} {cultivar}".strip().replace("  ", " ")
                if full_name:
                    self.ref_data[full_name] = params

                # 2. Opslaan onder Genus + Species
                if genus and species:
                    self.ref_data[f"{genus} {species}"] = params

                # 3. Opslaan onder alleen Genus (bijv. 'prunus')
                if genus and genus not in self.genus_data:
                    self.genus_data[genus] = params

            print(f"Groeimodel geladen: {len(self.ref_data)} specifieke namen en {len(self.genus_data)} geslachten (Genus).")

        except Exception as e:
            print(f"FOUT bij inladen referentietabel {path}: {e}")

    def get_parameters(self, scientific_name):
        """
        Zoekt parameters op. Neemt het eerste woord (Genus) als terugvaloptie.
        """
        if not scientific_name or pd.isna(scientific_name):
            return None, None, None
            
        import re

        raw_str = str(scientific_name).strip().lower()
        
        # 1. Probeer directe match op de opgeschoonde string
        clean_exact = raw_str.replace("'", "").replace('"', '')
        if clean_exact in self.ref_data:
            return self.ref_data[clean_exact]

        # 2. Verwijder cultivars tussen quotes of haakjes
        cleaned = re.sub(r"['\"\(].*?['\"\)]", "", raw_str)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        words = cleaned.split()
        if not words:
            return None, None, None

        genus = words[0] # Bijv. 'prunus'

        # 3. Probeer Genus + Species (bijv. 'prunus avium')
        if len(words) >= 2:
            genus_species = f"{words[0]} {words[1]}"
            if genus_species in self.ref_data:
                return self.ref_data[genus_species]

        # 4. Terugval op Genus (bijv. 'prunus')
        if genus in self.genus_data:
            return self.genus_data[genus]

        return None, None, None

    def calculate_crown_diameter(self, scientific_name, age_years):
        """
        Berekent CD = beta0 + beta1*age + beta2*(age^2)
        """
        b0, b1, b2 = self.get_parameters(scientific_name)
    
        if b0 is None or age_years is None or pd.isna(age_years):
            return None
            
        try:
            age = float(age_years)
            cd = b0 + (b1 * age) + (b2 * (age ** 2))
            cd_bounded = min(max(0.0, cd), 4.0)  # Beperk kroondiameter tot max 4m
            return max(0.0, round(cd_bounded, 2))  # Kroondiameter mag niet negatief zijn
        except Exception:
            return None

    def calculate_crown_volume(self, cd, height):
        """
        Berekent het kroondiametervolume (Crown Volume) op basis van de formule:
        Crown Volume = (1/3) * π * (CD/2)^2 * H
        waarbij CD = kroondiameter en H = hoogte van de boom.
        """
        if cd is None or height is None or pd.isna(cd) or pd.isna(height):
            return None
        
        try:
            radius = cd / 2.0
            volume = (1/3) * 3.14159 * (radius ** 2) * height
            return max(0.0, round(volume, 2))  # Volume mag niet negatief zijn
        except Exception:
            return None

    def measure_crown_volume(self, points):
        """
        Meet het kroonvolume op basis van een 3D Alpha Shape.
        """
        import alphashape
        import numpy as np

        # Unieke punten filteren
        unique_points = np.unique(points, axis=0)

        if len(unique_points) < 4:
            return 0.0

        # Vaste alpha (bijv. 0.5 tot 1.5 afhankelijk van je puntendichtheid)
        # Je kunt alphashape ook zelf de optimale alpha laten zoeken via: alphashape.alphashape(unique_points)
        alpha = 0.8  

        try:
            alpha_shape = alphashape.alphashape(unique_points, alpha)

            # Controleer of het resultaat een echt 3D volume (Mesh) is
            if hasattr(alpha_shape, 'volume'):
                return max(0.0, round(alpha_shape.volume, 2))
            else:
                # Fallback: als alpha te hoog was, is de vorm opgesplitst in platte vlakken
                # We proberen het met een lossere alpha (convex hull equivalent = 0.0)
                fallback_shape = alphashape.alphashape(unique_points, 0.0)
                return max(0.0, round(fallback_shape.volume, 2))

        except Exception:
            return 0.0
        