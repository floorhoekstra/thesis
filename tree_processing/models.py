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
        self.ctat_lookup = {}  # <-- Nieuwe dictionary voor C-TAT typen
        
        if ref_excel_path and os.path.exists(ref_excel_path):
            self.load_reference_table(ref_excel_path)


    ## Referentietabel inladen
    def load_reference_table(self, path):
        """
        Laadt Appendix 4 in en leest op basis van kolomposities.
        Vangt lege Species/Cultivar op en koppelt de Genus (geslacht).
        """
        try:
            df_growth = pd.read_excel(path, sheet_name=0)
            df_growth.columns = [str(c).strip() for c in df_growth.columns]

            for _, row in df_growth.iterrows():
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

        # C-TAT lookup tabel inladen
        try:
            df_ctat = pd.read_excel(path, sheet_name=1)
            df_ctat.columns = [str(c).strip() for c in df_ctat.columns]

            for _, row in df_ctat.iterrows():
                species = str(row['Species']).strip() if pd.notna(row.get('Species')) else None
                # Zoekt kolom 'C-TAT Type' of 'C-TAT\nType' als er een enter in de kolomkop staat
                ctat_val = row.get('C-TAT Type', row.get('C-TAT\nType', None))

                if species and pd.notna(ctat_val):
                    # Sla op onder de kleine letters van de soortnaam voor een snelle lookup
                    self.ctat_lookup[species.lower()] = ctat_val

            print(f"C-TAT lookup geladen: {len(self.ctat_lookup)} typen.")

        except Exception as e:
            print(f"FOUT bij inladen C-TAT lookup tabel {path}: {e}")

    def get_ctat_type(self, scientific_name):
        """
        Zoekt het C-TAT Type op voor een wetenschappelijke naam met een fallback naar Genus.
        """
        if not scientific_name or pd.isna(scientific_name):
            return None

        sci_clean = str(scientific_name).strip().lower()

        # 1. Directe exacte match
        if sci_clean in self.ctat_lookup:
            return self.ctat_lookup[sci_clean]

        # 2. Match zonder cultivar (bijv. "Acer cappadocicum 'Rubrum'" -> "Acer cappadocicum")
        base_species = sci_clean.split("'")[0].replace('"', '').strip()
        if base_species in self.ctat_lookup:
            return self.ctat_lookup[base_species]

        # 3. Fallback op Geslacht (bijv. "Acer pseudoplatanus" -> zoekt eerste "Acer")
        genus = sci_clean.split()[0]
        for species_key, ctat_val in self.ctat_lookup.items():
            if species_key.startswith(genus):
                return ctat_val

        return None
    
    # Groeimodel en parameters
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

    # Bereken oppervlakte, volume en biomassa
    def calculate_crown_area(self, points):
        """
        Berekent de kroonoppervlakte (Crown Area) op basis van een 2D Alpha Shape.
        """
        import alphashape
        import numpy as np

        # 1. Unieke punten filteren
        unique_points = np.unique(points, axis=0)

        if len(unique_points) < 3:
            return 0.0

        # 2. CENTREER DE PUNTEN ROND HET MIDDELPUNT (0,0)
        centroid = np.mean(unique_points[:, :2], axis=0)
        centered_points = unique_points[:, :2] - centroid

        alpha = 0.8  # Vaste alpha

        try:
            # Genereer Alpha Shape op de gecentreerde punten
            alpha_shape = alphashape.alphashape(centered_points, alpha)

            # Controleer of het resultaat een echt 2D oppervlak is
            if hasattr(alpha_shape, 'area'):
                area = float(alpha_shape.area)
                if abs(area) < 500.0:  # Als het oppervlak door een instabiele mesh bizar groot is (> 500 m² is vrijwel onmogelijk voor 1 boom)
                    return abs(round(area, 2))

            # Fallback: Convex hull (alpha = 0.0) als absolute ondergrens voor stabiliteit
            convex_shape = alphashape.alphashape(centered_points, 0.0)
            return abs(round(float(convex_shape.area), 2))

        except Exception:
            return 0.0

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
        Centreert de punten eerst om precisiefouten bij grote RD-coördinaten te voorkomen.
        """
        import alphashape
        import numpy as np

        # 1. Unieke punten filteren
        unique_points = np.unique(points, axis=0)

        if len(unique_points) < 4:
            return 0.0

        # 2. CENTREER DE PUNTEN ROND HET MIDDELPUNT (0,0,0)
        # Dit voorkomt floating-point afrondfouten bij grote RD New coördinaten
        centroid = np.mean(unique_points, axis=0)
        centered_points = unique_points - centroid

        alpha = 0.8  # Vaste alpha

        try:
            # Genereer Alpha Shape op de gecentreerde punten
            alpha_shape = alphashape.alphashape(centered_points, alpha)

            # Controleer of het resultaat een echt 3D volume (Mesh) is
            if hasattr(alpha_shape, 'volume'):
                vol = float(alpha_shape.volume)
                # Als het volume door een instabiele mesh bizar groot is (> 5000 m³ is vrijwel onmogelijk voor 1 boom)
                if abs(vol) < 5000.0:
                    return abs(round(vol, 2))

            # Fallback 1: Als alpha te hoog/instabiel was, probeer een lagere, veiligere alpha (bijv. 0.2)
            fallback_alpha = alphashape.alphashape(centered_points, 0.2)
            if hasattr(fallback_alpha, 'volume'):
                vol = float(fallback_alpha.volume)
                if abs(vol) < 5000.0:
                    return abs(round(vol, 2))

            # Fallback 2: Convex hull (alpha = 0.0) als absolute ondergrens voor stabiliteit
            convex_shape = alphashape.alphashape(centered_points, 0.0)
            return abs(round(float(convex_shape.volume), 2))

        except Exception:
            return 0.0

    def calculate_agb_bermudez(self, ca, height):
        """
        Berekent AGB (Above Ground Biomass) op basis van de formule:
        AGB = e^{alpha} * (CA * H)^{beta} like Bermudez et al., 2026
        waarbij CA = kroonoppervlakte en H = hoogte van de boom.
        """
        import numpy as np
        
        if ca is None or height is None or pd.isna(ca) or pd.isna(height):
            return None
        
        alpha = 3.0863

        beta = 0.8127

        try:
            agb = np.exp(alpha) * (ca * height) ** beta
            return max(0.0, round(agb, 2))  # AGB mag niet negatief zijn
        except Exception:
            return None

    def calculate_agb_bai(self, dbh, height):
        """
        Berekent AGB (Above Ground Biomass) (kroonbiomassa! niet de stam) op basis van AGB voor branches, fruits en leaves (Bai et al., 2020):
        AGB_branch = 0.0061 * (DBH^2 *H)^0.8905
        AGB_leaf = 0.2650*(DBH^2 *H)^0.4701
        AGB_fruit = 0.0342 * (DBH^2 *H)^0.5779
        AGB = AGB_branch + AGB_leaf + AGB_fruit
        waarbij DBH = diameter at breast height en H = hoogte van de boom.
        """
        import numpy as np

        if dbh is None or height is None or pd.isna(dbh) or pd.isna(height):
            return None
        try:
            dbh = float(dbh)
            height = float(height)

            if dbh < 2.0:
                dbh = dbh * 100

            AGB_branch = 0.0061 * (dbh ** 2 * height) ** 0.8905
            AGB_leaf = 0.2650 * (dbh ** 2 * height) ** 0.4701
            AGB_fruit = 0.0342 * (dbh ** 2 * height) ** 0.5779
            AGB = AGB_branch + AGB_leaf + AGB_fruit
            return max(0.0, round(AGB, 2))  # AGB mag niet negatief zijn
        except Exception:
            return None

    def parse_dbh_range_to_mean(self, dbh_value):
        """
        Zet een DBH attribuut om naar het gemiddelde als zwevendekommagetal (in meters).
        Ondersteunt:
        - Ranges: "20-30", "20 - 30 cm", "0.2-0.3 m", "20 tot 30"
        - Enkele waarden: "25", "25 cm", "0.25 m"
        """
        if dbh_value is None or dbh_value != dbh_value:  # Controleert ook op NaN (pd.isna)
            return None

        # Omzetten naar string en opschonen
        val_str = str(dbh_value).lower().replace(',', '.').strip()

        # Controleer of de waarde expliciet in meters is aangegeven (bijv. "0.25 m")
        is_meters = 'm' in val_str and 'cm' not in val_str

        # Zoek alle getallen (inclusief decimalen) in de tekst
        numbers = [float(n) for n in re.findall(r"\d+\.?\d*", val_str)]

        if not numbers:
            return None

        # Als er een range van 2 getallen is gevonden (bijv. 20 en 30)
        if len(numbers) >= 2:
            mean_val = sum(numbers[:2]) / 2.0
        else:
            mean_val = numbers[0]

        return mean_val

    def calculate_stem_volume(self, dbh, height):
        """
        Berekent het stamvolume (Stem Volume) op basis van de formule:
        ln(stem volume) = ln(a) + b * ln(DBH)
        """
        import numpy as np
        
        if dbh is None or height is None or pd.isna(dbh) or pd.isna(height):
            return None
        
        a = 0.0002143  
        b = 2.099     

        try:
            stem_volume = np.exp(np.log(a) + b * np.log(dbh))
            return max(0.0, round(stem_volume, 2))  # Stamvolume mag niet negatief zijn
        except Exception:
            return None
        