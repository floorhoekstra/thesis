import os
import glob
import sys
import time
import numpy as np
import geopandas as gpd
import pickle

from tree_processing.models import CrownGrowthModel
from tree_processing.separation import FoxTree

def process_file(input_path, output_path, radius, v_res, min_pts, municipal_trees_path=None, ref_excel_path=None, growth_model=None):
    """
    Handles loading, processing, and saving for a single file with timing and Pickle caching.
    """
    file_start_time = time.time()

    # 1. Definieer het cache pad
    cache_path = output_path.replace('.xyz', '_fox_tree_cache.pkl')
    fox_tree = None

    # 2. Check of er al een gecachte status bestaat
    if os.path.exists(cache_path):
        print(f"Gecachte boomclusters gevonden! Inladen vanaf: {os.path.basename(cache_path)}")
        t_cache_start = time.time()
        try:
            with open(cache_path, "rb") as f:
                fox_tree = pickle.load(f)
            t_cache_end = time.time()
            print(f"  [Time] Cache Inladen: {t_cache_end - t_cache_start:.4f} sec (Data inlezen & scheiden overgeslagen!)")
        except Exception as e:
            print(f"Kon cache niet inladen ({e}). Er wordt opnieuw gerekend.")
            fox_tree = None

    # 3. Als er geen cache is, voer het volledige inlees- en scheidingsproces uit
    if fox_tree is None:
        try:
            t_load_start = time.time()
            points = np.loadtxt(input_path, usecols=(0, 1, 2))
            t_load_end = time.time()
        except Exception as e:
            print(f"Error loading data from {input_path}: {e}")
            return

        if points.size == 0:
            print(f"Point cloud in {input_path} is empty or invalid.")
            return

        muni_trees = None
        if municipal_trees_path and os.path.exists(municipal_trees_path):
            try:
                muni_trees = gpd.read_file(municipal_trees_path)
                print(f"Succesvol {len(muni_trees)} gemeentebomen ingeladen uit .gpkg!")
            except Exception as e:
                print(f"Fout bij het inladen van gemeentebomen .gpkg ({municipal_trees_path}): {e}")

        growth_model_inst = CrownGrowthModel(ref_excel_path) if ref_excel_path else None

        print(f"\nProcessing: {os.path.basename(input_path)}")
        print(f"Loaded {len(points)} points.")
        print(f"  [Time] Data Loading: {t_load_end - t_load_start:.4f} sec")

        # Tree Initialization & Separation
        t_process_start = time.time()
        fox_tree = FoxTree(points, radius, v_res, min_pts, municipal_trees=muni_trees, growth_model=growth_model_inst)
        fox_tree.separate_trees()
        t_process_end = time.time()
        print(f"  [Time] Tree Separation: {t_process_end - t_process_start:.4f} sec")

        # Opslaan in cache voor de volgende keer
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(fox_tree, f)
            print(f"Resulaten opgeslagen in cache: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"Kon cache niet opslaan: {e}")

    # 4. Visualize a few sample trees (Draait nu supersnel uit de cache!)
    sample_tree_ids = list(fox_tree.trees.keys())[119:120]
    for t_id in sample_tree_ids:
        fox_tree.visualize_tree(tree_id=t_id, alpha=0.8)
        
    # 5. Output
    t_write_start = time.time()
    # fox_tree.output_trees(output_path)  # Optioneel uitgeschakeld als je alleen polygonen wilt
    fox_tree.output_tree_polygons(output_path.replace('.xyz', '_polygons.gpkg'))
    t_write_end = time.time()
    
    file_end_time = time.time()

    print(f"\n--- Timing Summary for {os.path.basename(input_path)} ---")
    print(f"  Writing Output:    {t_write_end - t_write_start:.4f} sec")
    print(f"  Total File Time:   {file_end_time - file_start_time:.4f} sec")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    # =========================================================
    #                    USER PARAMETERS
    # =========================================================
    
    # Directory settings
    INPUT_DIR_NAME = "Input"
    OUTPUT_DIR_NAME = "Output"
    MUNICIPAL_TREES_FILE = "bomen_denhaag_clipped.gpkg"  # Optional: Path to municipal trees file (GPKG or CSV)
    REF_EXCEL_PATH = "additional_parameters.xlsx"  # Optional: Path to Appendix 4 reference table for crown growth model
    
    # Algorithm parameters
    RADIUS = 2.5              # Search radius
    VERTICAL_RESOLUTION = 1.5 # Vertical slice resolution
    MIN_PTS_PER_CLUSTER = 3   # Minimum points to form a tree seed
    
    # =========================================================
    #                   MAIN EXECUTION
    # =========================================================
    
    batch_start_time = time.time()

    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, INPUT_DIR_NAME)
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    
    # 2. Check Input Directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{INPUT_DIR_NAME}' not found at {input_dir}")
        sys.exit(1)
        
    # 3. Create Output Directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
        
    # 4. Find all .xyz files
    xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
    
    if not xyz_files:
        print(f"No .xyz files found in {input_dir}")
        sys.exit(0)
        
    print(f"Found {len(xyz_files)} files to process.")
    
    # 5. Process loop
    for file_path in xyz_files:
        base_name = os.path.basename(file_path)
        name_root, ext = os.path.splitext(base_name)
        
        # Skip output files if they accidentally ended up in input folder
        if f"_{RADIUS}_{VERTICAL_RESOLUTION}_{MIN_PTS_PER_CLUSTER}" in name_root:
            continue

        # Construct output filename: name_radius_res_minpts.xyz
        out_filename = f"{name_root}_{RADIUS}_{VERTICAL_RESOLUTION}_{MIN_PTS_PER_CLUSTER}{ext}"
        out_full_path = os.path.join(output_dir, out_filename)
        
        process_file(file_path, out_full_path, RADIUS, VERTICAL_RESOLUTION, MIN_PTS_PER_CLUSTER, municipal_trees_path=os.path.join(input_dir, MUNICIPAL_TREES_FILE), ref_excel_path=os.path.join(input_dir, REF_EXCEL_PATH), growth_model=None)
        
    batch_end_time = time.time()
    print(f"\nAll tasks completed in {batch_end_time - batch_start_time:.4f} seconds.")

    
