import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

class FoxTree:
    def __init__(self, points_array, radius, vertical_resolution, min_pts_per_cluster, municipal_trees=None, growth_model=None):
        """
        :param points_array: numpy array of shape (N, 3) containing x, y, z
        """
        self.radius = radius
        self.vertical_resolution = vertical_resolution
        self.min_pts_seeds = min_pts_per_cluster

        self.growth_model = growth_model

        self.municipal_trees = municipal_trees
        self.matched_muni_indices = set()  # To track which municipal trees have been linked

        # Store original data
        self.points_data = points_array
        self.num_pts = len(points_array)
        
        # Initialize attributes to track state
        # tree_ids: -1 means unassigned
        self.tree_ids = np.full(self.num_pts, -1, dtype=int)
        
        # Bounding box
        self.z_min = np.min(points_array[:, 2])
        self.z_max = np.max(points_array[:, 2])
        
        # Tracking assigned points
        self.parsed_pt_indices = [] # List of indices
        self.next_tree_id = 0
        
        # Map of tree_id -> list of point indices (for final output)
        self.trees = {}
        self.tree_attributes = {}  # tree_id -> attributes (species, age, etc.)

        if municipal_trees is not None and len(municipal_trees) > 0:
            # Assign municipal trees to the nearest points
            self.seed_municipal_trees(municipal_trees)

    def seed_municipal_trees(self, municipal_trees, search_radius=4.0):
        """
        Koppelt punten dichtbij bekende gemeentebomen vooraf aan een uniek tree_id.
        search_radius: straal (in meters, XY-vlak) om punten rond de gemeenteboom te zoeken.
        """
        print(f"Pre-seeding met {len(municipal_trees)} gemeentebomen...")
        
        # 2D KDTree op alle x,y punten in de puntenwolk
        kdtree_2d = cKDTree(self.points_data[:, :2])
        
        if hasattr(municipal_trees, 'geometry'):
            # Gebruik de geometrie van de gemeentebomen
            muni_xy = np.column_stack((municipal_trees.geometry.x, municipal_trees.geometry.y))
        else:
            # Fallback als het toch een NumPy array is
            muni_xy = municipal_trees[:, :2]
        
        # Zoek alle punten binnen straal van elke gemeenteboom
        matched_indices_list = kdtree_2d.query_ball_point(muni_xy, r=search_radius)
        
        seeded_count = 0
        for i, pt_indices in enumerate(matched_indices_list):
            unassigned_indices = [idx for idx in pt_indices if self.tree_ids[idx] == -1]
            
            if unassigned_indices:
                # Sla op welke gemeenteboom succesvol gekoppeld is
                self.matched_muni_indices.add(i)
                
                row = municipal_trees.iloc[i] if hasattr(municipal_trees, 'iloc') else {}
                
                t_id = self.next_tree_id
                self.next_tree_id += 1
                
                self.tree_attributes[t_id] = {
                    "species_scientific": row.get("Boomsoort wetenschappelijk", row.get("BOOMSOORT", None)),
                    "age_years": row.get("Leeftijd", row.get("PLANTJAAR", None)),
                    "is_municipal": True
                }
                
                self.trees[t_id] = []
                for idx in unassigned_indices:
                    self.tree_ids[idx] = t_id
                    self.trees[t_id].append(idx)
                    self.parsed_pt_indices.append(idx)
                seeded_count += 1
                
        print(f"Succesvol {seeded_count} gemeentebomen gekoppeld aan initiële AHN-puntclusters.")

    def get_unlinked_municipal_polygons(self, default_radius=2.0):
            print("DEBUG: Self.growth_model bestaat?", self.growth_model is not None)
            if self.growth_model:
                print("DEBUG: Ingeladen geslachten (genus_data):", list(self.growth_model.genus_data.keys()))    
            if self.municipal_trees is None or len(self.municipal_trees) == 0:
                return []

            unlinked_records = []
            current_year = time.localtime().tm_year
            model_success_count = 0

            print("\n--- DIAGNOSTISCHE CHECK GROEIMODEL ---")

            for i in range(len(self.municipal_trees)):
                if i in self.matched_muni_indices:
                    continue

                row = self.municipal_trees.iloc[i]
                geom = row.geometry
                
                # Kolommen uitlezen
                scientific_name = row.get("Boomsoort wetenschappelijk", row.get("BOOMSOORT", row.get("SPECIES", None)))
                raw_age = row.get("Leeftijd", row.get("PLANTJAAR", row.get("PLANT_JAAR", None)))
                
                # Leeftijd omrekenen
                age = None
                if raw_age is not None and not pd.isna(raw_age):
                    try:
                        val = float(raw_age)
                        if val > 1800:
                            age = current_year - val
                        else:
                            age = val
                    except ValueError:
                        age = None

                # Berekend kroondiameter opvragen
                calc_cd = None
                if self.growth_model and scientific_name and age is not None:
                    calc_cd = self.growth_model.calculate_crown_diameter(scientific_name, age)

                # --- TUSSENTIJDSE PRINT CHECK (EERSTE 10 ONGEKOPPELDE BOMEN) ---
                if i < 10:
                    b0, b1, b2 = self.growth_model.get_parameters(scientific_name) if self.growth_model else (None, None, None)
                    print(f"Boom #{i}:")
                    print(f"  - Ingelezen Soortnaam : '{scientific_name}'")
                    print(f"  - Ingelezen Plantjaar/Leeftijd: {raw_age} -> Berekende Leeftijd: {age} jaar")
                    print(f"  - Gevonden Beta-parameters   : B0={b0}, B1={b1}, B2={b2}")
                    print(f"  - Berekende Kroondiameter    : {calc_cd} m")
                    if calc_cd is None:
                        print(f"  -> RESULTAAT: Fallback naar default radius {default_radius}m")
                    else:
                        print(f"  -> RESULTAAT: Succesvol berekend met radius {calc_cd/2.0}m")
                    print("-" * 50)
                # -------------------------------------------------------------

                if calc_cd is not None and calc_cd > 0:
                    radius = calc_cd / 2.0
                    model_success_count += 1
                else:
                    radius = default_radius

                circle_poly = geom.buffer(radius)

                unlinked_records.append({
                    "tree_id": -1,
                    "num_points": 0,
                    "height_m": None,
                    "max_z": None,
                    "species_sci": scientific_name,
                    "age_years": age,
                    "measured_cd_m": None,
                    "model_cd_m": calc_cd,
                    "is_municipal": True,
                    "poly_type": "unlinked_municipal_model",
                    "geometry": circle_poly
                })

            print(f"Groeimodel gelukt bij {model_success_count} van de {len(unlinked_records)} niet-gekoppelde bomen.\n")
            return unlinked_records

    def get_pts_in_layer(self, lower_z, higher_z):
        """
        Obtain point indices within the designated height interval.
        z > lower and z <= higher
        """
        # Note: The C++ condition is: this->m_Points[i].z <= higher && this->m_Points[i].z > lower
        condition = (self.points_data[:, 2] > lower_z) & (self.points_data[:, 2] <= higher_z)
        return np.where(condition)[0].tolist()

    def cluster_points(self, radius, pt_indices):
        """
        Cluster points that are within the distance of the given radius.
        Replicates the Custom BFS + Radius Search logic from C++.
        """
        clusters = []
        if not pt_indices:
            return clusters

        # Create a subset of points for KDTree construction
        current_points = self.points_data[pt_indices]
        
        # Map local index (0 to len-1) back to global index (pt_indices)
        local_to_global = {i: pid for i, pid in enumerate(pt_indices)}
        
        # Build KDTree for this layer
        kdtree = cKDTree(current_points)
        
        visited = set()
        pushed = set() # To keep track of what's added to stack/queue
        
        for i in range(len(pt_indices)):
            global_idx = pt_indices[i]
            
            if global_idx in visited:
                continue

            # Start a new cluster
            curr_cluster = []
            stack = [i] # Use local index for stack
            pushed.add(i)
            
            while stack:
                curr_local_idx = stack.pop()
                curr_global_idx = local_to_global[curr_local_idx]
                
                # Check visit status
                if curr_global_idx not in visited:
                    curr_cluster.append(curr_global_idx)
                    visited.add(curr_global_idx)
                
                # Query neighbors
                # query_ball_point returns indices in the `current_points` array (local indices)
                query_pt = current_points[curr_local_idx]
                neighbor_local_indices = kdtree.query_ball_point(query_pt, radius)
                
                for nb_local_idx in neighbor_local_indices:
                    # C++ logic: if (!isPushed) -> push
                    # We check if we have pushed this local index before
                    if nb_local_idx not in pushed:
                        stack.append(nb_local_idx)
                        pushed.add(nb_local_idx)

            if len(curr_cluster) >= self.min_pts_seeds:
                clusters.append(curr_cluster)
        
        return clusters

    def assign_pts_to_trees(self, new_pt_ids, radius):
        """
        Assign tree points based on nearest neighbor in already parsed points.
        Returns the list of points that were NOT assigned.
        """
        rest_pt_ids = []
        
        if not self.parsed_pt_indices:
            return new_pt_ids

        # Build KDTree from ALL previously parsed points (as per C++ logic)
        parsed_points_data = self.points_data[self.parsed_pt_indices]
        parsed_tree = cKDTree(parsed_points_data)
        
        # Query points in the current layer
        query_data = self.points_data[new_pt_ids]
        
        # k=1 for nearest neighbor
        distances, indices = parsed_tree.query(query_data, k=1)
        
        for i, dist in enumerate(distances):
            pt_id = new_pt_ids[i]
            
            if dist < radius:
                # Find the tree ID of the nearest neighbor
                nearest_parsed_idx = self.parsed_pt_indices[indices[i]]
                found_tree_id = self.tree_ids[nearest_parsed_idx]
                
                # Assign to current point
                self.tree_ids[pt_id] = found_tree_id
                
                # Update the tree cluster list
                if found_tree_id not in self.trees:
                    self.trees[found_tree_id] = []
                self.trees[found_tree_id].append(pt_id)
                
                # Mark as parsed
                self.parsed_pt_indices.append(pt_id)
            else:
                rest_pt_ids.append(pt_id)
                
        return rest_pt_ids

    def generate_tree_clusters(self, pt_clusters):
        """
        Assign new unique Tree IDs to the newly found clusters.
        """
        for cluster_indices in pt_clusters:
            # Assign new ID
            current_id = self.next_tree_id
            self.next_tree_id += 1
            
            self.trees[current_id] = []
            
            for idx in cluster_indices:
                self.tree_ids[idx] = current_id
                self.trees[current_id].append(idx)

    def concatenate_to_parsed_pts(self, clusters):
        """
        Add clustered points to the list of parsed points.
        """
        for cluster in clusters:
            self.parsed_pt_indices.extend(cluster)

    def separate_trees(self):
        print("Starting Top-Down Separation...")
        sep_start_time = time.time()
        
        # Als er al gemeentebomen zijn ge-seed, is dit GEEN top-layer meer
        is_top_layer = len(self.parsed_pt_indices) == 0
        layer_idx = 0
        
        curr_height = self.z_max
        while curr_height >= self.z_min:
            t0 = time.time()
            
            # 1. Haal punten op in deze laag die NOG NIET toegewezen zijn
            layer_pt_ids = self.get_pts_in_layer(curr_height - self.vertical_resolution, curr_height)
            pt_ids = [pid for pid in layer_pt_ids if self.tree_ids[pid] == -1]
            
            if not pt_ids:
                curr_height -= self.vertical_resolution
                continue
            
            print(f"Layer {layer_idx}: Height [{curr_height - self.vertical_resolution:.2f} - {curr_height:.2f}], Points: {len(pt_ids)}")
            
            curr_layer_clusters = []
            
            if is_top_layer:
                # Oorspronkelijke logica voor eerste laag
                print(f"  Clustering {len(pt_ids)} points (Top Layer)...")
                curr_layer_clusters = self.cluster_points(self.radius, pt_ids)
                self.generate_tree_clusters(curr_layer_clusters)
                self.concatenate_to_parsed_pts(curr_layer_clusters)
                
                if len(curr_layer_clusters) > 0:
                    is_top_layer = False
            else:
                # Punten toewijzen aan bestaande bomen (inclusief gemeentebomen)
                rest_pts = pt_ids
                print("  Incrementally assigning points...")
                
                while True:
                    prev_parsed_count = len(self.parsed_pt_indices)
                    rest_pts = self.assign_pts_to_trees(rest_pts, self.radius)
                    curr_parsed_count = len(self.parsed_pt_indices)
                    
                    if curr_parsed_count == prev_parsed_count:
                        break
                
                # Resterende punten die NIET bij een gemeenteboom/bestaande boom horen
                # worden hier als NIEUWE bomen geclusterd
                if rest_pts:
                    print(f"  Clustering remaining {len(rest_pts)} points...")
                    curr_layer_clusters = self.cluster_points(self.radius, rest_pts)
                    self.generate_tree_clusters(curr_layer_clusters)
                    self.concatenate_to_parsed_pts(curr_layer_clusters)
            
            t1 = time.time()
            print(f"  Layer Processing time: {t1 - t0:.3f} seconds.")
            print("=============================================")
            
            curr_height -= self.vertical_resolution
            layer_idx += 1

        sep_end_time = time.time()
        print(f"Total Separation Algorithm Time: {sep_end_time - sep_start_time:.4f} seconds.")

    def output_trees(self, filename):
        """
        Write the results to an XYZ file.
        Format: TreeID X Y Z R G B
        """
        print(f"Writing output to {filename}...")
        try:
            with open(filename, 'w') as f:
                # C++ iterates over the map of trees
                for t_id, indices in self.trees.items():
                    # Generate random color for this tree
                    r = np.random.randint(0, 255)
                    g = np.random.randint(0, 255)
                    b = np.random.randint(0, 255)
                    
                    for idx in indices:
                        pt = self.points_data[idx]
                        # Format: TreeID X Y Z R G B
                        f.write(f"{t_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")
            print("Finished writing file.")
        except IOError as e:
            print(f"Error writing file: {e}")

    def output_tree_polygons(self, filename, crs="EPSG:28992"):
        """
        Exporteert alle polygonen (gemeten, gemodelleerd en niet-gekoppeld) naar 1 GeoPackage.
        """
        from shapely.geometry import MultiPoint, Polygon

        print(f"Polygonen genereren en opslaan naar: {filename}...")
        all_records = []

        # 1. Verwerk alle gedetecteerde AHN-bomenwolk clusters
        for t_id, indices in self.trees.items():
            if len(indices) < 3:
                continue

            tree_points_xy = self.points_data[indices][:, :2]
            multi_pt = MultiPoint(tree_points_xy)
            hull = multi_pt.convex_hull

            if isinstance(hull, Polygon):
                z_values = self.points_data[indices][:, 2]
                max_z = float(np.max(z_values))
                min_z = float(np.min(z_values))
                tree_height = max_z - min_z
                
                attr = self.tree_attributes.get(t_id, {})
                scientific_name = attr.get("species_scientific", None)
                age = attr.get("age_years", None)
                is_municipal = attr.get("is_municipal", False)
                
                if age and float(age) > 1800:
                    current_year = time.localtime().tm_year
                    age = current_year - float(age)

                calc_cd = None
                if self.growth_model and scientific_name:
                    calc_cd = self.growth_model.calculate_crown_diameter(scientific_name, age)

                calc_cv = self.growth_model.calculate_crown_volume(round(2 * np.sqrt(hull.area / np.pi), 2), tree_height) if self.growth_model else None
                calc_agb = self.growth_model.calculate_agb(round(hull.area, 2), tree_height) if self.growth_model else None

                measured_cv = self.growth_model.measure_crown_volume(self.points_data[indices]) if self.growth_model else None

                base_info = {
                    "tree_id": int(t_id),
                    "num_points": int(len(indices)),
                    "height_m": round(tree_height, 2),
                    "max_z": round(max_z, 2),
                    "species_sci": scientific_name,
                    "age_years": age,
                    "measured_cd_m": round(2 * np.sqrt(hull.area / np.pi), 2),
                    "model_cd_m": calc_cd,
                    "measured_cv_m3": measured_cv,
                    "model_cv_m3": calc_cv,
                    "model_agb_t": calc_agb,
                    "is_municipal": is_municipal
                }

                if is_municipal:
                    # Gekoppelde gemeenteboom: Toon gemeten AHN polygoon + soortnaam
                    record = dict(base_info)
                    record["poly_type"] = "measured_municipal"
                    record["geometry"] = hull
                    all_records.append(record)
                else:
                    # Onbekende AHN boom
                    record = dict(base_info)
                    record["poly_type"] = "measured_ahn_tree"
                    record["geometry"] = hull
                    all_records.append(record)

        # 2. Voeg niet-gekoppelde gemeentebomen toe als ronde modelcirkels
        unlinked_records = self.get_unlinked_municipal_polygons(default_radius=2.0)
        #calculate volume for unlinked municipal trees if growth model is available
        if self.growth_model:
            for record in unlinked_records:
                model_cd = record.get("model_cd_m", None)
                # Assuming height is not available for unlinked municipal trees, we can set it to None
                record["model_cv_m3"] = self.growth_model.calculate_crown_volume(model_cd, None)
                record["model_agb_t"] = self.growth_model.calculate_agb(round(np.pi * (model_cd / 2) ** 2, 2), None) if model_cd else None
        all_records.extend(unlinked_records)

        if not all_records:
            print("Geen geldige boompolygonen gegenereerd.")
            return

        # Sla alle polygonen op in 1 GeoPackage
        gdf = gpd.GeoDataFrame(all_records, crs=crs)
        gdf.to_file(filename, driver="GPKG")
        print(f"Succesvol {len(gdf)} polygonen opgeslagen in {filename}!")

    def visualize_tree(self, tree_id, alpha=0.8):
        """
        Visualiseert de 3D Alpha Shape en puntenwolk van één specifieke boom.
        """
        import alphashape
        import numpy as np
        import pyvista as pv

        if tree_id not in self.trees:
            print(f"Boom ID {tree_id} niet gevonden!")
            return

        # Haal de punten van de specifieke boom op
        indices = self.trees[tree_id]
        tree_points = self.points_data[indices]
        unique_points = np.unique(tree_points, axis=0)

        if len(unique_points) < 4:
            print(f"Boom {tree_id} heeft te weinig unieke punten ({len(unique_points)}) voor een 3D mesh.")
            return

        # Bereken de Alpha Shape
        alpha_shape = alphashape.alphashape(unique_points, alpha)
        if not hasattr(alpha_shape, 'volume'):
            print(f"Alpha {alpha} was te hoog, fallback naar Convex Hull (alpha=0.0)...")
            alpha_shape = alphashape.alphashape(unique_points, 0.0)

        volume = round(alpha_shape.volume, 2)
        print(f"Visualiseren van Boom {tree_id} - Berekend Volume: {volume} m³")

        # PyVista Plotter
        plotter = pv.Plotter(window_size=[1024, 768])
        plotter.add_title(f"Tree ID: {tree_id} | Volume: {volume} m³ (Alpha = {alpha})")

        # Puntenwolk (Groen)
        point_cloud = pv.PolyData(unique_points)
        plotter.add_mesh(point_cloud, color="#2ecc71", point_size=5, render_points_as_spheres=True, label="Puntenwolk")

        # Mesh van de Alpha Shape (Blauw, transparant)
        faces = alpha_shape.faces
        pv_faces = np.c_[np.full(len(faces), 3), faces].ravel()
        mesh = pv.PolyData(alpha_shape.vertices, pv_faces)
        plotter.add_mesh(mesh, color="#3498db", opacity=0.45, show_edges=True, edge_color="#1b4f72", label="3D Volume Mesh")

        plotter.add_legend()
        plotter.add_axes()
        plotter.show_grid()
        plotter.show()