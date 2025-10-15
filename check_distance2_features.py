#!/usr/bin/env python3
"""
Quick diagnostic to check distance 2 network features format
"""

import os
import pickle
import pandas as pd
from PathLoader import PathLoader
from protein_id_mapper import load_protein_mapping, map_network_features, filter_available_features

def main():
    print("=== CHECKING DISTANCE 2 NETWORK FEATURES ===")
    
    # Initialize path loader
    path_loader = PathLoader("data_config.env", "current_user.env")
    
    # Load network structure
    network_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl"
    print(f"Loading network structure from: {network_file_path}")
    
    with open(network_file_path, 'rb') as f:
        nth_degree_neighbours = pickle.load(f)
    
    # Check distance 2 features
    distance = 2
    network_features = nth_degree_neighbours[distance - 1] if nth_degree_neighbours[distance - 1] is not None else []
    print(f"Distance {distance} features: {len(network_features)}")
    print("Sample distance 2 features:", network_features[:10])
    
    # Check what format they're in
    print("\n=== ANALYZING DISTANCE 2 FEATURE FORMATS ===")
    for i, feature in enumerate(network_features[:10]):
        print(f"Feature {i}: '{feature}'")
        print(f"  Contains ';': {';' in feature}")
        print(f"  Contains '_HUMAN': {'_HUMAN' in feature}")
        print(f"  Length: {len(feature)}")
        print(f"  Starts with P/Q/O/A: {feature[0] in ['P', 'Q', 'O', 'A'] if len(feature) > 0 else False}")
        print(f"  Is likely gene name: {len(feature) <= 10 and feature.isalpha()}")
        print()
    
    # Load proteomics data to check column format
    from DataLink import DataLink
    import numpy as np
    data_link = DataLink(path_loader, "data_codes.csv")
    loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
    proteomic_feature_data, _ = data_link.get_data_using_code(loading_code)
    proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])
    
    print("Sample proteomics column names:")
    for i, col in enumerate(proteomic_feature_data.columns[:5]):
        print(f"  {i}: '{col}'")
    
    # Try mapping distance 2 features
    mapping_df = load_protein_mapping(path_loader)
    if mapping_df is not None:
        mapped_features = map_network_features(network_features[:10], mapping_df)
        print(f"\nMapped distance 2 features (first 10):")
        for orig, mapped in zip(network_features[:10], mapped_features):
            print(f"  '{orig}' -> '{mapped}'")
        
        # Check which mapped features exist in proteomics data
        available_features = filter_available_features(mapped_features, proteomic_feature_data.columns)
        print(f"Available in proteomics: {len(available_features)}")
        print("Available features:", available_features)

if __name__ == "__main__":
    main()
