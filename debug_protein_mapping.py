#!/usr/bin/env python3
"""
Debug script for protein ID mapping issue
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from PathLoader import PathLoader
from DataLink import DataLink
from protein_id_mapper import load_protein_mapping, map_network_features, filter_available_features

def main():
    print("=== DEBUGGING PROTEIN ID MAPPING ISSUE ===")
    
    # Initialize path loader and data link
    path_loader = PathLoader("data_config.env", "current_user.env")
    data_link = DataLink(path_loader, "data_codes.csv")
    
    # Load proteomics data
    loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
    print(f"Loading proteomics data using code: {loading_code}")
    
    proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)
    proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])
    
    print(f"Proteomic feature data shape: {proteomic_feature_data.shape}")
    print(f"Proteomic label data shape: {proteomic_label_data.shape}")
    
    # Load network structure
    network_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl"
    print(f"Loading network structure from: {network_file_path}")
    
    with open(network_file_path, 'rb') as f:
        nth_degree_neighbours = pickle.load(f)
    
    print(f"Network structure loaded. Available distances: {list(range(1, len(nth_degree_neighbours) + 1))}")
    
    # Load protein mapping
    mapping_df = load_protein_mapping(path_loader)
    print(f"Mapping dataframe shape: {mapping_df.shape}")
    print("Mapping dataframe columns:", mapping_df.columns.tolist())
    
    # Test with distance 1 features
    distance = 1
    network_features = nth_degree_neighbours[distance - 1] if nth_degree_neighbours[distance - 1] is not None else []
    print(f"\n=== TESTING DISTANCE {distance} ===")
    print(f"Original network features: {len(network_features)}")
    print("Sample network features:", network_features[:5])
    
    # Map network features
    mapped_network_features = map_network_features(network_features, mapping_df)
    print(f"Mapped network features: {len(mapped_network_features)}")
    print("Sample mapped features:", mapped_network_features[:5])
    
    # Check which mapped features exist in proteomics data
    available_features = filter_available_features(mapped_network_features, proteomic_feature_data.columns)
    print(f"Available features in proteomics data: {len(available_features)}")
    print("Available features:", available_features)
    
    # Check if any original network features exist directly
    direct_matches = [f for f in network_features if f in proteomic_feature_data.columns]
    print(f"Direct matches (no mapping): {len(direct_matches)}")
    print("Direct matches:", direct_matches)
    
    # Investigate the proteomics column format
    print(f"\n=== INVESTIGATING PROTEOMICS COLUMN FORMAT ===")
    print("Sample proteomics column names:")
    for i, col in enumerate(proteomic_feature_data.columns[:10]):
        print(f"  {i}: {col}")
    
    # Check if mapped features might be in a different format
    print(f"\n=== CHECKING FOR ALTERNATIVE FORMATS ===")
    
    # Check if proteomics columns contain UniProt IDs in different formats
    uniprot_patterns = []
    for col in proteomic_feature_data.columns[:100]:
        # Look for patterns that might be UniProt IDs
        if len(col) >= 6 and len(col) <= 10:
            if col[0] in ['P', 'Q', 'O', 'A'] and col[1:].isdigit():
                uniprot_patterns.append(col)
    
    print(f"Potential UniProt ID patterns found: {len(uniprot_patterns)}")
    if uniprot_patterns:
        print("Sample UniProt patterns:", uniprot_patterns[:10])
    
    # Check if any mapped features match these patterns
    matches_with_patterns = [f for f in mapped_network_features if f in uniprot_patterns]
    print(f"Mapped features matching UniProt patterns: {len(matches_with_patterns)}")
    print("Matches:", matches_with_patterns)
    
    # Check the mapping file more carefully
    print(f"\n=== ANALYZING MAPPING FILE ===")
    print("Mapping file head:")
    print(mapping_df.head(10))
    
    # Check if the network features exist in the mapping file
    network_features_in_mapping = mapping_df[mapping_df['goncalve_protein_id'].isin(network_features)]
    print(f"Network features found in mapping file: {len(network_features_in_mapping)}")
    if len(network_features_in_mapping) > 0:
        print("Mapping entries for network features:")
        print(network_features_in_mapping[['goncalve_protein_id', 'protein_id']])
    
    # Check if the mapped protein_ids exist in proteomics data
    mapped_ids_in_proteomics = mapping_df[mapping_df['protein_id'].isin(proteomic_feature_data.columns)]
    print(f"Mapped protein_ids found in proteomics data: {len(mapped_ids_in_proteomics)}")
    if len(mapped_ids_in_proteomics) > 0:
        print("Sample successful mappings:")
        print(mapped_ids_in_proteomics[['goncalve_protein_id', 'protein_id']].head(10))
    
    # Check if there's a pattern in the proteomics column names
    print(f"\n=== PROTEOMICS COLUMN NAME PATTERNS ===")
    sample_columns = proteomic_feature_data.columns[:50]
    patterns = {}
    
    for col in sample_columns:
        # Check for various patterns
        if ';' in col:
            patterns['semicolon'] = patterns.get('semicolon', 0) + 1
        if '_HUMAN' in col:
            patterns['human_suffix'] = patterns.get('human_suffix', 0) + 1
        if col.startswith('P') or col.startswith('Q'):
            patterns['uniprot_like'] = patterns.get('uniprot_like', 0) + 1
        if len(col) <= 10 and col[0].isalpha() and col[1:].isdigit():
            patterns['potential_uniprot'] = patterns.get('potential_uniprot', 0) + 1
    
    print("Column name patterns:", patterns)
    
    # Final diagnostic summary
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Root issue: Mapped protein IDs not found in proteomics data")
    print(f"Network features (distance {distance}): {len(network_features)}")
    print(f"Mapped features: {len(mapped_network_features)}")
    print(f"Available in proteomics: {len(available_features)}")
    print(f"Direct matches (no mapping): {len(direct_matches)}")
    
    if len(available_features) == 0:
        print("SOLUTION NEEDED: The protein ID mapping is not working correctly.")
        print("Possible causes:")
        print("1. Proteomics data uses a different format than expected")
        print("2. Mapping file doesn't contain the correct mappings")
        print("3. Network features need a different conversion method")
        
        # Suggest next steps
        print("\nNEXT STEPS:")
        print("1. Check the exact format of proteomics column names")
        print("2. Verify the mapping file contains the correct network protein IDs")
        print("3. Try alternative mapping strategies")

if __name__ == "__main__":
    main()
