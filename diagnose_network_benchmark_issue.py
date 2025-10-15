# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ode-biomarker-project
#     language: python
#     name: python3
# ---

# %%
# Diagnostic Notebook for Network Benchmark Protein ID Mismatch Issue
# This notebook investigates the "not in index" error in benchmark_network.py

# %%
# Initialization
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.getcwd())

# %%
# Load PathLoader and DataLink
from PathLoader import PathLoader
from DataLink import DataLink

path_loader = PathLoader("data_config.env", "current_user.env")
data_link = DataLink(path_loader, "data_codes.csv")

print("PathLoader and DataLink initialized successfully")

# %%
# Load the proteomics data to examine column names
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
print(f"Loading proteomics data using code: {loading_code}")

proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)

print(f"Proteomic feature data shape: {proteomic_feature_data.shape}")
print(f"Proteomic label data shape: {proteomic_label_data.shape}")

# %%
# Examine proteomics data column names
print("Proteomics data column names (first 20):")
print(proteomic_feature_data.columns[:20].tolist())

print("\nProteomics data column names (last 20):")
print(proteomic_feature_data.columns[-20:].tolist())

print(f"\nTotal number of columns: {len(proteomic_feature_data.columns)}")

# %%
# Check the data type of column names
print("Data type of column names:", type(proteomic_feature_data.columns[0]))
print("Example column name:", proteomic_feature_data.columns[0])

# %%
# Load the network file to examine protein ID format
network_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl"
print(f"Loading network file from: {network_file_path}")

# Check if file exists
if os.path.exists(network_file_path):
    print("Network file exists")
    
    # Load the network data
    with open(network_file_path, 'rb') as f:
        nth_degree_neighbours = pickle.load(f)
    
    print(f"Network data type: {type(nth_degree_neighbours)}")
    print(f"Network data length: {len(nth_degree_neighbours)}")
    
    # Examine the structure of the network data
    for i, distance_features in enumerate(nth_degree_neighbours):
        if distance_features is not None:
            print(f"Distance {i+1}: {len(distance_features)} features")
            if len(distance_features) > 0:
                print(f"  First 5 features: {distance_features[:5]}")
                print(f"  Feature type: {type(distance_features[0])}")
                break  # Just show first non-empty distance
else:
    print("Network file does not exist!")

# %%
# Compare protein ID formats between proteomics data and network data
print("=== PROTEIN ID FORMAT COMPARISON ===")

# Get sample of proteomics column names
proteomics_sample = proteomic_feature_data.columns[:10].tolist()
print("Proteomics column names sample:")
for name in proteomics_sample:
    print(f"  {name}")

# Get sample of network protein IDs (from first non-empty distance)
network_sample = []
for distance_features in nth_degree_neighbours:
    if distance_features is not None and len(distance_features) > 0:
        network_sample = distance_features[:10]
        break

print("\nNetwork protein IDs sample:")
for protein_id in network_sample:
    print(f"  {protein_id}")

# %%
# Check if there's a mapping file available
mapping_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/goncalve_to_string_id_df.pkl"
print(f"Checking for mapping file: {mapping_file_path}")

if os.path.exists(mapping_file_path):
    print("Mapping file exists!")
    
    # Load the mapping file
    with open(mapping_file_path, 'rb') as f:
        mapping_df = pickle.load(f)
    
    print(f"Mapping data shape: {mapping_df.shape}")
    print("Mapping data columns:", mapping_df.columns.tolist())
    print("\nMapping data head:")
    print(mapping_df.head())
else:
    print("Mapping file does not exist")

# %%
# Check if network protein IDs exist in proteomics data columns
print("=== CHECKING NETWORK PROTEIN ID PRESENCE IN PROTEOMICS DATA ===")

# Test with a sample of network protein IDs
test_network_ids = network_sample[:5]
print(f"Testing {len(test_network_ids)} network protein IDs:")

for protein_id in test_network_ids:
    if protein_id in proteomic_feature_data.columns:
        print(f"  ✓ {protein_id} - FOUND in proteomics data")
    else:
        print(f"  ✗ {protein_id} - NOT FOUND in proteomics data")

# %%
# Check if there's a pattern in the proteomics column names that might match network IDs
print("=== ANALYZING PROTEOMICS COLUMN NAME PATTERNS ===")

# Look for common patterns in proteomics column names
proteomics_patterns = set()
for col_name in proteomic_feature_data.columns[:50]:  # Check first 50 columns
    # Extract potential protein ID patterns
    if ';' in col_name:
        proteomics_patterns.add('contains_semicolon')
    if '_HUMAN' in col_name:
        proteomics_patterns.add('contains_HUMAN')
    if col_name.startswith('P') or col_name.startswith('Q'):
        proteomics_patterns.add('starts_with_P_or_Q')

print("Proteomics column name patterns found:", proteomics_patterns)

# %%
# Check network protein ID patterns
print("=== ANALYZING NETWORK PROTEIN ID PATTERNS ===")

network_patterns = set()
for protein_id in network_sample:
    if ';' in protein_id:
        network_patterns.add('contains_semicolon')
    if '_HUMAN' in protein_id:
        network_patterns.add('contains_HUMAN')
    if protein_id.startswith('P') or protein_id.startswith('Q'):
        network_patterns.add('starts_with_P_or_Q')

print("Network protein ID patterns found:", network_patterns)

# %%
# Try to find a mapping between the formats
print("=== ATTEMPTING TO FIND MAPPING ===")

# If mapping file exists, check if it can help
if os.path.exists(mapping_file_path):
    print("Using mapping file to find correspondences...")
    
    # Check if mapping file has the right structure
    if 'goncalves_id' in mapping_df.columns and 'string_id' in mapping_df.columns:
        print("Mapping file has expected columns")
        
        # Check if any network protein IDs are in the mapping
        mapping_network_matches = mapping_df[mapping_df['string_id'].isin(network_sample)]
        print(f"Network IDs found in mapping: {len(mapping_network_matches)}")
        
        if len(mapping_network_matches) > 0:
            print("Sample mapping matches:")
            print(mapping_network_matches.head())
            
            # Check if mapped goncalves IDs are in proteomics data
            mapped_goncalves_ids = mapping_network_matches['goncalves_id'].tolist()
            proteomics_matches = [id for id in mapped_goncalves_ids if id in proteomic_feature_data.columns]
            print(f"Mapped goncalves IDs found in proteomics data: {len(proteomics_matches)}")
            
            if len(proteomics_matches) > 0:
                print("Successful mappings found!")
                for i, match in enumerate(proteomics_matches[:5]):
                    print(f"  {match}")

# %%
# Check if there's a simpler pattern conversion
print("=== CHECKING FOR SIMPLE PATTERN CONVERSION ===")

# Try removing the protein name part from network IDs (keep only UniProt ID)
print("Attempting to extract UniProt IDs from network protein IDs...")

uniprot_ids_from_network = []
for protein_id in network_sample:
    if ';' in protein_id:
        uniprot_id = protein_id.split(';')[0]
        uniprot_ids_from_network.append(uniprot_id)
        print(f"  {protein_id} -> {uniprot_id}")

# Check if these UniProt IDs exist in proteomics data
if uniprot_ids_from_network:
    print("\nChecking if extracted UniProt IDs exist in proteomics data:")
    for uniprot_id in uniprot_ids_from_network:
        if uniprot_id in proteomic_feature_data.columns:
            print(f"  ✓ {uniprot_id} - FOUND")
        else:
            print(f"  ✗ {uniprot_id} - NOT FOUND")

# %%
# Check the reverse: do proteomics column names contain UniProt IDs?
print("=== CHECKING PROTEOMICS COLUMN NAMES FOR UNIPROT IDS ===")

# Look for UniProt ID patterns in proteomics column names
uniprot_pattern_matches = []
for col_name in proteomic_feature_data.columns[:100]:  # Check first 100 columns
    # UniProt IDs typically start with P, Q, O, A, etc. and are 6-10 characters
    if len(col_name) >= 6 and len(col_name) <= 10:
        if col_name[0] in ['P', 'Q', 'O', 'A'] and col_name[1:].isdigit():
            uniprot_pattern_matches.append(col_name)

print(f"Potential UniProt IDs found in proteomics data: {len(uniprot_pattern_matches)}")
if uniprot_pattern_matches:
    print("Sample matches:", uniprot_pattern_matches[:10])

# %%
# Create a comprehensive diagnostic summary
print("=== DIAGNOSTIC SUMMARY ===")
print("\n1. PROTEOMICS DATA COLUMN FORMAT:")
print(f"   - Total columns: {len(proteomic_feature_data.columns)}")
print(f"   - Sample: {proteomic_feature_data.columns[0]} (type: {type(proteomic_feature_data.columns[0])})")
print(f"   - Patterns: {proteomics_patterns}")

print("\n2. NETWORK PROTEIN ID FORMAT:")
if nth_degree_neighbours:
    for i, distance_features in enumerate(nth_degree_neighbours):
        if distance_features is not None and len(distance_features) > 0:
            print(f"   - Distance {i+1}: {len(distance_features)} features")
            print(f"   - Sample: {distance_features[0]} (type: {type(distance_features[0])})")
            print(f"   - Patterns: {network_patterns}")
            break

print("\n3. MAPPING STATUS:")
if os.path.exists(mapping_file_path):
    print("   - Mapping file exists")
    if 'goncalves_id' in mapping_df.columns and 'string_id' in mapping_df.columns:
        print("   - Mapping file has correct structure")
        # Check if mapping can help
        test_mapping = mapping_df[mapping_df['string_id'].isin(network_sample[:3])]
        if len(test_mapping) > 0:
            print("   - Mapping can convert some network IDs")
        else:
            print("   - Mapping does not contain tested network IDs")
else:
    print("   - No mapping file found")

print("\n4. DIRECT COMPATIBILITY:")
test_matches = [id for id in network_sample if id in proteomic_feature_data.columns]
print(f"   - Direct matches: {len(test_matches)}/{len(network_sample)}")

print("\n5. PROPOSED SOLUTION:")
if len(test_matches) == 0:
    print("   - Network protein IDs need conversion to match proteomics column format")
    if uniprot_ids_from_network:
        uniprot_matches = [id for id in uniprot_ids_from_network if id in proteomic_feature_data.columns]
        if len(uniprot_matches) > 0:
            print("   - SOLUTION: Extract UniProt IDs from network IDs (remove ';PROTEIN_HUMAN' part)")
            print("   - This would convert 'Q06323;PSME1_HUMAN' -> 'Q06323'")
        else:
            print("   - SOLUTION: Need to find or create a mapping between formats")
    else:
        print("   - SOLUTION: Need to investigate alternative conversion methods")
else:
    print("   - Some IDs match directly, but others need conversion")

# %%
# Create a test function to demonstrate the fix
def create_protein_id_mapper(network_ids, proteomics_columns):
    """
    Create a function to map network protein IDs to proteomics column names
    """
    # First, try direct matching
    direct_matches = {}
    for network_id in network_ids:
        if network_id in proteomics_columns:
            direct_matches[network_id] = network_id
    
    # If no direct matches, try UniProt ID extraction
    if not direct_matches:
        uniprot_mapper = {}
        for network_id in network_ids:
            if ';' in network_id:
                uniprot_id = network_id.split(';')[0]
                if uniprot_id in proteomics_columns:
                    uniprot_mapper[network_id] = uniprot_id
        
        if uniprot_mapper:
            return uniprot_mapper
    
    # If still no matches, return empty dict (need manual mapping)
    return direct_matches

# Test the mapper function
print("=== TESTING PROTEIN ID MAPPER ===")
test_network_ids = network_sample
mapper = create_protein_id_mapper(test_network_ids, proteomic_feature_data.columns)

if mapper:
    print("Mapper function successful!")
    print("Sample mappings:")
    for network_id, proteomics_id in list(mapper.items())[:5]:
        print(f"  {network_id} -> {proteomics_id}")
else:
    print("Mapper function could not find matches")
    print("Alternative mapping strategy needed")

# %%
# Create the final diagnostic report
print("=== FINAL DIAGNOSTIC REPORT ===")
print("\nROOT CAUSE:")
print("The error occurs because network protein IDs (e.g., 'Q06323;PSME1_HUMAN')")
print("do not match the column names in the proteomics data.")

print("\nOBSERVED FORMATS:")
print("- Network IDs: 'UNIPROT_ID;PROTEIN_NAME_HUMAN'")
print("- Proteomics columns: Appear to use simpler identifiers (possibly just UniProt IDs)")

print("\nRECOMMENDED SOLUTION:")
if mapper:
    print("1. Extract UniProt IDs from network protein IDs by splitting on ';'")
    print("2. Use only the first part (before the semicolon)")
    print("3. This should match the proteomics data column names")
else:
    print("1. Investigate the exact format of proteomics column names")
    print("2. Create a custom mapping function")
    print("3. Modify the network feature selection wrapper to use the mapping")

print("\nNEXT STEPS:")
print("1. Modify the mrmr_network_select_wrapper function in benchmark_network.py")
print("2. Add protein ID mapping logic")
print("3. Test the fix with a small subset of data")

# %%
# Save diagnostic results to file
diagnostic_results = {
    'proteomics_columns_sample': proteomic_feature_data.columns[:20].tolist(),
    'network_ids_sample': network_sample,
    'mapping_available': os.path.exists(mapping_file_path),
    'direct_matches': len(test_matches),
    'proposed_solution': "Extract UniProt IDs from network IDs" if mapper else "Need custom mapping",
    'test_mapper_results': dict(list(mapper.items())[:5]) if mapper else {}
}

import json
diagnostic_file = "network_benchmark_diagnostic_results.json"
with open(diagnostic_file, 'w') as f:
    json.dump(diagnostic_results, f, indent=2)

print(f"Diagnostic results saved to: {diagnostic_file}")
