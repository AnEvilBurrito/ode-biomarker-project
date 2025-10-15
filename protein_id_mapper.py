"""
Protein ID Mapper for Network Benchmark
This module provides functions to map between different protein ID formats
"""

import pickle
import pandas as pd
from PathLoader import PathLoader


def load_protein_mapping(path_loader):
    """
    Load the protein ID mapping file
    
    Args:
        path_loader: PathLoader instance
        
    Returns:
        pandas.DataFrame: Mapping dataframe
    """
    mapping_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/goncalve_to_string_id_df.pkl"
    
    try:
        with open(mapping_file_path, 'rb') as f:
            mapping_df = pickle.load(f)
        return mapping_df
    except FileNotFoundError:
        print(f"Mapping file not found: {mapping_file_path}")
        return None


def create_network_to_proteomics_mapper(mapping_df):
    """
    Create a function to map network protein IDs to proteomics column names
    
    Args:
        mapping_df: Mapping dataframe from load_protein_mapping
        
    Returns:
        function: Mapper function that takes network ID and returns proteomics ID
    """
    # Create a mapping dictionary from the dataframe
    # Map from goncalve_protein_id to protein_id
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        network_id = row['goncalve_protein_id']
        proteomics_id = row['protein_id']
        mapping_dict[network_id] = proteomics_id
    
    def mapper(network_id):
        """
        Map a network protein ID to a proteomics column name
        
        Args:
            network_id: Protein ID from network file (e.g., 'Q06323;PSME1_HUMAN')
            
        Returns:
            str: Mapped protein ID for proteomics data (e.g., 'Q06323')
        """
        # First try direct mapping from the dictionary
        if network_id in mapping_dict:
            return mapping_dict[network_id]
        
        # If not found, try extracting UniProt ID (fallback method)
        if ';' in network_id:
            uniprot_id = network_id.split(';')[0]
            return uniprot_id
        
        # If no mapping found, return original ID
        return network_id
    
    return mapper


def map_network_features(network_features, mapping_df):
    """
    Map a list of network protein IDs to proteomics-compatible IDs
    
    Args:
        network_features: List of network protein IDs
        mapping_df: Mapping dataframe
        
    Returns:
        list: List of mapped protein IDs
    """
    mapper = create_network_to_proteomics_mapper(mapping_df)
    mapped_features = []
    
    for network_id in network_features:
        mapped_id = mapper(network_id)
        mapped_features.append(mapped_id)
    
    return mapped_features


def filter_available_features(mapped_features, proteomics_columns):
    """
    Filter mapped features to only include those available in proteomics data
    
    Args:
        mapped_features: List of mapped protein IDs
        proteomics_columns: Column names from proteomics data
        
    Returns:
        list: Filtered list of features that exist in proteomics data
    """
    available_features = []
    for feature in mapped_features:
        if feature in proteomics_columns:
            available_features.append(feature)
    
    return available_features


# Test function
def test_mapper():
    """Test the protein ID mapping functionality"""
    path_loader = PathLoader("data_config.env", "current_user.env")
    mapping_df = load_protein_mapping(path_loader)
    
    if mapping_df is not None:
        print("Mapping file loaded successfully")
        print(f"Mapping dataframe shape: {mapping_df.shape}")
        
        # Test with sample network IDs
        test_network_ids = [
            'P37108;SRP14_HUMAN',
            'Q96JP5;ZFP91_HUMAN', 
            'Q9Y4H2;IRS2_HUMAN'
        ]
        
        mapper = create_network_to_proteomics_mapper(mapping_df)
        
        print("\nTesting mapper function:")
        for network_id in test_network_ids:
            mapped_id = mapper(network_id)
            print(f"  {network_id} -> {mapped_id}")
        
        # Test batch mapping
        mapped_batch = map_network_features(test_network_ids, mapping_df)
        print(f"\nBatch mapping result: {mapped_batch}")
        
        return True
    else:
        print("Failed to load mapping file")
        return False


if __name__ == "__main__":
    test_mapper()
