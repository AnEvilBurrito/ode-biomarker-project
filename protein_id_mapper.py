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
    # Create multiple mapping dictionaries for different strategies
    
    # Strategy 1: Direct mapping from full format (e.g., 'P37108;SRP14_HUMAN' -> 'P37108')
    direct_mapping_dict = {}
    # Strategy 2: Gene name to protein_id mapping (e.g., 'CDK4' -> 'P11802')
    gene_to_protein_dict = {}
    # Strategy 3: Protein_id to full format mapping (e.g., 'P37108' -> 'P37108;SRP14_HUMAN')
    protein_to_full_dict = {}
    
    for _, row in mapping_df.iterrows():
        full_format = row['goncalve_protein_id']
        protein_id = row['protein_id']
        gene_name = row['protein_name']
        
        # Strategy 1: Full format to protein_id
        direct_mapping_dict[full_format] = protein_id
        
        # Strategy 2: Gene name to protein_id
        gene_to_protein_dict[gene_name] = protein_id
        
        # Strategy 3: Protein_id to full format
        protein_to_full_dict[protein_id] = full_format
    
    def mapper(network_id):
        """
        Map a network protein ID to a proteomics column name
        
        Args:
            network_id: Protein ID from network file (e.g., 'CDK4' or 'P37108;SRP14_HUMAN')
            
        Returns:
            str: Mapped protein ID for proteomics data (e.g., 'P37108;SRP14_HUMAN')
        """
        # Strategy 1: If network_id is already in full format (contains ';' and '_HUMAN'), return as-is
        if ';' in network_id and '_HUMAN' in network_id:
            # Check if this full format exists in the mapping file
            if network_id in direct_mapping_dict:
                # If it exists in mapping, we could extract the protein_id, but let's keep full format
                # since proteomics data uses full format
                return network_id
            else:
                # If not in mapping, still return as-is since it's already in the right format
                return network_id
        
        # Strategy 2: If network_id is a gene name, try to map to protein_id then to full format
        if network_id in gene_to_protein_dict:
            protein_id = gene_to_protein_dict[network_id]
            if protein_id in protein_to_full_dict:
                return protein_to_full_dict[protein_id]
            return protein_id
        
        # Strategy 3: If network_id looks like a UniProt ID, try to map to full format
        if len(network_id) >= 6 and len(network_id) <= 10:
            if network_id[0] in ['P', 'Q', 'O', 'A'] and network_id[1:].isdigit():
                if network_id in protein_to_full_dict:
                    return protein_to_full_dict[network_id]
        
        # Strategy 4: If network_id is in full format but missing _HUMAN, try to complete it
        if ';' in network_id and '_HUMAN' not in network_id:
            # Extract the protein part and try to find the full format
            parts = network_id.split(';')
            if len(parts) == 2:
                protein_id = parts[0]
                if protein_id in protein_to_full_dict:
                    return protein_to_full_dict[protein_id]
        
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
        
        # Test with actual network features (CDK4, CDK6) and other formats
        test_network_ids = [
            'CDK4',  # Gene name from network
            'CDK6',  # Gene name from network
            'P37108;SRP14_HUMAN',  # Full format
            'Q96JP5;ZFP91_HUMAN',  # Full format
            'P11802',  # UniProt ID (CDK4)
            'Q00534'   # UniProt ID (CDK6)
        ]
        
        mapper = create_network_to_proteomics_mapper(mapping_df)
        
        print("\nTesting mapper function:")
        for network_id in test_network_ids:
            mapped_id = mapper(network_id)
            print(f"  {network_id} -> {mapped_id}")
        
        # Test batch mapping
        mapped_batch = map_network_features(test_network_ids, mapping_df)
        print(f"\nBatch mapping result: {mapped_batch}")
        
        # Test with actual proteomics data
        from DataLink import DataLink
        import numpy as np
        data_link = DataLink(path_loader, "data_codes.csv")
        loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
        proteomic_feature_data, _ = data_link.get_data_using_code(loading_code)
        proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])
        
        # Check which mapped features exist in proteomics data
        available_features = filter_available_features(mapped_batch, proteomic_feature_data.columns)
        print(f"\nAvailable features in proteomics data: {len(available_features)}")
        print("Available features:", available_features)
        
        return True
    else:
        print("Failed to load mapping file")
        return False


if __name__ == "__main__":
    test_mapper()
