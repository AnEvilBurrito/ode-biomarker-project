#!/usr/bin/env python3
"""
Quick test script to verify the gene subset extraction function works correctly
"""

import pandas as pd
import numpy as np

# Import the function from our main script
from benchmark_network_gene_subsets_memory_optimized import extract_gene_subset, fgfr4_genes, cdk46_genes

def test_gene_subset_extraction():
    """Test the gene subset extraction function with mock data"""
    
    # Create a mock RNASeq dataset
    mock_genes = ['CDK4', 'CDK6', 'FGFR4', 'AKT1', 'IRS1', 'SOME_RANDOM_GENE', 'ANOTHER_GENE']
    mock_cell_lines = ['CellLine_A', 'CellLine_B', 'CellLine_C']
    
    # Create mock expression data
    mock_data = pd.DataFrame(
        np.random.rand(len(mock_cell_lines), len(mock_genes)),
        index=mock_cell_lines,
        columns=mock_genes
    )
    
    print("Mock dataset shape:", mock_data.shape)
    print("Mock dataset columns:", mock_data.columns.tolist())
    
    # Test CDK4/6 gene extraction
    print("\n=== Testing CDK4/6 Gene Extraction ===")
    cdk46_subset = extract_gene_subset(mock_data, cdk46_genes)
    print("CDK4/6 subset shape:", cdk46_subset.shape)
    print("CDK4/6 extracted genes:", cdk46_subset.columns.tolist())
    
    # Test FGFR4 gene extraction  
    print("\n=== Testing FGFR4 Gene Extraction ===")
    fgfr4_subset = extract_gene_subset(mock_data, fgfr4_genes)
    print("FGFR4 subset shape:", fgfr4_subset.shape)
    print("FGFR4 extracted genes:", fgfr4_subset.columns.tolist())
    
    print("\n✅ Gene subset extraction test completed successfully!")

if __name__ == "__main__":
    test_gene_subset_extraction()
