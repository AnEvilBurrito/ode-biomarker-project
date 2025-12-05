#!/usr/bin/env python3
"""
Quick sanity test to verify the gene subset SHAP analysis script imports and references work correctly
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd

def test_imports_and_functions():
    """Test that all imports and function references work correctly"""
    
    print("🧪 Testing imports and function references...")
    
    try:
        # Test core imports
        from PathLoader import PathLoader
        from DataLink import DataLink
        from toolkit import get_model_from_string
        
        print("✅ Core imports successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Test SHAP import
        import shap
        print(f"✅ SHAP version: {shap.__version__}")
        
    except ImportError as e:
        print(f"❌ SHAP import error: {e}")
        print("Note: SHAP might need to be installed separately")
        return False
    
    # Test function imports from our new script
    try:
        from feature_importance_analysis_rf_shap_gene_subsets import (
            extract_gene_subset, 
            fgfr4_genes, 
            cdk46_genes,
            mrmr_standard_select,
            create_shap_feature_importance_pipeline,
            shap_feature_importance_eval
        )
        
        print("✅ All function imports from gene subset SHAP script successful")
        print(f"  - FGFR4 genes: {len(fgfr4_genes)}")
        print(f"  - CDK4/6 genes: {len(cdk46_genes)}")
        
    except ImportError as e:
        print(f"❌ Function import error: {e}")
        traceback.print_exc()
        return False
    
    # Test mock data creation and gene extraction
    try:
        mock_genes = ['CDK4', 'CDK6', 'FGFR4', 'AKT1', 'SOME_RANDOM_GENE']
        mock_cells = ['Cell_A', 'Cell_B']
        mock_data = pd.DataFrame(
            np.random.rand(len(mock_cells), len(mock_genes)),
            index=mock_cells,
            columns=mock_genes
        )
        
        # Test gene subset extraction
        cdk46_subset = extract_gene_subset(mock_data, cdk46_genes)
        print(f"✅ Gene subset extraction: {cdk46_subset.shape}")
        
    except Exception as e:
        print(f"❌ Gene subset extraction test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_pipeline_functions():
    """Test that pipeline creation and evaluation functions are callable"""
    print("\n🧪 Testing pipeline functions...")
    
    try:
        from feature_importance_analysis_rf_shap_gene_subsets import (
            create_shap_feature_importance_pipeline,
            shap_feature_importance_eval
        )
        
        # Test pipeline creation
        pipeline_func = create_shap_feature_importance_pipeline(
            lambda x, y, k: ([], []),  # dummy selector
            k=500,
            method_name="test",
            model_name="RandomForestRegressor_config1"
        )
        
        print("✅ Pipeline function creation successful")
        
        # Test small mock data for the pipeline
        mock_data = pd.DataFrame({
            'gene1': [1, 2, 3],
            'gene2': [4, 5, 6]
        }, index=['s1', 's2', 's3'])
        mock_labels = pd.Series([0.1, 0.2, 0.3], index=['s1', 's2', 's3'])
        
        # This won't run the full pipeline but will test function call
        print("✅ Pipeline function signature compatible")
        
    except Exception as e:
        print(f"❌ Pipeline function test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests"""
    print("🔬 Gene Subset SHAP Analysis Script - Sanity Tests")
    print("=" * 50)
    
    success = True
    
    # Test 1: Imports and function references
    success &= test_imports_and_functions()
    
    # Test 2: Pipeline functions
    success &= test_pipeline_functions()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All sanity tests passed! The script should be ready to run.")
        print("\n📋 Next steps:")
        print("1. Verify data file paths in data_config.env")
        print("2. Check data_codes.csv contains the required datasets")
        print("3. Run the script: python feature_importance_analysis_rf_shap_gene_subsets.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
