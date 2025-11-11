#!/usr/bin/env python3
"""
Feature Difference Visualization Demo
This script demonstrates the Venn diagrams and feature importance comparison plots
using sample data to show how the visualizations work.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Create sample data for demonstration
def create_sample_feature_data():
    """Create sample feature importance data for three methods"""
    
    # Define methods and their descriptions
    methods = {
        "network_only_d3": "Network only (distance 3)", 
        "mrmr_only": "MRMR only",
        "mrmr_network_d3": "MRMR + Network (distance 3)"
    }
    
    # Create sample feature names (simulating proteomics features)
    feature_names = [f"Gene{i}P{10000+i}:HUMAN" for i in range(1, 101)]
    
    # Create sample data for each method
    method_features = {}
    
    for method_name, method_desc in methods.items():
        # Create SHAP data
        shap_data = {}
        for i, feature in enumerate(feature_names):
            # Simulate importance scores with some randomness
            base_importance = np.random.normal(0.1, 0.05)
            # Make some features more important for specific methods
            if method_name == "network_only_d3" and i < 20:
                base_importance += 0.2  # Network features are more important for network method
            elif method_name == "mrmr_only" and i >= 20 and i < 40:
                base_importance += 0.15  # Different features for MRMR
            elif method_name == "mrmr_network_d3" and i >= 40 and i < 60:
                base_importance += 0.1  # Combined method preferences
            
            shap_data[f"{method_name}_shap"] = {
                'features': set(feature_names[:50]),  # Top 50 features
                'importance_scores': {feature: max(0.01, base_importance + np.random.normal(0, 0.02)) 
                                    for feature in feature_names[:50]},
                'method': method_name,
                'importance_method': 'shap',
                'description': f"{method_desc} (SHAP)"
            }
        
        # Create MDI data (slightly different feature preferences)
        mdi_data = {}
        for i, feature in enumerate(feature_names):
            base_importance = np.random.normal(0.08, 0.04)
            # MDI might prefer different features
            if method_name == "network_only_d3" and i >= 10 and i < 30:
                base_importance += 0.18
            elif method_name == "mrmr_only" and i >= 25 and i < 45:
                base_importance += 0.12
            elif method_name == "mrmr_network_d3" and i >= 35 and i < 55:
                base_importance += 0.08
            
            mdi_data[f"{method_name}_mdi"] = {
                'features': set(feature_names[:50]),
                'importance_scores': {feature: max(0.01, base_importance + np.random.normal(0, 0.02)) 
                                    for feature in feature_names[:50]},
                'method': method_name,
                'importance_method': 'mdi',
                'description': f"{method_desc} (MDI)"
            }
        
        # Combine both importance methods
        method_features.update(shap_data)
        method_features.update(mdi_data)
    
    return method_features, methods

def create_venn_diagrams(method_features, methods, top_n=50):
    """Create Venn diagrams showing feature overlaps between methods"""
    try:
        from matplotlib_venn import venn2, venn3
        
        # Group features by method (combining SHAP and MDI for each method)
        method_groups = {}
        for key, data in method_features.items():
            method_name = data['method']
            if method_name not in method_groups:
                method_groups[method_name] = {
                    'features': set(),
                    'description': methods[method_name],
                    'importance_scores': {}
                }
            method_groups[method_name]['features'].update(data['features'])
            method_groups[method_name]['importance_scores'].update(data['importance_scores'])
        
        # Create pairwise Venn diagrams for each method combination
        method_names = list(method_groups.keys())
        
        print("Creating Venn diagrams...")
        
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1 = method_names[i]
                method2 = method_names[j]
                
                features1 = method_groups[method1]['features']
                features2 = method_groups[method2]['features']
                
                plt.figure(figsize=(8, 6), dpi=300)
                venn = venn2([features1, features2], 
                            set_labels=(method_groups[method1]['description'], 
                                      method_groups[method2]['description']),
                            set_colors=('#ff7f0e', '#2ca02c'), alpha=0.7)
                
                plt.title(f'Feature Overlap: {method_groups[method1]["description"]} vs {method_groups[method2]["description"]}\n(Top {top_n} Features)', 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Add counts to the plot
                if venn.get_label_by_id('10') is not None:
                    venn.get_label_by_id('10').set_text(f"{len(features1 - features2)}")
                if venn.get_label_by_id('01') is not None:
                    venn.get_label_by_id('01').set_text(f"{len(features2 - features1)}")
                if venn.get_label_by_id('11') is not None:
                    venn.get_label_by_id('11').set_text(f"{len(features1 & features2)}")
                
                filename = f"demo_venn_{method1}_vs_{method2}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✓ Created Venn diagram: {filename}")
        
        # Create three-way Venn diagram if we have exactly 3 methods
        if len(method_groups) == 3:
            method1, method2, method3 = method_names
            features1 = method_groups[method1]['features']
            features2 = method_groups[method2]['features']
            features3 = method_groups[method3]['features']
            
            plt.figure(figsize=(10, 8), dpi=300)
            venn = venn3([features1, features2, features3], 
                        set_labels=(method_groups[method1]['description'],
                                  method_groups[method2]['description'],
                                  method_groups[method3]['description']),
                        set_colors=('#ff7f0e', '#2ca02c', '#d62728'), alpha=0.7)
            
            plt.title(f'Three-Way Feature Overlap\n(Top {top_n} Features)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = "demo_venn_three_way.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Created three-way Venn diagram: {filename}")
            
    except ImportError:
        print("matplotlib_venn not available for Venn diagrams")
        create_overlap_bar_plots(method_features, methods)

def create_overlap_bar_plots(method_features, methods):
    """Create bar plots showing feature overlaps as fallback"""
    # Group features by method
    method_groups = {}
    for key, data in method_features.items():
        method_name = data['method']
        if method_name not in method_groups:
            method_groups[method_name] = {
                'features': set(),
                'description': methods[method_name]
            }
        method_groups[method_name]['features'].update(data['features'])
    
    # Calculate pairwise overlaps
    method_names = list(method_groups.keys())
    overlap_data = []
    
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1 = method_names[i]
            method2 = method_names[j]
            
            features1 = method_groups[method1]['features']
            features2 = method_groups[method2]['features']
            
            intersection = features1 & features2
            unique1 = features1 - features2
            unique2 = features2 - features1
            
            overlap_data.append({
                'method_pair': f"{method1} vs {method2}",
                'unique_to_method1': len(unique1),
                'unique_to_method2': len(unique2),
                'overlap': len(intersection),
                'total_features': len(features1 | features2)
            })
    
    # Create bar plot
    if overlap_data:
        df = pd.DataFrame(overlap_data)
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        x_pos = np.arange(len(df))
        width = 0.25
        
        # Plot bars
        bars1 = ax.bar(x_pos - width, df['unique_to_method1'], width, 
                      label='Unique to Method 1', color='#ff7f0e', alpha=0.7)
        bars2 = ax.bar(x_pos, df['overlap'], width, 
                      label='Overlap', color='#2ca02c', alpha=0.7)
        bars3 = ax.bar(x_pos + width, df['unique_to_method2'], width, 
                      label='Unique to Method 2', color='#d62728', alpha=0.7)
        
        ax.set_xlabel('Method Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        ax.set_title('Feature Overlap Between Methods (Demo)', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([row['method_pair'] for _, row in df.iterrows()], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("demo_feature_overlap_bar.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Created feature overlap bar plot (Venn diagram fallback)")

def create_feature_importance_comparison_plots(method_features, methods, top_n=20):
    """Create side-by-side comparison plots of feature importance across methods"""
    # Group by method and importance method
    comparison_data = {}
    
    for key, data in method_features.items():
        method_name = data['method']
        imp_method = data['importance_method']
        
        if method_name not in comparison_data:
            comparison_data[method_name] = {}
        comparison_data[method_name][imp_method] = data
    
    print("Creating feature importance comparison plots...")
    
    # Create comparison plots for each method
    for method_name, imp_data in comparison_data.items():
        if len(imp_data) == 2:  # Both SHAP and MDI available
            shap_data = imp_data.get('shap')
            mdi_data = imp_data.get('mdi')
            
            if shap_data and mdi_data:
                # Get top features from both methods
                top_shap_features = list(shap_data['features'])[:top_n]
                top_mdi_features = list(mdi_data['features'])[:top_n]
                
                # Create unified feature list (union of top features)
                all_features = set(top_shap_features) | set(top_mdi_features)
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(index=list(all_features))
                
                for feature in all_features:
                    comparison_df.loc[feature, 'SHAP_importance'] = shap_data['importance_scores'].get(feature, 0)
                    comparison_df.loc[feature, 'MDI_importance'] = mdi_data['importance_scores'].get(feature, 0)
                
                # Sort by maximum importance
                comparison_df['max_importance'] = comparison_df[['SHAP_importance', 'MDI_importance']].max(axis=1)
                comparison_df = comparison_df.sort_values('max_importance', ascending=False)
                
                # Create comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300)
                
                # SHAP importance plot
                y_pos = np.arange(len(comparison_df))
                ax1.barh(y_pos, comparison_df['SHAP_importance'], color='#1f77b4', alpha=0.7)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(comparison_df.index, fontsize=10)
                ax1.set_xlabel('SHAP Importance', fontsize=12, fontweight='bold')
                ax1.set_title(f'SHAP Importance - {methods[method_name]}', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                
                # MDI importance plot
                ax2.barh(y_pos, comparison_df['MDI_importance'], color='#ff7f0e', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(comparison_df.index, fontsize=10)
                ax2.set_xlabel('MDI Importance', fontsize=12, fontweight='bold')
                ax2.set_title(f'MDI Importance - {methods[method_name]}', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                filename = f"demo_importance_comparison_{method_name}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✓ Created importance comparison plot: {filename}")
                
                # Create correlation scatter plot
                plt.figure(figsize=(8, 6), dpi=300)
                plt.scatter(comparison_df['SHAP_importance'], comparison_df['MDI_importance'], 
                           alpha=0.6, s=50, color='#2ca02c')
                plt.xlabel('SHAP Importance', fontsize=12, fontweight='bold')
                plt.ylabel('MDI Importance', fontsize=12, fontweight='bold')
                plt.title(f'SHAP vs MDI Importance Correlation\n{methods[method_name]} (Demo)', fontsize=14, fontweight='bold')
                
                # Add correlation line if there's a relationship
                if len(comparison_df) > 1:
                    z = np.polyfit(comparison_df['SHAP_importance'], comparison_df['MDI_importance'], 1)
                    p = np.poly1d(z)
                    plt.plot(comparison_df['SHAP_importance'], p(comparison_df['SHAP_importance']), 
                            "r--", alpha=0.8, linewidth=2)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = f"demo_importance_correlation_{method_name}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✓ Created importance correlation plot: {filename}")

def main():
    """Main function to run the demonstration"""
    print("=== Feature Difference Visualization Demo ===")
    print("Creating sample data for three feature selection methods...")
    
    # Create sample data
    method_features, methods = create_sample_feature_data()
    
    print(f"Created data for {len(methods)} methods:")
    for method_name, description in methods.items():
        print(f"  - {description}")
    
    # Create Venn diagrams
    create_venn_diagrams(method_features, methods)
    
    # Create feature importance comparison plots
    create_feature_importance_comparison_plots(method_features, methods)
    
    print("\n=== Demo Complete ===")
    print("Visualizations have been created and saved as PNG files.")
    print("Key visualizations created:")
    print("  - Venn diagrams showing feature overlaps between methods")
    print("  - Feature importance comparison plots (SHAP vs MDI)")
    print("  - Correlation scatter plots")
    print("  - Overlap bar plots (fallback)")

if __name__ == "__main__":
    main()
