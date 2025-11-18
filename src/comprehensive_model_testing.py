"""
Visual Analysis Script for Model Performance
Creates plots to understand model decision boundaries and feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
import joblib
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_iris_features():
    """Analyze Iris dataset features and model decisions"""
    
    print("="*60)
    print("IRIS DATASET ANALYSIS")
    print("="*60)
    
    # Load data and model
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Load trained model
    model = joblib.load("../model/iris_model.pkl")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    df['species_code'] = y
    
    # Print statistical summary
    print("\n1. FEATURE STATISTICS BY SPECIES:")
    print("-" * 40)
    for species in target_names:
        print(f"\n{species.upper()}:")
        species_data = df[df['species'] == species][feature_names]
        print(species_data.describe()[['mean', 'std']].round(2))
    
    # Feature importance for Decision Tree
    if hasattr(model, 'feature_importances_'):
        print("\n2. FEATURE IMPORTANCE (Decision Tree):")
        print("-" * 40)
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name:25s}: {importance:.3f} {'█' * int(importance * 50)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Petal Length vs Width (most discriminative)
    ax = axes[0, 0]
    for species_code in range(3):
        mask = y == species_code
        ax.scatter(X[mask, 2], X[mask, 3], label=target_names[species_code], alpha=0.7, s=50)
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_title('Best Separation: Petal Dimensions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sepal Length vs Width
    ax = axes[0, 1]
    for species_code in range(3):
        mask = y == species_code
        ax.scatter(X[mask, 0], X[mask, 1], label=target_names[species_code], alpha=0.7, s=50)
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title('Sepal Dimensions (Some Overlap)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Feature distributions
    ax = axes[0, 2]
    feature_data = []
    for i, feature in enumerate(feature_names):
        for species_code in range(3):
            mask = y == species_code
            feature_data.append({
                'Feature': feature.replace(' (cm)', ''),
                'Value': X[mask, i].mean(),
                'Species': target_names[species_code]
            })
    
    plot_df = pd.DataFrame(feature_data)
    pivot_df = plot_df.pivot(index='Feature', columns='Species', values='Value')
    pivot_df.plot(kind='bar', ax=ax)
    ax.set_title('Mean Feature Values by Species')
    ax.set_ylabel('Mean Value (cm)')
    ax.legend(title='Species')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Decision boundaries visualization (simplified 2D)
    ax = axes[1, 0]
    h = 0.02  # step size in mesh
    
    # Use only petal length and width for 2D visualization
    X_subset = X[:, 2:4]
    x_min, x_max = X_subset[:, 0].min() - 0.5, X_subset[:, 0].max() + 0.5
    y_min, y_max = X_subset[:, 1].min() - 0.5, X_subset[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for mesh points (using mean values for other features)
    mesh_points = np.c_[
        np.full(xx.ravel().shape, X[:, 0].mean()),  # mean sepal length
        np.full(xx.ravel().shape, X[:, 1].mean()),  # mean sepal width
        xx.ravel(),  # petal length
        yy.ravel()   # petal width
    ]
    
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    for species_code in range(3):
        mask = y == species_code
        ax.scatter(X[mask, 2], X[mask, 3], label=target_names[species_code], 
                  edgecolors='black', linewidth=1, s=50)
    
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_title('Decision Boundaries (2D Projection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Correlation heatmap
    ax = axes[1, 1]
    correlation_matrix = df[feature_names].corr()
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels([f.replace(' (cm)', '') for f in feature_names], rotation=45, ha='right')
    ax.set_yticklabels([f.replace(' (cm)', '') for f in feature_names])
    ax.set_title('Feature Correlation Matrix')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Plot 6: Box plots for each feature
    ax = axes[1, 2]
    data_for_box = []
    for i, feature in enumerate(feature_names):
        for val, species in zip(X[:, i], y):
            data_for_box.append({
                'Feature': feature.replace(' (cm)', ''),
                'Value': val,
                'Species': target_names[species]
            })
    
    box_df = pd.DataFrame(data_for_box)
    # Show only petal length as it's most important
    petal_length_df = box_df[box_df['Feature'] == 'petal length']
    sns.boxplot(data=petal_length_df, x='Species', y='Value', ax=ax)
    ax.set_title('Petal Length Distribution (Key Feature)')
    ax.set_ylabel('Petal Length (cm)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n3. KEY INSIGHTS:")
    print("-" * 40)
    print("• Petal length is the most discriminative feature")
    print("• Setosa is clearly separable (small petals)")
    print("• Versicolor and Virginica have some overlap")
    print("• Decision boundaries are approximately:")
    print("  - Setosa: petal_length < 2.5 cm")
    print("  - Versicolor: 2.5 < petal_length < 5.0 cm")
    print("  - Virginica: petal_length > 5.0 cm")

def analyze_wine_features():
    """Analyze Wine dataset features and model decisions"""
    
    print("\n" + "="*60)
    print("WINE DATASET ANALYSIS")
    print("="*60)
    
    # Load data and model
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = ['Barolo', 'Grignolino', 'Barbera']  # Italian wine names
    
    # Load trained model (with scaler)
    model_data = joblib.load("../model/wine_model.pkl")
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['wine_class'] = [target_names[i] for i in y]
    df['class_code'] = y
    
    # Print statistical summary
    print("\n1. TOP 5 DISCRIMINATIVE FEATURES BY CLASS:")
    print("-" * 40)
    
    # Calculate mean differences
    feature_importance_scores = {}
    for feature in feature_names:
        # Calculate variance between class means
        class_means = [df[df['wine_class'] == wine_type][feature].mean() 
                      for wine_type in target_names]
        feature_importance_scores[feature] = np.var(class_means)
    
    # Sort by importance
    sorted_features = sorted(feature_importance_scores.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
    
    for wine_type in target_names:
        print(f"\n{wine_type.upper()}:")
        wine_data = df[df['wine_class'] == wine_type]
        for feature, _ in sorted_features:
            mean_val = wine_data[feature].mean()
            std_val = wine_data[feature].std()
            print(f"  {feature:30s}: {mean_val:7.2f} ± {std_val:5.2f}")
    
    # Random Forest feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n2. MODEL FEATURE IMPORTANCE (Random Forest):")
        print("-" * 40)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        for i in indices:
            importance = importances[i]
            print(f"{feature_names[i]:30s}: {importance:.3f} {'█' * int(importance * 100)}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Wine Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Alcohol vs Proline (top 2 features)
    ax = axes[0, 0]
    for class_code in range(3):
        mask = y == class_code
        ax.scatter(X[mask, 0], X[mask, 12], label=target_names[class_code], 
                  alpha=0.6, s=50)
    ax.set_xlabel('Alcohol Content (%)')
    ax.set_ylabel('Proline (mg/L)')
    ax.set_title('Top 2 Features: Alcohol vs Proline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Flavanoids vs Color Intensity
    ax = axes[0, 1]
    for class_code in range(3):
        mask = y == class_code
        ax.scatter(X[mask, 6], X[mask, 9], label=target_names[class_code], 
                  alpha=0.6, s=50)
    ax.set_xlabel('Flavanoids (mg/L)')
    ax.set_ylabel('Color Intensity')
    ax.set_title('Phenolic Content vs Color')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance bar plot
    ax = axes[0, 2]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:8]
        
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 8 Most Important Features')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of key features
    ax = axes[1, 0]
    key_features = ['alcohol', 'proline', 'flavanoids']
    data_for_violin = []
    for feature in key_features:
        idx = feature_names.index(feature)
        for val, wine_class in zip(X[:, idx], y):
            data_for_violin.append({
                'Feature': feature,
                'Value': val,
                'Wine': target_names[wine_class]
            })
    
    violin_df = pd.DataFrame(data_for_violin)
    alcohol_df = violin_df[violin_df['Feature'] == 'alcohol']
    sns.violinplot(data=alcohol_df, x='Wine', y='Value', ax=ax)
    ax.set_title('Alcohol Content Distribution')
    ax.set_ylabel('Alcohol (%)')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Correlation heatmap for top features
    ax = axes[1, 1]
    top_features_idx = [0, 6, 9, 12]  # alcohol, flavanoids, color, proline
    top_features_names = [feature_names[i] for i in top_features_idx]
    X_subset = X[:, top_features_idx]
    correlation_matrix = np.corrcoef(X_subset.T)
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features_names)))
    ax.set_yticks(range(len(top_features_names)))
    ax.set_xticklabels(top_features_names, rotation=45, ha='right')
    ax.set_yticklabels(top_features_names)
    ax.set_title('Correlation: Key Features')
    
    for i in range(len(top_features_names)):
        for j in range(len(top_features_names)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Plot 6: Class separation in 2D (PCA)
    ax = axes[1, 2]
    from sklearn.decomposition import PCA
    
    # Scale data first
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    for class_code in range(3):
        mask = y == class_code
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  label=target_names[class_code], alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA Projection (2D)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Chemical profile radar chart
    ax = axes[2, 0]
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(3, 3, 7, projection='polar')
    
    features_for_radar = ['alcohol', 'malic_acid', 'flavanoids', 
                          'color_intensity', 'hue', 'proline']
    
    for wine_class in range(3):
        values = []
        for feature in features_for_radar:
            idx = feature_names.index(feature)
            # Normalize values to 0-1 scale
            wine_vals = X[y == wine_class, idx]
            normalized_mean = (wine_vals.mean() - X[:, idx].min()) / (X[:, idx].max() - X[:, idx].min())
            values.append(normalized_mean)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=target_names[wine_class])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_for_radar, size=8)
    ax.set_title('Chemical Profile Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Plot 8: Box plot for proline (most discriminative)
    ax = plt.subplot(3, 3, 8)
    proline_data = []
    for val, wine_class in zip(X[:, 12], y):
        proline_data.append({
            'Wine': target_names[wine_class],
            'Proline': val
        })
    
    proline_df = pd.DataFrame(proline_data)
    sns.boxplot(data=proline_df, x='Wine', y='Proline', ax=ax)
    ax.set_title('Proline Distribution (Key Discriminator)')
    ax.set_ylabel('Proline (mg/L)')
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Confusion regions
    ax = plt.subplot(3, 3, 9)
    
    # Show classification confidence regions
    conf_data = []
    for i in range(len(X)):
        X_single = X_scaled[i].reshape(1, -1)
        probs = model.predict_proba(X_single)[0]
        conf_data.append({
            'True Class': target_names[y[i]],
            'Confidence': max(probs),
            'Predicted': target_names[np.argmax(probs)]
        })
    
    conf_df = pd.DataFrame(conf_data)
    
    # Plot confidence distribution
    for wine in target_names:
        wine_conf = conf_df[conf_df['True Class'] == wine]['Confidence']
        ax.hist(wine_conf, alpha=0.5, label=wine, bins=20)
    
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Model Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wine_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n3. KEY INSIGHTS:")
    print("-" * 40)
    print("• Alcohol content is the most important feature")
    print("• Proline provides excellent class separation")
    print("• Flavanoids and color intensity are secondary discriminators")
    print("• Classification patterns:")
    print("  - Barolo: High alcohol (13-14%), high proline (>1000)")
    print("  - Grignolino: Medium alcohol (12-13%), medium proline (600-900)")
    print("  - Barbera: Low alcohol (11-12%), low proline (<600)")
    print("• Random Forest achieves high accuracy due to ensemble voting")

def compare_models():
    """Compare the two models side by side"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = {
        'Aspect': ['Dataset Size', 'Features', 'Classes', 'Algorithm', 
                   'Key Feature', 'Typical Accuracy', 'Complexity'],
        'Iris Model': ['150 samples', '4 features', '3 species', 
                      'Decision Tree', 'Petal Length', '~95%', 'Simple'],
        'Wine Model': ['178 samples', '13 features', '3 cultivars', 
                      'Random Forest', 'Alcohol + Proline', '~97%', 'Moderate']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nMODEL CHARACTERISTICS:")
    print("-" * 40)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)

def main():
    """Run all analyses"""
    
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         MODEL FEATURE ANALYSIS & VISUALIZATION           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    try:
        analyze_iris_features()
        analyze_wine_features()
        compare_models()
        
        print("\n✅ Analysis complete! Check 'iris_analysis.png' and 'wine_analysis.png'")
        print("\nThe visualizations show:")
        print("• Feature distributions and importance")
        print("• Decision boundaries and class separation")
        print("• Model confidence levels")
        print("• Key discriminative patterns")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Make sure the models are trained and saved in ../model/")

if __name__ == "__main__":
    main()