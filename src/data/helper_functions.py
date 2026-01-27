from random import randint

from torch import Tensor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def metalearning_binary_target_changer(labels: Tensor) -> Tensor:
    """Change the binary labels randomly.

    Args:
        labels: The binary labels to change.

    Returns:
        The changed binary labels.
    """
    to_change = randint(0, 1)
    labels = (labels + to_change) % 2
    return labels

def select_features_by_explained_variance(X, variance_threshold=0.90, n_components=200, weighting='sqrt'):
    """
    Select features based on their contributions until they explain a target percentage of variance.
    """    
    # Run PCA
    # n_components = min(n_components, min(X.shape))
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Calculate loadings and weight by explained variance
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # Calculate weights
    if weighting == 'linear':
        weights = pca.explained_variance_ratio_
    elif weighting == 'sqrt':
        weights = np.sqrt(pca.explained_variance_ratio_)
    elif weighting == 'log':
        weights = np.log1p(pca.explained_variance_ratio_ * 10)
    else:
        weights = np.ones(len(pca.explained_variance_ratio_))

    # Apply weights
    weighted_loadings = loadings * weights
    
    # Calculate importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(weighted_loadings).sum(axis=1)
    }).sort_values('Importance', ascending=False)
    
    # Calculate cumulative importance
    total_importance = importance['Importance'].sum()
    importance['Relative_Importance'] = importance['Importance'] / total_importance
    importance['Cumulative_Variance'] = importance['Relative_Importance'].cumsum()
    
    # Select features until threshold is reached
    selected_features_df = importance[importance['Cumulative_Variance'] <= variance_threshold]
    
    # Add one more feature to cross the threshold if needed
    if len(selected_features_df) < len(importance) and selected_features_df['Cumulative_Variance'].iloc[-1] < variance_threshold:
        selected_features_df = pd.concat([
            selected_features_df, 
            importance.iloc[len(selected_features_df):len(selected_features_df)+1]
        ])
    
    selected_features = selected_features_df['Feature'].tolist()
    
    # Calculate PC contributions
    pc_contributions = pd.DataFrame(
        {f'PC{i+1}_contrib': np.abs(loadings[f'PC{i+1}']) * pca.explained_variance_ratio_[i] 
         for i in range(n_components)}
    )
    pc_contributions.index = X.columns
    pc_contributions['Total_importance'] = importance.set_index('Feature')['Importance']
    
    return selected_features, selected_features_df, pc_contributions.sort_values('Total_importance', ascending=False), pca


