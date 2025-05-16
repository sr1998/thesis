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

def select_features_by_pc_loadings(X, n_features=500, n_components=1000, weighting='sqrt'):
    """
    Select features based on their loadings across multiple principal components.
    """    
    # Run PCA
    n_components = min(n_components, min(X.shape))
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

    # Apply weights to all columns at once using broadcasting
    weighted_loadings = loadings * weights
    
    # Calculate importance across components
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(weighted_loadings).sum(axis=1)
    }).sort_values('Importance', ascending=False)
    
    # Select top features
    selected_features = importance.head(n_features)['Feature'].tolist()
    
    # Calculate PC contributions to importance
    pc_contributions = pd.DataFrame(
        {f'PC{i+1}_contrib': np.abs(loadings[f'PC{i+1}']) * pca.explained_variance_ratio_[i] 
         for i in range(n_components)}
    )
    pc_contributions.index = X.columns
    pc_contributions['Total_importance'] = importance.set_index('Feature')['Importance']
    
    return selected_features, pc_contributions.sort_values('Total_importance', ascending=False), pca
