def add_combined_feature(X):
    """
    Feature engineering function: ajoute une feature combinée
    'Combined_radius_texture' = 'mean radius' * 'mean texture'.
    """
    X = X.copy()
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X