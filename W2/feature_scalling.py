import numpy as np

def main():
    ...

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # Find the mean of each column/feature
    mu = np.mean(X, axis=0)                 # mu will have shape (n,)
    # Find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)                  # sigma will have shape (n,)
    # Element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)

main()