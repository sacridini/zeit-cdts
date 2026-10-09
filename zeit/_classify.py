import numpy as np
from sklearn.ensemble import RandomForestClassifier
import rasterio
from typing import Optional

def train_ccdc_classifier(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier for CCDC land cover classification.
    X_train: array-like of shape (n_samples, n_features). Features should be the harmonic coefficients and RMSE.
    y_train: array-like of shape (n_samples,). The land cover class labels.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def classify_ccdc_stack(clf: RandomForestClassifier, coef_stack_path: str, output_path: str, chunk_size: int = 512) -> None:
    """
    Applies the trained Random Forest classifier to a full CCDC coefficient GeoTIFF stack.
    Assumes the model was trained on the exact band configuration present in the TIFF.
    """
    with rasterio.open(coef_stack_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8", nodata=0)
        
        with rasterio.open(output_path, "w", **profile) as dst:
            for row in range(0, src.height, chunk_size):
                for col in range(0, src.width, chunk_size):
                    window = rasterio.windows.Window(
                        col, row, 
                        min(chunk_size, src.width - col), 
                        min(chunk_size, src.height - row)
                    )
                    
                    data = src.read(window=window)
                    n_features, h, w = data.shape
                    
                    data_reshaped = data.transpose(1, 2, 0).reshape(-1, n_features)
                    
                    mask = (data_reshaped[:, 0] != 0) | (data_reshaped[:, 1] != 0)
                    
                    predictions = np.zeros(h * w, dtype=np.uint8)
                    
                    if np.any(mask):
                        preds = clf.predict(data_reshaped[mask])
                        predictions[mask] = preds
                        
                    out_img = predictions.reshape(1, h, w)
                    dst.write(out_img, window=window)
                    
    print(f"Classification saved to {output_path}")

