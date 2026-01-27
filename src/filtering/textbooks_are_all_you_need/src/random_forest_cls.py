import os
import numpy as np
import joblib
import pandas as pd
from typing import Tuple, Dict, Any, Union, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class RandomForestTrainer:
    """
    Simple trainer for Random Forest classifier.
    Handles data splitting, training, evaluation, and model persistence.
    No hyperparameter tuning – uses provided or default model parameters.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        model_path: str = "best_random_forest_model.joblib",
        datasets_path: str = "datasets.npz",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        test_size : float
            Proportion of test split (default: 0.2).
        random_state : int
            Seed for reproducibility (default: 42).
        model_path : str
            Path to save the trained model (default: "best_random_forest_model.joblib").
        datasets_path : str
            Path to save train/test splits (default: "datasets.npz").
        model_params : dict or None
            Parameters passed to RandomForestClassifier (e.g., n_estimators, max_depth).
            If None, uses scikit-learn defaults except for random_state and n_jobs.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model_path = model_path
        self.datasets_path = datasets_path
        self.model_params = model_params or {}
        self.model: Optional[RandomForestClassifier] = None
        self.X_test = None
        self.y_test = None

    def train(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Tuple[
        RandomForestClassifier,
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Dict[str, Any],
    ]:
        """
        Train a Random Forest model and evaluate on test set.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Feature matrix.
        y : array-like or pandas Series
            Target vector.

        Returns
        -------
        model : RandomForestClassifier
            Trained model.
        datasets : tuple
            (X_train, X_test, y_train, y_test) as numpy arrays.
        metadata : dict
            Training metadata including test metrics and model parameters.
        """
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

        # Stratified train-test split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X_array,
            y_array,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_array,
        )

        # Initialize and train model
        rf_params = {
            "random_state": self.random_state,
            "n_jobs": -1,
            **self.model_params,
        }
        self.model = RandomForestClassifier(**rf_params)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_report = classification_report(self.y_test, y_pred, output_dict=True)

        # Save artifacts
        self._save_model()
        self._save_datasets(X_train, self.X_test, y_train, self.y_test)

        # Prepare metadata
        metadata = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "model_path": self.model_path,
            "datasets_path": self.datasets_path,
            "test_accuracy": test_accuracy,
            "test_classification_report": test_report,
            "feature_importances": self.model.feature_importances_,
            "n_features": X_array.shape[1],
            "n_classes": len(np.unique(y_array)),
            "model_params_used": self.model.get_params(),
        }

        datasets = (X_train, self.X_test, y_train, self.y_test)
        return self.model, datasets, metadata

    def _save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def _save_datasets(self, X_train, X_test, y_train, y_test):
        """Save train/test splits to a compressed numpy file."""
        np.savez(
            self.datasets_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def load_model(self, model_path: Optional[str] = None) -> RandomForestClassifier:
        """
        Load a pre-trained Random Forest model.

        Parameters
        ----------
        model_path : str or None
            Path to the model file. If None, uses the path from initialization.

        Returns
        -------
        RandomForestClassifier
            Loaded model.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ValueError
            If the loaded object is not a RandomForestClassifier.
        """
        path = model_path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at path: {path}")

        loaded_model = joblib.load(path)
        if not isinstance(loaded_model, RandomForestClassifier):
            raise ValueError("Loaded object is not a RandomForestClassifier")

        self.model = loaded_model
        return self.model
