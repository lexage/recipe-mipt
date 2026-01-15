
import os
import numpy as np
import joblib
import pandas as pd
from typing import Tuple, Dict, Any, Union, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint


class RandomForestTrainer:
    """
    A class for training, tuning, evaluating, and saving the best Random Forest model.
    Follows Kaggle best practices: hyperparameter tuning, test evaluation, and model persistence.
    """

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 model_path: str = 'best_random_forest_model.joblib',
                 datasets_path: str = 'datasets.npz',
                 cv_folds: int = 5,
                 n_iter_search: int = 20,
                 scoring: str = 'f1_macro',  # or 'accuracy', 'roc_auc', etc.
                 use_random_search: bool = True):
        """
        Initialize the trainer with configuration.

        Parameters
        ----------
        test_size : float
            Proportion of test split.
        random_state : int
            For reproducibility.
        model_path : str
            Path to save the best model.
        datasets_path : str
            Path to save train/test splits.
        cv_folds : int
            Number of cross-validation folds.
        n_iter_search : int
            Number of parameter settings sampled (for RandomizedSearchCV).
        scoring : str
            Scoring metric for model selection.
        use_random_search : bool
            If True, uses RandomizedSearchCV; otherwise, you can plug in GridSearchCV manually.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model_path = model_path
        self.datasets_path = datasets_path
        self.cv_folds = cv_folds
        self.n_iter_search = n_iter_search
        self.scoring = scoring
        self.use_random_search = use_random_search
        self.best_model: Optional[RandomForestClassifier] = None
        self.X_test = None
        self.y_test = None

    def train(self,
              X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series]
              ) -> Tuple[RandomForestClassifier,
                         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         Dict[str, Any]]:
        """
        Train and tune a Random Forest model, evaluate on test set, and save the best model.
        """
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Split data (stratified)
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X_array, y_array,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_array
        )

        # Define base model
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

        # Define hyperparameter space
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(range(5, 31, 5)),
            'min_samples_split': randint(2, 21),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None]
        }

        # Perform hyperparameter search
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=self.n_iter_search,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )

        search.fit(X_train, y_train)

        # Save best model
        self.best_model = search.best_estimator_

        # Evaluate on test set
        y_pred = self.best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_report = classification_report(self.y_test, y_pred, output_dict=True)

        # Save artifacts
        self._save_model()
        self._save_datasets(X_train, self.X_test, y_train, self.y_test)

        # Metadata
        metadata = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'model_path': self.model_path,
            'datasets_path': self.datasets_path,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_accuracy': test_accuracy,
            'test_classification_report': test_report,
            'feature_importances': self.best_model.feature_importances_,
            'n_features': X_array.shape[1],
            'n_classes': len(np.unique(y_array))
        }

        datasets = (X_train, self.X_test, y_train, self.y_test)
        return self.best_model, datasets, metadata

    def _save_model(self):
        joblib.dump(self.best_model, self.model_path)

    def _save_datasets(self, X_train, X_test, y_train, y_test):
        np.savez(self.datasets_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    def load_model(self, model_path: Optional[str] = None) -> RandomForestClassifier:
        """
        Load a pre-trained Random Forest model from a file.
        
        Parameters
        ----------
        model_path : Optional[str], default=None
            Path to the saved model file. If None, uses the model_path from initialization.
        
        Returns
        -------
        RandomForestClassifier
            The loaded Random Forest model.
        
        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the specified path.
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
