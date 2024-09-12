from sklearn.decomposition import PCA
from openTSNE.sklearn import TSNE
from umap import UMAP
from ugtm import eGTM as ScikitLearnGTM
from cdr.optimization.params import DimReducerParams
import numpy as np
from typing import Optional, Any, Dict


class DimReducer:
    """Dimensionality reduction class supporting multiple methods."""

    def __init__(self, params: DimReducerParams):
        self.params = params
        self.method = self.params.method
        self.model_params = self._merge_params_with_defaults()
        self._initialize_model()

    @staticmethod
    def default_params() -> Dict[str, Dict[str, Any]]:
        """Returns default parameters for each method."""
        return {
            'PCA': {'n_components': 2},
            'UMAP': {'n_components': 2},
            't-SNE': {'n_components': 2, 'verbose': False},
            'GTM': {'n_components': 2, 'num_nodes': 225, 'num_basis_functions': 169, 'basis_width': 1.1, 'reg_coeff': 1,
                    'standardize': False}
        }

    @staticmethod
    def valid_methods() -> Dict[str, Any]:
        """Returns valid methods for dimensionality reduction."""
        return {
            'PCA': PCA,
            'UMAP': UMAP,
            't-SNE': TSNE,
            'GTM': ScikitLearnGTM  # Default to BishopGTM unless specified otherwise
        }

    def _merge_params_with_defaults(self) -> Dict[str, Any]:
        """Merge user parameters with default parameters."""
        default_params = self.default_params().get(self.method, {})
        user_params = {k: v for k, v in self.params.__dict__.items() if v is not None and k != 'method'}
        merged_params = {**default_params, **user_params}
        return merged_params

    def _initialize_model(self):
        """Initialize the model based on the method and parameters."""
        model_class = self.valid_methods().get(self.method, None)
        if not model_class:
            raise ValueError(
                f"Invalid method '{self.method}'. Supported methods are: {', '.join(self.valid_methods().keys())}")

        if self.method == 'GTM':
            self.model = self._gtm_preprocessing()
        else:
            self.model = model_class(**self.model_params)

    def _gtm_preprocessing(self):
        """Preprocess GTM model parameters."""
        if self.model_params.get('kernel', False):
            return KernelGTM(**self.model_params)
        else:
            return ScikitLearnGTM(**self.model_params)

    def update_params(self, **new_params: Any):
        """Update parameters and reinitialize the model."""
        self.model_params.update(new_params)
        self._initialize_model()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the model."""
        return self.model.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        return self.model.fit_transform(X, y)

    @staticmethod
    def check_method_implemented(method: str):
        """Check if the given method is implemented."""
        implemented_methods = ['PCA', 't-SNE', 'UMAP', 'GTM']
        if method not in implemented_methods:
            raise ValueError(
                f"Method '{method}' is not implemented. Available methods: {', '.join(implemented_methods)}")
