import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADDaMM:
    def __init__(self, X, bandwidth='auto', kernel='gaussian'):
        self.logger = logger
        self._bwSearch = None
        self._bandwidth = 0.95
        self._kernel = kernel
        self._likelihoods = None
        self._hiThreshold = None
        self._loThreshold = None
        self.fit(X)

    def tune(self, X):
        # Define grid params
        gridParams = {
            'bandwidth': [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            'kernel': ['gaussian', 'tophat', 'exponential'],
        }
        kde = KernelDensity(bandwidth=1,kernel='gaussian')
        gs = GridSearchCV(kde, gridParams, cv=5)
        gs.fit(X)
        self.logger.info(f'Best Parameters : {gs.best_params_}')
        self.logger.info(f'Cross Validation mean score : {gs.best_score_}')
        return gs.best_estimator_

    def selectThreshold(self, X):
        # Compute threshold from likelihood array
        self._likelihoods = self.kde.score_samples(X)
        minLikelihood = min(self._likelihoods)
        if minLikelihood > 0:
            self._hiThreshold = minLikelihood * 0.85
            self._loThreshold = minLikelihood * 0.50
        else:
            self._hiThreshold = minLikelihood - abs(minLikelihood) * 0.012
            self._loThreshold = minLikelihood - abs(minLikelihood) * 0.036

    def fit(self, X):
        if self._bwSearch:
            self.kde = self.tune(X)
        else:
            self.kde = KernelDensity(bandwidth=0.95 ,kernel=self._kernel)
            self.kde.fit(X)
        self.selectThreshold(X)

    def predict(self, X):
        # Compute kernel density
        likelihood = self.kde.score_samples(X)
        self.logger.info(
            f'Current likelihood {likelihood}, lo {self._loThreshold}, hi {self._hiThreshold}',
        )
        # Create array of no outliers
        outliers = np.zeros(len(likelihood))
        # Outliers are indexes where likelihood is between low and hi threshold
        outliers[
            np.where(
                (likelihood >= self._loThreshold) & (likelihood < self._hiThreshold)
            )
        ] = -1
        # Outliers are indexes where likelihood is below low threshold
        outliers[np.where(likelihood < self._loThreshold)] = -2
        return outliers




class SimplifiedADDaMM:
    def __init__(self, bandwidth=0.05):
        self.bandwidth = bandwidth

    def fit(self, X):
        self.kde = KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(X)

    def detect(self, x):
        log_prob = self.kde.score_samples(x)
        return log_prob
