"""Non-gaussian converter
"""
import matplotlib.pyplot as plt
import numpy as np
from dtwalign import dtw
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kde import KDEUnivariate

class LinearTransformer():
    """Transform data linearly
    used for indices of cdf
    """
    def __init__(self,min_value=0,max_value=1000):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self,X,y=None):
        self.a = (self.max_value - self.min_value)/(2*(X.max() - X.min()))
        self.b = self.max_value - self.a * (X.max() + (X.max()-X.min())/2)
        return self


    def transform(self,X):
        X_transformed = self.__lineq(X)
        return X_transformed

    def inverse_transform(self,X):
        X_inv = self.__lineq_inv(X)
        return X_inv

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def __lineq(self,X):
        Xr = self.a * X + self.b
        return Xr

    def __lineq_inv(self,X):
        Xr = 1/self.a * X - self.b/self.a
        return Xr


class GaussianTransformer(TransformerMixin,BaseEstimator):
    """Transform non-gaussian data to gaussian for each variable
    """
    def __init__(self,resolution=1000):
        self.target_cdfs = []
        self.gaussian_cdf = norm().cdf(np.linspace(-5,5,resolution))
        self.lin_tfs = []
        self.lin_tf_gcdf = LinearTransformer(max_value=resolution).fit(np.array([-5,5]))
        self.warping_funcs = []
        self.inv_warping_funcs = []
        self.resolution = resolution
        self._dtw_res = []

    def fit(self,X,y=None):
        # check and convert to 2D array
        X = self.__to_2D(X)
        self.is_constant = np.zeros(X.shape[1],dtype=bool)
        for vidx in range(X.shape[1]):
            x = X[:,vidx]
            if np.abs(np.diff(x)).sum() < 1e-10:
                """
                if the target value is almost constant,
                the value will be not converted.
                """
                self.is_constant[vidx] = True
                self.lin_tfs.append(None)
                self.target_cdfs.append(None)
                self._dtw_res.append(None)
                self.warping_funcs.append(None)
                self.inv_warping_funcs.append(None)
                continue
            xmin = x.min(); xmax = x.max()
            xrange = xmax - xmin
            # create instance for index transformation
            mt = LinearTransformer(min_value=0,max_value=self.resolution).fit(x)
            self.lin_tfs.append(mt)
            # estimate pdf by kde
            kde = KDEUnivariate(x)
            kde.fit()
            kde.fit(bw=kde.bw*0.01)  # re-fit with tighter bandwidth
            # calculate cdf of target variable
            cdf = kde.evaluate(np.linspace(xmin-xrange/2,xmax+xrange/2,self.resolution)).cumsum()
            cdf /= cdf.max()
            self.target_cdfs.append(cdf)
            # get correspondence between target cdf and gaussian cdf
            res = dtw(cdf,self.gaussian_cdf,step_pattern="symmetric2")
            self._dtw_res.append(res)
            warping_func = interp1d(res.path[:,0],res.path[:,1],kind="linear",\
                bounds_error=False,fill_value=(0,self.resolution-1))
            inv_warping_func = interp1d(res.path[:,1],res.path[:,0],kind="linear",\
                bounds_error=False,fill_value=(0,self.resolution-1))
            self.warping_funcs.append(warping_func)
            self.inv_warping_funcs.append(inv_warping_func)
        return self

    def transform(self,X,y=None):
        # check and convert to 2D array
        X = self.__to_2D(X)
        X_r = np.zeros_like(X)
        for vidx in range(X.shape[1]):
            if self.is_constant[vidx]:
                X_r[:,vidx] = X[:,vidx]
                continue
            x = X[:,vidx]
            # get target cdf index
            x_idcs = self.lin_tfs[vidx].transform(x)
            # get gaussian cdf index
            x_gauss_idcs = self.warping_funcs[vidx](x_idcs)
            # convert to real value (normalized value)
            X_r[:,vidx] = self.lin_tf_gcdf.inverse_transform(x_gauss_idcs)
        return X_r

    def inverse_transform(self,X,y=None):
        # check and convert to 2D array
        X = self.__to_2D(X)
        X_inv = np.zeros_like(X)
        for vidx in range(X.shape[1]):
            if self.is_constant[vidx]:
                X_inv[:,vidx] = X[:,vidx]
                continue
            x = X[:,vidx]
            # get gaussian cdf index
            x_gauss_idcs = self.lin_tf_gcdf.transform(x)
            # get target cdf index
            x_idcs = self.inv_warping_funcs[vidx](x_gauss_idcs)
            # convert to real value (original space)
            X_inv[:,vidx] = self.lin_tfs[vidx].inverse_transform(x_idcs)
        return X_inv

    def plot_correspondence(self,var_index):
        res = self._dtw_res[var_index]
        plt.figure()
        plt.plot(self.gaussian_cdf,label="gaussian cdf")
        plt.plot(self.target_cdfs[var_index][res.get_warping_path()],\
            label="warped target cdf")
        plt.legend()
        plt.show()

    @classmethod
    def __to_2D(cls,X):
        """check and convert to 2D array
        """
        if X.ndim == 1:
            X = X[:,None]
        return X.astype(float)
