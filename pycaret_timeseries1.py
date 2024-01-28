
# https://nbviewer.org/github/pycaret/pycaret/blob/master/tutorials/time_series/forecasting/univariate_with_exogeneous_part1.ipynb

import numpy as np
import pandas as pd
import pdb
from pycaret.datasets import get_data
from pycaret.time_series import *

data = get_data("airquality")
data["index"] = pd.to_datetime(data["Date"] + " " + data["Time"])
data.drop(columns=["Date", "Time"], inplace=True)
target = "CO(GT)"

# Replace tharget value=-200 with NaN, later we will impute these Nan values in pycaret.
data[data[target] == -200].head()
data.replace(-200, np.nan, inplace=True)


#-------- Initial stats analysis -----------------------------------------

eda = TSForecastingExperiment()
eda.setup(
    data=data,
    target=target,
    index="index",  # time column
    fh=48,

    numeric_imputation_target="ffill",
    numeric_imputation_exogenous="ffill",
    
    #fig_kwargs=global_fig_settings,  # Set defaults for the plots ----
    session_id=42,
)

#eda.plot_model(fig_kwargs={"renderer": "png", "width": 1000, "height": 1200})

# Plots original data with first difference (order d = 1) by default
# NOTE: Uncomment out display_format to use "plotly-widget".
eda.plot_model(
    plot="diff",
    fig_kwargs={
        "height": 900,
        # No need to pass show_dash kwargs when using a plotly-widget
        "resampler_kwargs": {"default_n_shown_samples": 1500 }
        },
    data_kwargs={"acf": True, "pacf": True, "periodogram": True},
    # display_format="plotly-widget",
    )

eda.plot_model(
    plot="diff",
    fig_kwargs={
        "height": 900,
        "resampler_kwargs": {
            "default_n_shown_samples": 1500,
            "show_dash": {"mode": "inline", "port": 8056},
        },
    },
    data_kwargs={
        "lags_list": [1, [1, 24]],
        "acf": True,
        "pacf": True,
        "periodogram": True,
    },
    # display_format='plotly-dash',
    )


pdb.set_trace()


'''
def setup(
    data: Series | DataFrame = None,
    data_func: (() -> (Series | DataFrame)) | None = None,
    target: str | None = None,
    index: str | None = None,
    ignore_features: List | None = None,
    numeric_imputation_target: int | float | str | None = None,
    numeric_imputation_exogenous: int | float | str | None = None,
    transform_target: str | None = None,
    transform_exogenous: str | None = None,
    scale_target: str | None = None,
    scale_exogenous: str | None = None,
    fe_target_rr: list | None = None,
    fe_exogenous: list | None = None,
    fold_strategy: str | Any = "expanding",
    fold: int = 3,
    fh: List[int] | int | ndarray | ForecastingHorizon | None = 1,
    hyperparameter_split: str = "all",
    seasonal_period: List[int | str] | int | str | None = None,
    ignore_seasonality_test: bool = False,
    sp_detection: str = "auto",
    max_sp_to_consider: int | None = 60,
    remove_harmonics: bool = False,
    harmonic_order_method: str = "harmonic_max",
    num_sps_to_use: int = 1,
    seasonality_type: str = "mul",
    point_alpha: float | None = None,
    coverage: float | List[float] = 0.9,
    enforce_exogenous: bool = True,
    n_jobs: int | None = -1,
    use_gpu: bool = False,
    custom_pipeline: Any | None = None,
    html: bool = True,
    session_id: int | None = None,
    system_log: bool | str | Logger = True,
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False,
    experiment_name: str | None = None,
    experiment_custom_tags: Dict[str, Any] | None = None,
    log_plots: bool | list = False,
    log_profile: bool = False,
    log_data: bool = False,
    engine: Dict[str, str] | None = None,
    verbose: bool = True,
    profile: bool = False,
    profile_kwargs: Dict[str, Any] | None = None,
    fig_kwargs: Dict[str, Any] | None = None
) -> TSForecastingExperiment


data : pandas.Series or pandas.DataFrame = None
    Shape (n_samples, 1), when pandas.DataFrame, otherwise (n_samples, ).

data_func: Callable[[], Union[pd.Series, pd.DataFrame]] = None
    The function that generate data (the dataframe-like input). This is useful when the dataset is large, and you need parallel operations such as compare_models. It can avoid broadcasting large dataset from driver to workers. Notice one and only one of data and data_func must be set.

target : Optional[str], default = None
    Target name to be forecasted. Must be specified when data is a pandas DataFrame with more than 1 column. When data is a pandas Series or pandas DataFrame with 1 column, this can be left as None.

index: Optional[str], default = None
    Column name to be used as the datetime index for modeling. If 'index' column is specified & is of type string, it is assumed to be coercible to pd.DatetimeIndex using pd.to_datetime(). It can also be of type Int (e.g. RangeIndex, Int64Index), or DatetimeIndex or PeriodIndex in which case, it is processed appropriately. If None, then the data's index is used as is for modeling.

ignore_features: Optional[List], default = None
    List of features to ignore for modeling when the data is a pandas Dataframe with more than 1 column. Ignored when data is a pandas Series or Dataframe with 1 column.

numeric_imputation_target: Optional[Union[int, float, str]], default = None
    Indicates how to impute missing values in the target. If None, no imputation is done. If the target has missing values, then imputation is mandatory. If str, then value passed as is to the underlying sktime imputer. Allowed values are:
        "drift", "linear", "nearest", "mean", "median", "backfill", "bfill", "pad", "ffill", "random"
    If int or float, imputation method is set to "constant" with the given value.

numeric_imputation_exogenous: Optional[Union[int, float, str]], default = None
    Indicates how to impute missing values in the exogenous variables. If None, no imputation is done. If exogenous variables have missing values, then imputation is mandatory. If str, then value passed as is to the underlying sktime imputer. Allowed values are:
        "drift", "linear", "nearest", "mean", "median", "backfill", "bfill", "pad", "ffill", "random"
    If int or float, imputation method is set to "constant" with the given value.

transform_target: Optional[str], default = None
    Indicates how the target variable should be transformed. If None, no transformation is performed. Allowed values are
        "box-cox", "log", "sqrt", "exp", "cos"

transform_exogenous: Optional[str], default = None
    Indicates how the exogenous variables should be transformed. If None, no transformation is performed. Allowed values are
        "box-cox", "log", "sqrt", "exp", "cos"

scale_target: Optional[str], default = None
    Indicates how the target variable should be scaled. If None, no scaling is performed. Allowed values are
        "zscore", "minmax", "maxabs", "robust"

scale_exogenous: Optional[str], default = None
    Indicates how the exogenous variables should be scaled. If None, no scaling is performed. Allowed values are
        "zscore", "minmax", "maxabs", "robust"

fe_target_rr: Optional[list], default = None
    The transformers to be applied to the target variable in order to extract useful features. By default, None which means that the provided target variable are used "as is".
    NOTE: Most statistical and baseline models already use features (lags) for target variables implicitly. The only place where target features have to be created explicitly is in reduced regression models. Hence, this feature extraction is only applied to reduced regression models.

fe_exogenous : Optional[list] = None
    The transformations to be applied to the exogenous variables. These transformations are used for all models that accept exogenous variables. By default, None which means that the provided exogenous variables are used "as is".

fold_strategy: str or sklearn CV generator object, default = 'expanding'
    Choice of cross validation strategy. Possible values are:
	'expanding'
	'rolling' (same as/aliased to 'expanding')
	'sliding'
    You can also pass an sktime compatible cross validation object such as SlidingWindowSplitter or ExpandingWindowSplitter. In this case, the fold and fh parameters will be ignored and these values will be extracted from the fold_strategy object directly.

fold: int, default = 3
    Number of folds to be used in cross validation. Must be at least 2. This is a global setting that can be over-written at function level by using fold parameter. Ignored when fold_strategy is a custom object.

fh: Optional[int or list or np.array or ForecastingHorizon], default = 1
    The forecast horizon to be used for forecasting. Default is set to 1 i.e. forecast one point ahead. Valid options are:
    (1) Integer: When integer is passed it means N continuous points in
        the future without any gap.
    (2) List or np.array: Indicates points to predict in the future. e.g.
        fh = [1, 2, 3, 4] or np.arange(1, 5) will predict 4 points in the future.
    (3) If you want to forecast values with gaps, you can pass an list or array
        with gaps. e.g. np.arange([13, 25]) will skip the first 12 future points and forecast from the 13th point till the 24th point ahead (note in numpy right value is inclusive and left is exclusive).
    (4) Can also be a sktime compatible ForecastingHorizon object. (5) If fh = None, then fold_strategy must be a sktime compatible cross validation
        object. In this case, fh is derived from this object.

hyperparameter_split: str, default = "all"
    The split of data used to determine certain hyperparameters such as "seasonal_period", whether multiplicative seasonality can be used or not, whether the data is white noise or not, the values of non-seasonal difference "d" and seasonal difference "D" to use in certain models.
    Allowed values are: ["all", "train"].
    Refer for more details: https://github.com/pycaret/pycaret/issues/3202

seasonal_period: list or int or str, default = None
    Seasonal periods to use when performing seasonality checks (i.e. candidates).
    Users can provide seasonal_period by passing it as an integer or a string corresponding to the keys below (e.g. 'W' for weekly data, 'M' for monthly data, etc.). * B, C = 5 * D = 7 * W = 52 * M, BM, CBM, MS, BMS, CBMS = 12 * SM, SMS = 24 * Q, BQ, QS, BQS = 4 * A, Y, BA, BY, AS, YS, BAS, BYS = 1 * H = 24 * T, min = 60 * S = 60
    Users can also provide a list of such values to use in models that accept multiple seasonal values (currently TBATS). For models that don't accept multiple seasonal values, the first value of the list will be used as the seasonal period.
    NOTE: (1) If seasonal_period is provided, whether the seasonality check is performed or not depends on the ignore_seasonality_test setting. (2) If seasonal_period is not provided, then the candidates are detected per the sp_detection setting. If seasonal_period is provided, sp_detection setting is ignored.

ignore_seasonality_test: bool = False
    Whether to ignore the seasonality test or not. Applicable when seasonal_period is provided. If False, then a seasonality tests is performed to determine if the provided seasonal_period is valid or not. If it is found to be not valid, no seasonal period is used for modeling. If True, then the the provided seasonal_period is used as is.

sp_detection: str, default = "auto"
    If seasonal_period is None, then this parameter determines the algorithm to use to detect the seasonal periods to use in the models.
    Allowed values are ["auto" or "index"].
    If "auto", then seasonal periods are detected using statistical tests. If "index", then the frequency of the data index is mapped to a seasonal period as shown in seasonal_period.

max_sp_to_consider: Optional[int], default = 60,
    Max period to consider when detecting seasonal periods. If None, all periods up to int(("length of data"-1)/2) are considered. Length of the data is determined by hyperparameter_split setting.

remove_harmonics: bool, default = False
    Should harmonics be removed when considering what seasonal periods to use for modeling.

harmonic_order_method: str, default = "harmonic_max"
    Applicable when remove_harmonics = True. This determines how the harmonics are replaced. Allowed values are "harmonic_strength", "harmonic_max" or "raw_strength.
	If set to "harmonic_max", then lower seasonal period is replaced by its highest harmonic seasonal period in same position as the lower seasonal period.
	If set to "harmonic_strength", then lower seasonal period is replaced by its highest strength harmonic seasonal period in same position as the lower seasonal period.
	If set to "raw_strength", then lower seasonal periods is removed and the higher harmonic seasonal periods is retained in its original position based on its seasonal strength.
		e.g. Assuming detected seasonal periods in strength order are [2, 3, 4, 50] and remove_harmonics = True, then:
	If harmonic_order_method = "harmonic_max", result = [50, 3, 4]
	If harmonic_order_method = "harmonic_strength", result = [4, 3, 50]
	If harmonic_order_method = "raw_strength", result = [3, 4, 50]
	
num_sps_to_use: int, default = 1
    It determines the maximum number of seasonal periods to use in the models. Set to -1 to use all detected seasonal periods (in models that allow multiple seasonalities). If a model only allows one seasonal period and num_sps_to_use > 1, then the most dominant (primary) seasonal that is detected is used.

seasonality_type : str, default = "mul"
    The type of seasonality to use. Allowed values are ["add", "mul" or "auto"]
    The detection flow sequence is as follows: (1) If seasonality is not detected, then seasonality type is set to None. (2) If seasonality is detected but data is not strictly positive, then seasonality type is set to "add". (3) If seasonality_type is "auto", then the type of seasonality is determined using an internal algorithm as follows - If seasonality is detected, then data is decomposed using additive and multiplicative seasonal decomposition. Then seasonality type is selected based on seasonality strength per FPP (https://otexts.com/fpp2/seasonal-strength.html). NOTE: For Multiplicative, the denominator multiplies the seasonal and residual components instead of adding them. Rest of the calculations remain the same. If seasonal decomposition fails for any reason, then defaults to multiplicative seasonality. (4) Otherwise, seasonality_type is set to the user provided value.

point_alpha: Optional[float], default = None
    The alpha (quantile) value to use for the point predictions. By default this is set to None which uses sktime's predict() method to get the point prediction (the mean or the median of the forecast distribution). If this is set to a floating point value, then it switches to using the predict_quantiles() method to get the point prediction at the user specified quantile.
    Reference: https://robjhyndman.com/hyndsight/quantile-forecasts-in-r/
    NOTE: (1) Not all models support predict_quantiles(), hence, if a float value is provided, these models will be disabled. (2) Under some conditions, the user may want to only work with models that support prediction intervals. Utilizing note 1 to our advantage, the point_alpha argument can be set to 0.5 (or any float value depending on the quantile that the user wants to use for point predictions). This will disable models that do not support prediction intervals.

coverage: Union[float, List[float]], default = 0.9
    The coverage to be used for prediction intervals (only applicable for models that support prediction intervals).
    If a float value is provides, it corresponds to the coverage needed (e.g. 0.9 means 90% coverage). This corresponds to lower and upper quantiles = 0.05 and 0.95 respectively.
    Alternately, if user wants to get the intervals at specific quantiles, a list of 2 values can be provided directly. e.g. coverage = [0.2. 0.9] will return the lower interval corresponding to a quantile of 0.2 and an upper interval corresponding to a quantile of 0.9.

enforce_exogenous: bool, default = True
    When set to True and the data includes exogenous variables, only models that support exogenous variables are loaded in the environment.When set to False, all models are included and in this case, models that do not support exogenous variables will model the data as a univariate forecasting problem.

n_jobs: int, default = -1
    The number of jobs to run in parallel (for functions that supports parallel processing) -1 means using all processors. To run all functions on single processor set n_jobs to None.

use_gpu: bool or str, default = False
    Parameter not in use for now. Behavior may change in future.

custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
    Parameter not in use for now. Behavior may change in future.

html: bool, default = True
    When set to False, prevents runtime display of monitor. This must be set to False when the environment does not support IPython. For example, command line terminal, Databricks Notebook, Spyder and other similar IDEs.

session_id: int, default = None
    Controls the randomness of experiment. It is equivalent to 'random_state' in scikit-learn. When None, a pseudo random number is generated. This can be used for later reproducibility of the entire experiment.

system_log: bool or str or logging.Logger, default = True
    Whether to save the system logging file (as logs.log). If the input is a string, use that as the path to the logging file. If the input already is a logger object, use that one instead.

log_experiment: bool, default = False
    When set to True, all metrics and parameters are logged on the MLflow server.

experiment_name: str, default = None
    Name of the experiment for logging. Ignored when log_experiment is not True.

log_plots: bool or list, default = False
    When set to True, certain plots are logged automatically in the MLFlow server. To change the type of plots to be logged, pass a list containing plot IDs. Refer to documentation of plot_model. Ignored when log_experiment is not True.

log_profile: bool, default = False
    When set to True, data profile is logged on the MLflow server as a html file. Ignored when log_experiment is not True.

log_data: bool, default = False
    When set to True, dataset is logged on the MLflow server as a csv file. Ignored when log_experiment is not True.

engine: Optional[Dict[str, str]] = None
    The engine to use for the models, e.g. for auto_arima, users can switch between "pmdarima" and "statsforecast" by specifying
    engine={"auto_arima": "statsforecast"}

verbose: bool, default = True
    When set to False, Information grid is not printed.

profile: bool, default = False
    When set to True, an interactive EDA report is displayed.

profile_kwargs: dict, default = {} (empty dict)
    Dictionary of arguments passed to the ProfileReport method used to create the EDA report. Ignored if profile is False.

fig_kwargs: dict, default = {} (empty dict)
    The global setting for any plots. Pass these as key-value pairs.
    Example: fig_kwargs = {"height": 1000, "template": "simple_white"}

    Available keys are:

    hoverinfo: hoverinfo passed to Plotly figures. Can be any value supported
        by Plotly (e.g. "text" to display, "skip" or "none" to disable.). When not provided, hovering over certain plots may be disabled by PyCaret when the data exceeds a certain number of points (determined by big_data_threshold).

    renderer: The renderer used to display the plotly figure. Can be any value
        supported by Plotly (e.g. "notebook", "png", "svg", etc.). Note that certain renderers (like "svg") may need additional libraries to be installed. Users will have to do this manually since they don't come preinstalled with plotly. When not provided, plots use plotly's default render when data is below a certain number of points (determined by big_data_threshold) otherwise it switches to a static "png" renderer.

    template: The template to use for the plots. Can be any value supported by Plotly.
        If not provided, defaults to "ggplot2"

    width: The width of the plot in pixels. If not provided, defaults to None
        which lets Plotly decide the width.

    height: The height of the plot in pixels. If not provided, defaults to None
        which lets Plotly decide the height.

    rows: The number of rows to use for plots where this can be customized,
        e.g. ccf. If not provided, defaults to None which lets PyCaret decide based on number of subplots to be plotted.

    cols: The number of columns to use for plots where this can be customized,
        e.g. ccf. If not provided, defaults to 4

    big_data_threshold: The number of data points above which hovering over
        certain plots can be disabled and/or renderer switched to a static renderer. This is useful when the time series being modeled has a lot of data which can make notebooks slow to render. Also note that setting the display_format to a plotly-resampler figure ("plotly-dash" or "plotly-widget") can circumvent these problems by performing dynamic data aggregation.

    resampler_kwargs: The keyword arguments that are fed to configure the
        plotly-resampler visualizations (i.e., display_format "plotly-dash" or "plotly-widget") which down sampler will be used; how many data points are shown in the front-end. When the plotly-resampler figure is rendered via Dash (by setting the display_format to "plotly-dash"), one can also use the "show_dash" key within this dictionary to configure the show_dash method its args.

    example:

        fig_kwargs = {
            ...,
            "resampler_kwargs":  {
                "default_n_shown_samples": 1000,
                "show_dash": {"mode": "inline", "port": 9012}
            }
        }
		
'''



'''
def create_model(
    estimator: str | Any,
    fold: int | Any | None = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: dict | None = None,
    experiment_custom_tags: Dict[str, Any] | None = None,
    engine: str | None = None,
    verbose: bool = True,
    **kwargs: Any
) -> Any

estimator: str or sktime compatible object
    ID of an estimator available in model library or pass an untrained model object consistent with scikit-learn API. Estimators available in the model library (ID - Name):
    NOTE: The available estimators depend on multiple factors such as what libraries have been installed and the setup of the experiment. As such, some of these may not be available for your experiment. To see the list of available models, please run setup() first, then models().

    'naive' - Naive Forecaster
    'grand_means' - Grand Means Forecaster
    'snaive' - Seasonal Naive Forecaster (disabled when seasonal_period = 1)
    'polytrend' - Polynomial Trend Forecaster
    'arima' - ARIMA family of models (ARIMA, SARIMA, SARIMAX)
    'auto_arima' - Auto ARIMA
    'exp_smooth' - Exponential Smoothing
    'stlf' - STL Forecaster
    'croston' - Croston Forecaster
    'ets' - ETS
    'theta' - Theta Forecaster
    'tbats' - TBATS
    'bats' - BATS
    'prophet' - Prophet Forecaster
    'lr_cds_dt' - Linear w/ Cond. Deseasonalize & Detrending
    'en_cds_dt' - Elastic Net w/ Cond. Deseasonalize & Detrending
    'ridge_cds_dt' - Ridge w/ Cond. Deseasonalize & Detrending
    'lasso_cds_dt' - Lasso w/ Cond. Deseasonalize & Detrending
    'llar_cds_dt' - Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending
    'br_cds_dt' - Bayesian Ridge w/ Cond. Deseasonalize & Deseasonalize & Detrending
    'huber_cds_dt' - Huber w/ Cond. Deseasonalize & Detrending
    'omp_cds_dt' - Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending
    'knn_cds_dt' - K Neighbors w/ Cond. Deseasonalize & Detrending
    'dt_cds_dt' - Decision Tree w/ Cond. Deseasonalize & Detrending
    'rf_cds_dt' - Random Forest w/ Cond. Deseasonalize & Detrending
    'et_cds_dt' - Extra Trees w/ Cond. Deseasonalize & Detrending
    'gbr_cds_dt' - Gradient Boosting w/ Cond. Deseasonalize & Detrending
    'ada_cds_dt' - AdaBoost w/ Cond. Deseasonalize & Detrending
    'lightgbm_cds_dt' - Light Gradient Boosting w/ Cond. Deseasonalize & Detrending
    'catboost_cds_dt' - CatBoost w/ Cond. Deseasonalize & Detrending

fold: int or scikit-learn compatible CV generator, default = None
    Controls cross-validation. If None, the CV generator in the fold_strategy parameter of the setup function is used. When an integer is passed, it is interpreted as the 'n_splits' parameter of the CV generator in the setup function.

round: int, default = 4
    Number of decimal places the metrics in the score grid will be rounded to.

cross_validation: bool, default = True
    When set to False, metrics are evaluated on holdout set. fold param is ignored when cross_validation is set to False.

fit_kwargs: dict, default = {} (empty dict)
    Dictionary of arguments passed to the fit method of the model.

engine: Optional[str] = None
    The engine to use for the model, e.g. for auto_arima, users can switch between "pmdarima" and "statsforecast" by specifying engine="statsforecast".

verbose: bool, default = True
    Score grid is not printed when verbose is set to False.

**kwargs:
    Additional keyword arguments to pass to the estimator.

Returns:
    Trained Model
'''



'''
def plot_model(
    estimator: Any | None = None,
    plot: str | None = None,
    return_fig: bool = False,
    return_data: bool = False,
    verbose: bool = False,
    display_format: str | None = None,
    data_kwargs: Dict | None = None,
    fig_kwargs: Dict | None = None,
    save: str | bool = False
) -> (Tuple[str, list] | None)
This function analyzes the performance of a trained model on holdout set. When used without any estimator, this function generates plots on the original data set. When used with an estimator, it will generate plots on the model residuals.

Example
>>> from pycaret.datasets import get_data
>>> airline = get_data('airline')
>>> from pycaret.time_series import *
>>> exp_name = setup(data = airline,  fh = 12)
>>> plot_model(plot="diff", data_kwargs={"order_list": [1, 2], "acf": True, "pacf": True})
>>> plot_model(plot="diff", data_kwargs={"lags_list": [[1], [1, 12]], "acf": True, "pacf": True})
>>> arima = create_model('arima')
>>> plot_model(plot = 'ts')
>>> plot_model(plot = 'decomp', data_kwargs = {'type' : 'multiplicative'})
>>> plot_model(plot = 'decomp', data_kwargs = {'seasonal_period': 24})
>>> plot_model(estimator = arima, plot = 'forecast', data_kwargs = {'fh' : 24})
>>> tuned_arima = tune_model(arima)
>>> plot_model([arima, tuned_arima], data_kwargs={"labels": ["Baseline", "Tuned"]})
estimator: sktime compatible object, default = None
    Trained model object

plot: str, default = None
    Default is 'ts' when estimator is None, When estimator is not None, default is changed to 'forecast'. List of available plots (ID - Name):

'ts' - Time Series Plot
'train_test_split' - Train Test Split
'cv' - Cross Validation
'acf' - Auto Correlation (ACF)
'pacf' - Partial Auto Correlation (PACF)
'decomp' - Classical Decomposition
'decomp_stl' - STL Decomposition
'diagnostics' - Diagnostics Plot
'diff' - Difference Plot
'periodogram' - Frequency Components (Periodogram)
'fft' - Frequency Components (FFT)
'ccf' - Cross Correlation (CCF)
'forecast' - "Out-of-Sample" Forecast Plot
'insample' - "In-Sample" Forecast Plot
'residuals' - Residuals Plot
return_fig: bool, default = False
    When set to True, it returns the figure used for plotting. When set to False (the default), it will print the plot, but not return it.

return_data: bool, default = False
    When set to True, it returns the data for plotting. If both return_fig and return_data is set to True, order of return is figure then data.

verbose: bool, default = True
    Unused for now

display_format: str, default = None
    Display format of the plot. Must be one of [None, 'streamlit', 'plotly-dash', 'plotly-widget'], if None, it will render the plot as a plain plotly figure.

    The 'plotly-dash' and 'plotly-widget' formats will render the figure via
    plotly-resampler (https://github.com/predict-idlab/plotly-resampler) figures. These plots perform dynamic aggregation of the data based on the front-end graph view. This approach is especially useful when dealing with large data, as it will retain snappy, interactive performance.

'plotly-dash' uses a dash-app to realize this dynamic aggregation. The dash app requires a network port, and can be configured with various modes more information can be found at the show_dash documentation. (https://predict-idlab.github.io/plotly-resampler/figure_resampler.html#plotly_resampler.figure_resampler.FigureResampler.show_dash)
'plotly-widget' uses a plotly FigureWidget to realize this dynamic aggregation, and should work in IPython based environments (given that the external widgets are supported and the jupyterlab-plotly extension is installed).
    To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.

data_kwargs: dict, default = None
    Dictionary of arguments passed to the data for plotting.

    Available keys are:

    nlags: The number of lags to use when plotting correlation plots, e.g.
        ACF, PACF, CCF. If not provided, default internally calculated values are used.

    seasonal_period: The seasonal period to use for decomposition plots.
        If not provided, the default internally detected seasonal period is used.

    type: The type of seasonal decomposition to perform. Options are:
        ["additive", "multiplicative"]

    order_list: The differencing orders to use for difference plots. e.g.
        [1, 2] will plot first and second order differences (corresponding to d = 1 and 2 in ARIMA models).

    lags_list: An alternate and more explicit alternate to "order_list"
        allowing users to specify the exact lags to plot. e.g. [1, [1, 12]] will plot first difference and a second plot with first difference (d = 1 in ARIMA) and seasonal 12th difference (D=1, s=12 in ARIMA models). Also note that "order_list" = [2] can be alternately specified as lags_list = [[1, 1]] i.e. successive differencing twice.

    acf: True/False
        When specified in difference plots and set to True, this will plot the ACF of the differenced data as well.

    pacf: True/False
        When specified in difference plots and set to True, this will plot the PACF of the differenced data as well.

    periodogram: True/False
        When specified in difference plots and set to True, this will plot the Periodogram of the differenced data as well.

    fft: True/False
        When specified in difference plots and set to True, this will plot the FFT of the differenced data as well.

    labels: When estimator(s) are provided, the corresponding labels to
        use for the plots. If not provided, the model class is used to derive the labels.

    include: When data contains exogenous variables, then only specific
        exogenous variables can be plotted using this key. e.g. include = ["col1", "col2"]

    exclude: When data contains exogenous variables, specific exogenous
        variables can be excluded from the plots using this key. e.g. exclude = ["col1", "col2"]

    alpha: The quantile value to use for point prediction. If not provided,
        then the value specified during setup is used.

    coverage: The coverage value to use for prediction intervals. If not
        provided, then the value specified during setup is used.

    fh: The forecast horizon to use for forecasting. If not provided, then
        the one used during model training is used.

    X: When a model trained with exogenous variables has been finalized,
        user can provide the future values of the exogenous variables to make future target time series predictions using this key.

    plot_data_type: When plotting the data used for modeling, user may
        wish to see plots with the original data set provided, the imputed dataset (if imputation is set) or the transformed dataset (which includes any imputation and transformation set by the user). This keyword can be used to specify which data type to use.

        NOTE: (1) If no imputation is specified, then plotting the "imputed"
            data type will produce the same results as the "original" data type.
        (2) If no transformations are specified, then plotting the "transformed"
            data type will produce the same results as the "imputed" data type.

        Allowed values are (if not specified, defaults to the first one in the list):

        "ts": ["original", "imputed", "transformed"]
        "train_test_split": ["original", "imputed", "transformed"]
        "cv": ["original"]
        "acf": ["transformed", "imputed", "original"]
        "pacf": ["transformed", "imputed", "original"]
        "decomp": ["transformed", "imputed", "original"]
        "decomp_stl": ["transformed", "imputed", "original"]
        "diagnostics": ["transformed", "imputed", "original"]
        "diff": ["transformed", "imputed", "original"]
        "forecast": ["original", "imputed"]
        "insample": ["original", "imputed"]
        "residuals": ["original", "imputed"]
        "periodogram": ["transformed", "imputed", "original"]
        "fft": ["transformed", "imputed", "original"]
        "ccf": ["transformed", "imputed", "original"]

        Some plots (marked as True below) will also allow specifying multiple of data types at once.

        "ts": True
        "train_test_split": True
        "cv": False
        "acf": True
        "pacf": True
        "decomp": True
        "decomp_stl": True
        "diagnostics": True
        "diff": False
        "forecast": False
        "insample": False
        "residuals": False
        "periodogram": True
        "fft": True
        "ccf": False

fig_kwargs: dict, default = {} (empty dict)
    The setting to be used for the plot. Overrides any global setting passed during setup. Pass these as key-value pairs. For available keys, refer to the setup documentation.

    Time-series plots support more display_formats, as a result the fig-kwargs can also contain the resampler_kwargs key and its corresponding dict. These are additional keyword arguments that are fed to the display function. This is mainly used for configuring plotly-resampler visualizations (i.e., display_format "plotly-dash" or "plotly-widget") which down sampler will be used; how many data points are shown in the front-end.

    When the plotly-resampler figure is rendered via Dash (by setting the display_format to "plotly-dash"), one can also use the "show_dash" key within this dictionary to configure the show_dash args.

    example:

        fig_kwargs = {
            "width": None,
            "resampler_kwargs":  {
                "default_n_shown_samples": 1000,
                "show_dash": {"mode": "inline", "port": 9012}
            }
        }
save: string or bool, default = False
    When set to True, Plot is saved as a 'png' file in current working directory. When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.

Returns:
    Path to saved file and list containing figure and data, if any.
'''