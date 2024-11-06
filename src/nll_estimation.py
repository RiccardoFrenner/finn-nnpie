import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator, UnivariateSpline

PDF_FIX_METHOD = "omit"  # {omit, clip}


def _fix_pdf_estimates(pdf_estimates: np.ndarray) -> np.ndarray:
    if PDF_FIX_METHOD == "omit":
        return pdf_estimates[pdf_estimates > 0.0]
    if PDF_FIX_METHOD == "clip":
        return np.clip(pdf_estimates, 1e-10, None)

    raise NotImplementedError()


def _hist_bin_selection(samples, data, method: str):
    eps = np.abs(data).max() * 1e-6

    if method == "extra_bin":
        bin_edges = np.histogram_bin_edges(samples, bins="auto")

        # add an extra bin such that data is always in bins
        if data.min() < bin_edges[0]:
            bin_edges = np.sort([data.min() - eps] + bin_edges.tolist())
        if data.max() >= bin_edges[-1]:
            bin_edges = np.sort(bin_edges.tolist() + [data.max() + eps])

    elif method == "extended_ends":
        bin_edges = np.histogram_bin_edges(samples, bins="auto")
        if data.min() < bin_edges[0]:
            bin_edges[0] = data.min() - eps
        if data.max() >= bin_edges[-1]:
            bin_edges[-1] = data.max() + eps

    elif method == "data_scaled":
        data_scaled = data.copy()
        data_scaled[np.argmin(data)] -= eps
        data_scaled[np.argmax(data)] += eps
        bin_edges = np.histogram_bin_edges(data_scaled, bins="auto")

    elif method == "samples":
        bin_edges = np.sort(samples)
        # add an extra bin such that data is always in bins
        if data.min() < bin_edges[0]:
            bin_edges = np.sort([data.min() - eps] + bin_edges.tolist())
        if data.max() >= bin_edges[-1]:
            bin_edges = np.sort(bin_edges.tolist() + [data.max() + eps])

    else:
        raise ValueError(f"Unknown method: '{method}'")

    return bin_edges


def _nll_from_quantiles(probs, quantiles, data_point):
    bin_idx = np.searchsorted(quantiles, data_point) - 1
    if bin_idx < 0 or bin_idx + 1 >= len(probs):
        return 0.0
    return (probs[bin_idx + 1] - probs[bin_idx]) / (
        quantiles[bin_idx + 1] - quantiles[bin_idx]
    )


def nll_baseline(samples, data):
    qs = np.linspace(0, 1, 100)
    quantiles = np.quantile(samples, qs)
    pdf_values = np.array([_nll_from_quantiles(qs, quantiles, p) for p in data])
    pdf_values = _fix_pdf_estimates(pdf_values)
    return -np.log(pdf_values).mean()


def nll_baseline2(samples, data):
    """
    This will compute the hist from the linear interpolation of the empirical distribution.
    nll_baseline does almost the same but more interpolation I think.

    This should be equivalent to hist_samples because

    hist_vals = np.diff(probs) / np.diff(quantiles)

    is computed in _nll_from_quantiles
    """
    probs = np.arange(1, len(samples) + 1) / len(samples)
    quantiles = np.quantile(samples, probs, method="inverted_cdf")
    pdf_values = np.array([_nll_from_quantiles(probs, quantiles, p) for p in data])
    pdf_values = _fix_pdf_estimates(pdf_values)
    return -np.log(pdf_values).mean()


def nll_cdf_fit(samples, data):
    sorted_samples = np.sort(samples)
    cdf_empirical = np.arange(1, len(samples) + 1) / len(samples)
    cdf = UnivariateSpline(sorted_samples, cdf_empirical, s=0.01)
    pdf = cdf.derivative(1)
    pdf_values = _fix_pdf_estimates(pdf(data))
    return -np.log(pdf_values).mean()


def nll_cdf_fit_monotone(samples, data):
    sorted_samples = np.sort(samples)
    cdf_empirical = np.arange(1, len(samples) + 1) / len(samples)
    pre_cdf = UnivariateSpline(sorted_samples, cdf_empirical, s=0.01)
    x_eval = np.linspace(data.min(), data.max(), 20)
    cdf = PchipInterpolator(x_eval, pre_cdf(x_eval))
    pdf = cdf.derivative(1)
    pdf_values = _fix_pdf_estimates(pdf(data))
    return -np.log(pdf_values).mean()


def nll_kde_fit(samples, data):
    kde = stats.gaussian_kde(samples)
    pdf_values = kde.evaluate(data)
    pdf_values = _fix_pdf_estimates(pdf_values)
    return -np.log(pdf_values).mean()


def nll_hist(samples, data, method=None, **hist_kwargs):
    # best default (empirically tested)
    if method is None:
        method = "extra_bin" if len(data) > len(samples) else "data_scaled"

    bin_edges = _hist_bin_selection(samples, data, method)

    dens, bin_edges = np.histogram(samples, bin_edges, density=True)

    assert data.min() > bin_edges[0], (data.min(), bin_edges[0])
    assert data.max() < bin_edges[-1], (data.max(), bin_edges[-1])

    data_bin_indices = np.digitize(data, bin_edges) - 1
    pdf_values = dens[data_bin_indices]
    pdf_values = _fix_pdf_estimates(pdf_values)

    return -np.log(pdf_values).mean()

def weighted_percentile(samples, weights, percentile):
    """Calculates a weighted percentile."""
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]
    normalized_weights = sorted_weights / np.sum(sorted_weights)
    cumulative_weights = np.cumsum(normalized_weights)
    idx = np.searchsorted(cumulative_weights, percentile, side='left')
    return sorted_samples[idx]

def compute_PI(samples: np.ndarray, weights: np.ndarray, percentile: float):
    dens, bin_edges = np.histogram(samples, bins=30, density=True, weights=weights)
    # mean = np.mean(samples * weights)
    # print("Unweighted mean:", np.mean(samples))
    # print("  Weighted mean:", mean)
    # assert mean < bin_edges[-1]
    # assert mean > bin_edges[0]
    i = 0
    area = 0.0
    while area < percentile:
        area += dens[i]*(bin_edges[i+1] - bin_edges[i])
        i += 1
    return bin_edges[i]


def _compute_loss_pattern_weights(N_data: int, N_b: int, alpha: float) -> float:
    """
    Assumption: Training on $D_b$ which is obtained by removing a (x,y) with prob. $\alpha$
    N_b = size(D_B)
    \omega_b = p(D_b | D) = (1-\alpha)^{N_b} \alpha^{N - N_b}
    """
    return (1-alpha)**N_b * alpha**(N_data - N_b)


def _compute_importance_weight(y_data: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    """Computes the importance weight for a given hyperparameter setting."""
    log_likelihood = 0
    for prediction, y_i in zip(y_pred, y_data):
        log_likelihood += -0.5 * ((y_i - prediction) / sigma)**2 - np.log(np.sqrt(2 * np.pi) * sigma)
    return np.exp(log_likelihood)


def compute_hist_weights(y_data: np.ndarray, y_prior_predictive_samples: np.ndarray, sigma: float):
    """
    Approximates the posterior predictive distribution p(y | x, D) using importance sampling.

    Args:
        y_data: Training data: ys of (x_i, y_i), i=1..N
        y_prior_predictive_samples: [[model(x_1, h_1), ..., model(x_N, h_1)], ... [model(x_1, h_M), ..., model(x_N, h_M)]]
        solver: Function to compute weights given hyperparameters and data: w = solver(h, D).
        model: The neural network model: y = model(x, w).
        sigma: Standard deviation of the Gaussian noise.

    Returns:
        bin_edges: Edges of the histogram bins.
        posterior_probabilities: Normalized probabilities for each bin.
    """

    num_samples = y_prior_predictive_samples.shape[0]
    assert y_prior_predictive_samples.shape == (num_samples, y_data.shape[0]), f"{y_prior_predictive_samples.shape} != {(num_samples, y_data.shape[0])}"
    importance_weights = np.zeros(num_samples)

    for m in range(num_samples):
        importance_weights[m] = _compute_importance_weight(y_data, y_prior_predictive_samples[m], sigma)

    importance_weights /= np.sum(importance_weights)

    return importance_weights