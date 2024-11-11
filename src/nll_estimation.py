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


def _likelihood_from_quantiles(probs, quantiles, data_point) -> float:
    bin_idx = np.searchsorted(quantiles, data_point) - 1
    likelihood = None
    
    if bin_idx < 0:
        assert quantiles[0] > data_point
        likelihood = (probs[0] - 0) / (quantiles[0] - data_point)

    if bin_idx + 1 >= len(probs):
        assert quantiles[-1] < data_point
        likelihood = (1 - probs[-1]) / (data_point - quantiles[-1])

    if likelihood is None:
        likelihood = (probs[bin_idx + 1] - probs[bin_idx]) / (
            quantiles[bin_idx + 1] - quantiles[bin_idx]
        )

    if likelihood <= 0.0:
        print(probs.tolist())
        print(quantiles.tolist())
        print(data_point, bin_idx, len(probs))

    assert likelihood > 0.0, likelihood

    return likelihood


def nll_baseline(samples, data):
    probs = np.linspace(0, 1, 100, endpoint=True)[1:-1]
    quantiles = np.quantile(samples, probs)
    pdf_values = np.array(
        [_likelihood_from_quantiles(probs, quantiles, p) for p in data]
    )
    return -np.log(pdf_values).sum()


def nll_baseline2(samples, data):
    """
    This will compute the hist from the linear interpolation of the empirical distribution.
    nll_baseline does almost the same but more interpolation I think.

    This should be equivalent to hist_samples because

    hist_vals = np.diff(probs) / np.diff(quantiles)

    is computed in _likelihood_from_quantiles
    """
    probs = np.arange(1, len(samples) + 1) / (len(samples) + 1)
    quantiles = np.quantile(samples, probs, method="inverted_cdf")
    pdf_values = np.array(
        [_likelihood_from_quantiles(probs, quantiles, p) for p in data]
    )
    return -np.log(pdf_values).sum()


def nll_cdf_fit(samples, data):
    sorted_samples = np.sort(samples)
    cdf_empirical = np.arange(1, len(samples) + 1) / len(samples)
    cdf = UnivariateSpline(sorted_samples, cdf_empirical, s=0.01)
    pdf = cdf.derivative(1)
    pdf_values = _fix_pdf_estimates(pdf(data))
    return -np.log(pdf_values).sum()


def nll_cdf_fit_monotone(samples, data):
    sorted_samples = np.sort(samples)
    cdf_empirical = np.arange(1, len(samples) + 1) / len(samples)
    pre_cdf = UnivariateSpline(sorted_samples, cdf_empirical, s=0.01)
    x_eval = np.linspace(data.min(), data.max(), 20)
    cdf = PchipInterpolator(x_eval, pre_cdf(x_eval))
    pdf = cdf.derivative(1)
    pdf_values = _fix_pdf_estimates(pdf(data))
    return -np.log(pdf_values).sum()


def nll_kde_fit(samples, data):
    kde = stats.gaussian_kde(samples)
    pdf_values = kde.evaluate(data)
    pdf_values = _fix_pdf_estimates(pdf_values)
    return -np.log(pdf_values).sum()


def _old_nll_hist(samples, data, method=None, **hist_kwargs):
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

    return -np.log(pdf_values).sum()


def nll_hist(samples, data):
    dens, bins = np.histogram(samples, density=True, bins="auto")
    dens[dens <= 0.0] = np.mean(dens) * 1e-8
    
    eps = np.abs(data).max() * 1e-6

    def dens_fun(x):
        if data.min() < bins[0]:
            bins = [data.min() - eps] + bins.tolist()
            bins = np.array(bins)
        if data.max() > bins[-1]:
            bins = bins.tolist() + [data.max() + eps]
            bins = np.array(bins)
        

def nll_hist2(samples, data):
    dens, bins = np.histogram(samples, density=True, bins="auto")
    dens[dens <= 0.0] = np.mean(dens) * 1e-8
    assert np.all(dens > 0), dens

    bin_indices = np.digitize(data, bins) - 1
    inside_indices = (bin_indices >= 0) & (bin_indices < len(dens))
    inside_pdfs = dens[bin_indices[inside_indices]]
    
    eps = np.abs(data).max() * 1e-6

    def left_extension(x):
        val = np.interp(x, [data.min() - eps, bins[0]], [0, dens[0]])
        if np.any(val <= 0.0):
            print("left:", x, [data.min() - eps, bins[0]], [0, dens[0]])
        return val

    def right_extension(x):
        val = np.interp(x, [bins[-1], data.max() + eps], [dens[-1], 0])
        if np.any(val <= 0.0):
            print("right:", x, [bins[-1], data.max() + eps], [dens[-1], 0])
        return val

    left_data = data[bin_indices < 0]
    right_data = data[bin_indices >= len(dens)]
    outside_pdfs = []
    if data.min() <= samples.min():
        outside_pdfs.extend(left_extension(left_data))
    if data.max() >= samples.max():
        outside_pdfs.extend(right_extension(right_data))

    pdf_vals = np.array(inside_pdfs.tolist() + outside_pdfs)
    assert np.all(pdf_vals > 0), (inside_pdfs, outside_pdfs)

    return -np.log(pdf_vals).sum()


def weighted_percentile(samples, weights, percentile):
    """Calculates a weighted percentile."""
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]
    normalized_weights = sorted_weights / np.sum(sorted_weights)
    cumulative_weights = np.cumsum(normalized_weights)
    idx = np.searchsorted(cumulative_weights, percentile, side="left")
    return sorted_samples[idx]


def compute_PI(samples: np.ndarray, weights: np.ndarray, percentile: float):
    dens, bin_edges = np.histogram(samples, bins=30, density=True, weights=weights)
    i = 0
    area = 0.0
    while area < percentile:
        area += dens[i] * (bin_edges[i + 1] - bin_edges[i])
        i += 1
    return bin_edges[i]


def _compute_importance_weight(
    y_data: np.ndarray, y_pred: np.ndarray, sigma: float
) -> float:
    """Computes the importance weight for a given hyperparameter setting."""
    log_likelihood = 0
    for prediction, y_i in zip(y_pred, y_data):
        log_likelihood += -0.5 * ((y_i - prediction) / sigma) ** 2 - np.log(
            np.sqrt(2 * np.pi) * sigma
        )
    return np.exp(log_likelihood)


def compute_hist_weights(
    y_data: np.ndarray, y_prior_predictive_samples: np.ndarray, sigma: float
):
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
    assert y_prior_predictive_samples.shape == (
        num_samples,
        y_data.shape[0],
    ), f"{y_prior_predictive_samples.shape} != {(num_samples, y_data.shape[0])}"
    importance_weights = np.zeros(num_samples)

    for m in range(num_samples):
        importance_weights[m] = _compute_importance_weight(
            y_data, y_prior_predictive_samples[m], sigma
        )

    importance_weights /= np.sum(importance_weights)

    return importance_weights
