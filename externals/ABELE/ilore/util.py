import numpy as np
import pandas as pd
import scipy.stats as st

import warnings


def vector2dict(x, feature_names):
    return {k: v for k, v in zip(feature_names, x)}


def record2str(x, feature_names, numeric_columns):
    xd = vector2dict(x, feature_names)
    s = '{ '
    for att, val in xd.items():
        if att not in numeric_columns and val == 0.0:
            continue
        if att in numeric_columns:
            s += '%s = %s, ' % (att, val)
        else:
            att_split = att.split('=')
            s += '%s = %s, ' % (att_split[0], att_split[1])

    s = s[:-2] + ' }'
    return s


def multilabel2str(y, class_name):
    mstr = ', '.join([class_name[i] for i in range(len(y)) if y[i] == 1.0])
    return mstr


def multi_dt_predict(X, dt_list):
    nbr_labels = len(dt_list)
    Y = np.zeros((X.shape[0], nbr_labels))
    for l in range(nbr_labels):
        Y[:, l] = dt_list[l].predict(X)
    return Y


def calculate_feature_values(X, numeric_columns_index, categorical_use_prob=False, continuous_fun_estimation=False,
                             size=1000):

    feature_values = list()
    for i in range(X.shape[1]):
        values = X[:, i]
        unique_values = np.unique(values)
        if len(unique_values) == 1:
            new_values = np.array([unique_values[0]] * size)
        else:
            if i in numeric_columns_index:
                values = values.astype(np.float)
                if continuous_fun_estimation:
                    new_values = get_distr_values(values, size)
                else:  # suppose is gaussian
                    mu = float(np.mean(values))
                    sigma = float(np.std(values))
                    new_values = np.random.normal(mu, sigma, size)
                new_values = np.concatenate((values, new_values), axis=0)
            else:
                if categorical_use_prob:
                    diff_values, counts = np.unique(values, return_counts=True)
                    prob = 1.0 * counts / np.sum(counts)
                    new_values = np.random.choice(diff_values, size=size, p=prob)
                else:  # uniform distribution
                    diff_values = unique_values
                    new_values = diff_values

        feature_values.append(new_values)

    return feature_values


def get_distr_values(x, size=1000):
    nbr_bins = int(np.round(estimate_nbr_bins(x)))
    name, params = best_fit_distribution(x, nbr_bins)
    # print(name, params)
    dist = getattr(st, name)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    distr_values = np.linspace(start, end, size)

    return distr_values


# Distributions to check
DISTRIBUTIONS = [st.uniform, st.exponweib, st.expon, st.expon, st.gamma, st.beta, st.alpha,
                 st.chi, st.chi2, st.laplace, st.lognorm, st.norm, st.powerlaw] #st.dweibull,


def freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = max(2.0 * iqr / n**(1.0/3.0), 1)
    k = np.ceil((np.max(x) - np.min(x))/h)
    return k


def struges(x):
    n = len(x)
    k = np.ceil(np.log2(n)) + 1
    return k


def estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
                #print 'aaa'
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                # print distribution.name, sse
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params


# def amp2math(c):
#     if '&le;' in c:
#         idx = c.find('&le;')
#         cnew = '%s%s%s' % (c[:idx], '<=', c[idx + 4:])
#         return cnew
#     elif '&lt;' in c:
#         idx = c.find('&lt;')
#         cnew = '%s%s%s' % (c[:idx], '<', c[idx + 4:])
#         return cnew
#     elif '&gl;' in c:
#         idx = c.find('&gl;')
#         cnew = '%s%s%s' % (c[:idx], '>=', c[idx + 4:])
#         return cnew
#     elif '&gt;' in c:
#         idx = c.find('&gt;')
#         cnew = '%s%s%s' % (c[:idx], '>', c[idx + 4:])
#         return cnew
#     return c
#
#
# def math2amp(c):
#     if '<=' in c:
#         idx = c.find('<=')
#         cnew = '%s%s%s' % (c[:idx], '&le;', c[idx + 2:])
#         return cnew
#     elif '<' in c:
#         idx = c.find('<')
#         cnew = '%s%s%s' % (c[:idx], '&lt;', c[idx + 1:])
#         return cnew
#     elif '>=' in c:
#         idx = c.find('>=')
#         cnew = '%s%s%s' % (c[:idx], '&gl;', c[idx + 2:])
#         return cnew
#     elif '>' in c:
#         idx = c.find('>')
#         cnew = '%s%s%s' % (c[:idx], '&gt;', c[idx + 1:])
#         return cnew
#     return c


def sigmoid(x, x0=0.5, k=10.0, L=1.0):
    """
    A logistic function or logistic curve is a common "S" shape (sigmoid curve

    :param x: value to transform
    :param x0: the x-value of the sigmoid's midpoint
    :param k: the curve's maximum value
    :param L: the steepness of the curve
    :return: sigmoid of x
    """
    return L / (1.0 + np.exp(-k * (x - x0)))


def neuclidean(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))


def nmeandev(x, y):  # normalized mean deviation
    return np.mean(np.abs(x-y)/np.max([np.abs(x), np.abs(y)], axis=0))


def get_knee_point_value(values):
    y = values
    x = np.arange(0, len(y))

    index = 0
    max_d = -float('infinity')

    for i in range(0, len(x)):
        c = closest_point_on_segment(a=[x[0], y[0]], b=[x[-1], y[-1]], p=[x[i], y[i]])
        d = np.sqrt((c[0] - x[i]) ** 2 + (c[1] - y[i]) ** 2)
        if d > max_d:
            max_d = d
            index = i

    return index


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    cp_x = sx1 + u * x_delta
    cp_y = sy1 + u * y_delta
    closest_point = [cp_x, cp_y]

    return closest_point
