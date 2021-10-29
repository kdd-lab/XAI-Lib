import numpy as np
from scipy.spatial.distance import cdist
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from late.utils.utils import euclidean_norm
from sklearn.model_selection import train_test_split
import warnings
from late.blackboxes.blackbox_wrapper import DecodePredictWrapper


class NormalGenerator(object):
    def __init__(self, **kwargs):
        pass

    def generate_neighborhood(self, z, n=1000, **kwargs):
        Z = np.random.normal(size=(n, z.shape[1]))
        return Z



class NeighborhoodGenerator(object):
    def __init__(self, blackbox, decoder=None):
        self.blackbox = blackbox
        self.decoder = decoder
        if decoder is not None:
            self.blackbox_decoder = DecodePredictWrapper(decoder, blackbox)
        else:
            self.blackbox_decoder = blackbox
        self.kind = None
        self.closest_counterfactual = None
        self.best_threshold = None

    def generate_neighborhood(
            self,
            z,
            n_search=10000,
            n_batch=1000,
            lower_threshold=0,
            upper_threshold=4,
            kind="gaussian_matched",
            sampling_kind=None,
            vicinity_sampler_kwargs=dict(),
            stopping_ratio=0.01,
            check_upper_threshold=True,
            final_counterfactual_search=True,
            verbose=True,
            custom_sampling_threshold=None,
            custom_closest_counterfactual=None,
            n=500,
            balance=False,
            forced_balance_ratio=None,
            redo_search=True,
            cut_radius=False,
            **kwargs,
    ):
        """Search and generates a neighborhood around z
        Parameters
        ----------
        z: array of shape (1, n_features)
            instance to explain
        n_search: int, optional (default = 10000)
            total n. of instances generated at any step of the search algorithm
        n_batch: int, optional (default = 1000)
            batch n. of instances generated at any step of the search algorithm
        lower_threshold: int, optional (default = 0)
            threshold to refine the search, only used if downward_only=False
        upper_threshold: int, optional (default = 4)
            starting threshold
        kind: string, optional (default = "gaussian_matched")
            counterfactual search kind
        sampling_kind: string, optional (default = None)
            sampling_kind, if None the sampling kind is the same as kind
        vicinity_sampler_kwargs: dictionary, optional (default = dict())
        stopping_ratio: float, optional (default = 0.01)
            ratio at which to stop the counterfactual search algorithm i.e. stop if
            lower_threshold/upper_threshold < stopping_ratio. Only used if downward_only=True
        check_upper_threshold: bool, optional (default = True)
            check if with the starting upper threshold the search finds some counterexemplars
        final_counterfactual_search: bool, optional (default = True)
            after the search algorithm stops, search a last time for counterexemplars
        verbose: bool, optional (default = True)
        custom_sampling_threshold: float, optional (default = None)
            pass a threshold directly without searching it
        custom_closest_counterfactual: array of size (1, n_features), optional (default = None)
            pass a counterexemplar directly
        n: int, optional (default = 500)
            n. of instance of the neighborhood
        balance: bool, optional (default = False)
            balance the neighborhood labels after generating it (reduces n)
        forced_balance_ratio: float, optional (default = None)
            balance the neighborhood labels while generating it.
            A value of 0.5 means we want the same n. of instances per label
        redo_search: bool, optional (default = True)
            redo the search if even if it has been run before
        cut_radius:
            after the search of the counterexemplar and best_threshold, replace the threshold with the distance between
            the counterexemplar and z (useful only if the threshold of sampling_kind is a distance)
        kwargs

        Returns
        -------
        Z : array of size (n, n_features)
            generated neighborhood
        """
        self.kind = kind
        z_label = self.blackbox_decoder.predict(z)
        if (self.closest_counterfactual is None or redo_search) and (custom_closest_counterfactual is None):
            self.counterfactual_search(
                z=z,
                z_label=z_label,
                n_search=n_search,
                n_batch=n_batch,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                kind=kind,
                vicinity_sampler_kwargs=vicinity_sampler_kwargs,
                stopping_ratio=stopping_ratio,
                check_upper_threshold=check_upper_threshold,
                final_counterfactual_search=final_counterfactual_search,
                verbose=verbose,
                **kwargs,
            )

        kind = sampling_kind if sampling_kind is not None else kind

        Z = self.neighborhood_sampling(
            z=z,
            z_label=z_label,
            n_batch=n_batch,
            kind=kind,
            vicinity_sampler_kwargs=vicinity_sampler_kwargs,
            verbose=verbose,
            custom_sampling_threshold=custom_sampling_threshold,
            custom_closest_counterfactual=custom_closest_counterfactual,
            n=n,
            balance=balance,
            forced_balance_ratio=forced_balance_ratio,
            cut_radius=cut_radius,
            **kwargs,
        )
        return Z

    def counterfactual_search(
            self,
            z,
            z_label=None,
            n_search=10000,
            n_batch=1000,
            lower_threshold=0,
            upper_threshold=4,
            kind="gaussian_matched",
            vicinity_sampler_kwargs=dict(),
            stopping_ratio=0.01,
            check_upper_threshold=True,
            final_counterfactual_search=True,
            verbose=True,
            **kwargs,
    ):
        if z_label is None:
            z_label = self.blackbox_decoder.predict(z)
        self.closest_counterfactual, self.best_threshold = binary_sampling_search(
            z=z,
            z_label=z_label,
            blackbox=self.blackbox_decoder,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            n=n_search,
            n_batch=n_batch,
            kind=kind,
            vicinity_sampler_kwargs=vicinity_sampler_kwargs,
            stopping_ratio=stopping_ratio,
            check_upper_threshold=check_upper_threshold,
            final_counterfactual_search=final_counterfactual_search,
            verbose=verbose,
            **kwargs
        )
        return self.closest_counterfactual, self.best_threshold

    def neighborhood_sampling(
            self,
            z,
            z_label=None,
            n_batch=1000,
            kind="gaussian_matched",
            vicinity_sampler_kwargs=dict(),
            verbose=True,
            custom_sampling_threshold=None,
            custom_closest_counterfactual=None,
            n=500,
            balance=False,
            forced_balance_ratio=None,
            cut_radius=False,
            **kwargs,
    ):
        if z_label is None:
            z_label = self.blackbox_decoder.predict(z)
        if custom_closest_counterfactual is not None:
            self.closest_counterfactual = custom_closest_counterfactual
        if cut_radius:
            self.best_threshold = np.linalg.norm(z - self.closest_counterfactual)
            if verbose:
                print("Setting new threshold at radius:", self.best_threshold)
            if kind not in ["uniform_sphere"]:
                warnings.warn("cut_radius=True, but for the method " + kind + " the threshold is not a radius.")
        if custom_sampling_threshold is not None:
            self.best_threshold = custom_sampling_threshold
            if verbose:
                print("Setting custom threshold:", self.best_threshold)

        Z = vicinity_sampling(
            z=self.closest_counterfactual,
            n=n,
            threshold=self.best_threshold,
            kind=kind,
            verbose=verbose,
            **vicinity_sampler_kwargs
        )

        if forced_balance_ratio is not None:
            y = self.blackbox_decoder.predict(Z)
            y = 1 * (y == z_label)
            n_minority_instances = np.unique(y, return_counts=True)[1].min()
            if (n_minority_instances / n) < forced_balance_ratio:
                if verbose:
                    print("Forced balancing neighborhood...", end=" ")
                n_desired_minority_instances = int(forced_balance_ratio * n)
                n_desired_majority_instances = n - n_desired_minority_instances
                minority_class = np.argmin(np.unique(y, return_counts=True)[1])
                sampling_strategy = n_desired_minority_instances / n_desired_majority_instances
                while n_minority_instances < n_desired_minority_instances:
                    Z_ = vicinity_sampling(
                        z=self.closest_counterfactual,
                        n=n_batch,
                        threshold=self.best_threshold if custom_sampling_threshold is None else custom_sampling_threshold,
                        kind=kind,
                        verbose=False,
                        **vicinity_sampler_kwargs
                    )
                    y_ = self.blackbox_decoder.predict(Z_)
                    y_ = 1 * (y_ == z_label)
                    n_minority_instances += np.unique(y_, return_counts=True)[1][minority_class]
                    Z = np.concatenate([Z, Z_])
                    y = np.concatenate([y, y_])
                rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
                Z, y = rus.fit_resample(Z, y)
                if len(Z) > n:
                    Z, _ = train_test_split(Z, train_size=n, stratify=y)
                if verbose:
                    print("Done!")

        if balance:
            if verbose:
                print("Balancing neighborhood...", end=" ")
            rus = RandomUnderSampler(random_state=0)
            y = self.blackbox_decoder.predict(Z)
            y = 1 * (y == self.blackbox_decoder.predict(z))
            Z, _ = rus.fit_resample(Z, y)
            if verbose:
                print("Done!")
        return Z


def check_neighborhood_norm(size, kind, n=1000, n_single_neighborhood=1, threshold=1, distribution=np.random.normal,
                            **kwargs):
    Zs = list()
    for i in range(n):
        z = np.random.normal(size=size)
        Z = vicinity_sampling(z=z, n=n_single_neighborhood, threshold=threshold, kind=kind, distribution=distribution,
                              verbose=False, **kwargs)
        Zs.append(Z)
    Zs = np.concatenate(Zs)
    return Zs, euclidean_norm(Zs)


def filter_neighborhood(Z, y, ratio=0.5, ignore_instance_after_first_match=False, inverse=False):
    labels = np.unique(y)
    if len(labels) > 2:
        raise Exception("Dataset labels must be binarized.")
    idxs_a = np.argwhere(y == labels[0]).ravel()
    Z_a = Z[idxs_a]
    idxs_b = np.argwhere(y == labels[1]).ravel()
    Z_b = Z[idxs_b]
    distance_matrix = cdist(Z_a, Z_b)
    distance_dict = {"a": list(), "b": list(), "dist": list()}
    for row in range(distance_matrix.shape[0]):
        for column in range(distance_matrix.shape[1]):
            distance_dict["a"].append(idxs_a[row])
            distance_dict["b"].append(idxs_b[column])
            distance_dict["dist"].append(distance_matrix[row, column])
    df = pd.DataFrame(distance_dict)
    df_sorted = df.sort_values(["dist"], axis=0)
    idxs_to_filter = set()
    df_idx = 0
    while len(idxs_to_filter) / len(Z) < ratio:
        if ignore_instance_after_first_match:
            if df_sorted.iloc[df_idx]["a"] in idxs_to_filter or df_sorted.iloc[df_idx]["b"] in idxs_to_filter:
                df_idx += 1
                continue
        idxs_to_filter.add(df_sorted.iloc[df_idx]["a"])
        idxs_to_filter.add(df_sorted.iloc[df_idx]["b"])
        df_idx += 1
    idxs = set(range(Z.shape[0]))
    if inverse:
        idxs_to_keep = idxs_to_filter
    else:
        idxs_to_keep = idxs.difference(idxs_to_filter)
    idxs_to_keep = np.array(list(idxs_to_keep), dtype=np.int)
    Z_filtered = Z[idxs_to_keep]
    return Z_filtered


def test_generators():
    z = np.array([[1, 1]])
    for kind in ["gaussian_matched", "gaussian", "gaussian_global", "uniform_sphere", "by_rejection"]:
        Z = vicinity_sampling(
            z,
            kind=kind,
            epsilon=1,
            r=1,
            distribution=np.random.normal,
            n=500)
        plt.scatter(Z[:, 0], Z[:, 1])
        plt.scatter(z[:, 0], z[:, 1], c="red")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def vicinity_sampling(
        z,
        n=1000,
        threshold=None,
        kind="gaussian_matched",
        distribution=None,
        distribution_kwargs=dict(),
        verbose=True,
        **kwargs
):
    if verbose:
        print("\nSampling -->", kind)
    if kind == "gaussian":
        Z = gaussian_vicinity_sampling(z, threshold, n)
    elif kind == "gaussian_matched":
        Z = gaussian_matched_vicinity_sampling(z, threshold, n)
    elif kind == "gaussian_global":
        Z = gaussian_global_sampling(z, n)
    elif kind == "uniform_sphere":
        Z = uniform_sphere_vicinity_sampling(z, n, threshold)
    elif kind == "uniform_sphere_scaled":
        Z = uniform_sphere_scaled_vicinity_sampling(z, n, threshold)
    elif kind == "by_rejection":
        Z = sample_by_rejection(
            distribution=distribution,
            center=z,
            r=threshold,
            distribution_kwargs=distribution_kwargs,
            n=n,
            verbose=verbose
        )
    else:
        raise Exception("Vicinity sampling kind not valid")
    return Z


def gaussian_matched_vicinity_sampling(z, epsilon, n=1):
    return gaussian_vicinity_sampling(z, epsilon, n) / np.sqrt(1 + (epsilon ** 2))


"""
def gaussian_matched_vicinity_sampling(z, epsilon):
    return (z + (np.random.normal(size=z.shape) * epsilon)) / np.sqrt(1 + (epsilon ** 2))
"""


def gaussian_vicinity_sampling(z, epsilon, n=1):
    return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)


def gaussian_global_sampling(z, n=1):
    return np.random.normal(size=(n, z.shape[1]))


def sample_by_rejection(distribution, center, r, distribution_kwargs=dict(), n=1000, verbose=True):
    Z = []
    count = 0
    while len(Z) < n:
        Z_sample = distribution(size=(n, center.shape[1]), **distribution_kwargs)
        distances = cdist(center, Z_sample).ravel()
        Z.extend(Z_sample[np.nonzero((distances <= r).ravel())])
        count += 1
        if verbose:
            print("   iteration", str(count) + ":", "found", len(Z), "samples", end='\r')
    if verbose:
        print()
    Z = np.array(Z)
    Z = Z[np.random.choice(Z.shape[0], n, replace=False), :]
    return Z


def uniform_sphere_origin(n, d, r=1):
    """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
    by "radius" (length of points are in range [0, "radius"]).

    Parameters
    ----------
    n : int
        number of points to generate
    d : int
        dimensionality of each point
    r : float
        radius of the sphere

    Returns
    -------
    array of shape (n, d)
        sampled points
    """
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(d, n))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(n) ** (1 / d)
    # Return the list of random (direction & length) points.
    return r * (random_directions * random_radii).T


def uniform_sphere_vicinity_sampling(z, n=1, r=1):
    Z = uniform_sphere_origin(n, z.shape[1], r)
    translate(Z, z)
    return Z


def uniform_sphere_scaled_vicinity_sampling(z, n=1, threshold=1):
    Z = uniform_sphere_origin(n, z.shape[1], r=1)
    Z *= threshold
    translate(Z, z)
    return Z


def translate(X, center):
    """Translates a origin centered array to a new center

    Parameters
    ----------
    X : array
        data to translate centered in the axis origin
    center : array
        new center point

    Returns
    -------
    None
    """
    for axis in range(center.shape[-1]):
        X[..., axis] += center[..., axis]


def spherical_interpolation(a, b, t):
    if t <= 0:
        return a
    elif t >= 1:
        return b
    elif np.allclose(a, b):
        return a
    omega = np.arccos(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * a + np.sin(t * omega) / so * b


def linear_interpolation(a, b, t):
    return (t * a) + ((1 - t) * b)


def gaussian_matched_interpolation(a, b, t):
    return linear_interpolation(a, b, t) / np.sqrt((t ** 2) + (1 - t) ** 2)


def interpolate(a, b, kind="linear", n=10):
    a = a.ravel()
    b = b.ravel()
    interpolation_matrix = list()
    for t in np.arange(1 / n, 1, 1 / n):
        if kind == "linear":
            interpolation_vector = linear_interpolation(a, b, t)
        elif kind == "gaussian_matched":
            interpolation_vector = gaussian_matched_interpolation(a, b, t)
        elif kind == "slerp":
            interpolation_vector = spherical_interpolation(a, b, t)
        else:
            raise ValueError("Invalid interpolation kind")
        interpolation_matrix.append(interpolation_vector)
    return np.array(interpolation_matrix)


def find_endpoint(a, midpoint, kind="linear"):
    a = a.ravel()
    midpoint = midpoint.ravel()
    if kind == "linear":
        b = (2 * midpoint) - a
    elif kind == "gaussian_matched":
        b = (midpoint * (2 ** (1 / 2))) - a
    return b


def binary_sampling_search(
        z,
        z_label,
        blackbox,
        lower_threshold=0,
        upper_threshold=4,
        n=10000,
        n_batch=1000,
        stopping_ratio=0.01,
        kind="gaussian_matched",
        vicinity_sampler_kwargs=dict(),
        verbose=True,
        check_upper_threshold=True,
        final_counterfactual_search=True,
        downward_only=True,
        **kwargs
):
    if verbose:
        print("Binary sampling search:", kind)

    # sanity check for the upper threshold
    if check_upper_threshold:
        for i in range(int(n / n_batch)):
            Z = vicinity_sampling(
                z=z,
                n=n_batch,
                threshold=upper_threshold,
                kind=kind,
                verbose=False,
                **vicinity_sampler_kwargs
            )
            y = blackbox.predict(Z)
            if not np.all(y == z_label):
                break
        if i == list(range(int(n / n_batch)))[-1]:
            raise Exception("No counterfactual found, increase upper threshold or n_search.")

    change_lower = False
    latest_working_threshold = upper_threshold
    Z_counterfactuals = list()
    while lower_threshold / upper_threshold < stopping_ratio:
        if change_lower:
            if downward_only:
                break
            lower_threshold = threshold
        threshold = (lower_threshold + upper_threshold) / 2
        change_lower = True
        if verbose:
            print("   Testing threshold value:", threshold)
        for i in range(int(n / n_batch)):
            Z = vicinity_sampling(
                z=z,
                n=n_batch,
                threshold=threshold,
                kind=kind,
                verbose=False,
                **vicinity_sampler_kwargs
            )
            y = blackbox.predict(Z)
            if not np.all(y == z_label):  # if we found already some counterfactuals
                counterfactuals_idxs = np.argwhere(y != z_label).ravel()
                Z_counterfactuals.append(Z[counterfactuals_idxs])
                latest_working_threshold = threshold
                upper_threshold = threshold
                change_lower = False
                break
    if verbose:
        print("   Best threshold found:", latest_working_threshold)
    if final_counterfactual_search:
        if verbose:
            print("   Final counterfactual search... (this could take a while)", end=" ")
        Z = vicinity_sampling(
            z=z,
            n=n,
            threshold=latest_working_threshold,
            kind=kind,
            verbose=False,
            **vicinity_sampler_kwargs
        )
        y = blackbox.predict(Z)
        counterfactuals_idxs = np.argwhere(y != z_label).ravel()
        Z_counterfactuals.append(Z[counterfactuals_idxs])
        if verbose:
            print("Done!")
    Z_counterfactuals = np.concatenate(Z_counterfactuals)
    closest_counterfactual = min(Z_counterfactuals, key=lambda p: sum((p - z.ravel()) ** 2))
    return closest_counterfactual.reshape(1, -1), latest_working_threshold


if __name__ == "__main__":
    from late.blackboxes.blackbox_wrapper import BlackboxWrapper
    from late.datasets.datasets import build_cbf
    from joblib import load
    import matplotlib.pyplot as plt
    from late.autoencoders.variational_autoencoder import load_model
    from late.explainers.lasts import plot_latent_space
    from late.utils.utils import choose_z

    random_state = 0
    dataset_name = "cbf"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600,
                                                               random_state=random_state)

    knn = load("./trained_models/cbf/cbf_knn.joblib")

    _, _, autoencoder = load_model("./trained_models/cbf/cbf_vae")

    blackbox = BlackboxWrapper(knn, 2, 1)
    encoder = autoencoder.layers[2]
    decoder = autoencoder.layers[3]

    x = X_exp_test[2].ravel().reshape(1, -1, 1)
    z = choose_z(x, encoder, decoder)
    z_label = blackbox.predict(decoder.predict(z))[0]
    K = encoder.predict(X_exp_train)

    neigh = NeighborhoodGenerator(blackbox, decoder)
    neigh_kwargs = {
        "balance": False,
        "n": 500,
        "n_search": 10000,
        "threshold": 2,
        "kind": "uniform_sphere",
        "verbose": True,
        "stopping_ratio": 0.01,
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True
    }

    Z = neigh.generate_neighborhood(z=z, **neigh_kwargs)

    plot_latent_space(Z, blackbox.predict(decoder.predict(Z)), z.ravel(), z_label, K=K,
                      closest_counterfactual=neigh.closest_counterfactual)

    plot_latent_space(Z, blackbox.predict(decoder.predict(Z)), z.ravel(), z_label,
                      closest_counterfactual=neigh.closest_counterfactual)

    from late.explainers.lasts import plot_exemplars_and_counterexemplars

    plot_exemplars_and_counterexemplars(decoder.predict(Z),
                                        blackbox.predict(decoder.predict(Z)),
                                        x.ravel(),
                                        decoder.predict(z.ravel().reshape(1, -1)),
                                        z_label)
