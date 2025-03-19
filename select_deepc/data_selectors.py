import warnings

import numpy as np
import torch
import torch.nn
import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.autograd import Variable

from .deepc_dataclasses import DeePCDims, TrajectoryData, TrajectoryDataSet
from .hankel_generation import HankelMatrixGenerator


class LkSelector:
    def __init__(
        self,
        order=1,
        deepc_dims: DeePCDims = None,
        forgetting_factor=1,
        custom_callback=None,
    ) -> None:
        """Simple L norm selector of order k. Can use a forgetting factor."""
        self._order = order
        self._custom_callback = custom_callback

        if deepc_dims is not None:
            self._forgetter_u = np.array(
                [
                    np.ones((deepc_dims.m)) * forgetting_factor**i
                    for i in range(deepc_dims.T_past + deepc_dims.T_fut)
                ]
            ).reshape(-1, 1)
            self._forgetter_y = np.array(
                [
                    np.ones((deepc_dims.p)) * forgetting_factor**i
                    for i in range(deepc_dims.T_past + deepc_dims.T_fut)
                ]
            ).reshape(-1, 1)

    def get_selector_name(self):
        return f"l{self._order}"

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        adjusted_H_u = H_u - input_traj
        adjusted_H_y = H_y - state_traj

        if self._custom_callback is not None:
            adjusted_H_u, adjusted_H_y = self._custom_callback(
                adjusted_H_u, adjusted_H_y
            )

        if hasattr(self, "_forgetter_u"):
            adjusted_H_u *= self._forgetter_u
            adjusted_H_y *= self._forgetter_y

        norms = np.linalg.norm(adjusted_H_u, ord=self._order, axis=0) + np.linalg.norm(
            adjusted_H_y, ord=self._order, axis=0
        )

        idcs = np.argsort(norms)
        return idcs, norms

    @staticmethod
    def position_equivariancer(H_u, H_y):
        """Sets positional elements of Hankel matrix to 0"""
        H_y[::6, :] = 0
        H_y[1::6, :] = 0
        return H_u, H_y


class CosineDistances:
    def __init__(
        self,
        deepc_dims: DeePCDims = None,
        forgetting_factor=1,
    ) -> None:
        if deepc_dims is not None:
            self._forgetter_u = np.array(
                [
                    np.ones((deepc_dims.m)) * forgetting_factor**i
                    for i in range(deepc_dims.T_past + deepc_dims.T_fut)
                ]
            ).reshape(-1, 1)
            self._forgetter_y = np.array(
                [
                    np.ones((deepc_dims.p)) * forgetting_factor**i
                    for i in range(deepc_dims.T_past + deepc_dims.T_fut)
                ]
            ).reshape(-1, 1)

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        current_traj = np.append(input_traj, state_traj).reshape(1, -1)
        hankel_stack = np.vstack((H_u, H_y))
        # NOTE(@naefjo): expects inputs as (n_samples, n_features)
        distances = cosine_distances(current_traj, hankel_stack.T).reshape(-1)
        idcs = np.argsort(distances)
        return idcs, distances

    def get_selector_name(self):
        return "cosine_distance"


class IsoMapEmbeddedDistances:
    def __init__(
        self,
        data: TrajectoryData,
        deepc_dims: DeePCDims,
        use_scaler: bool = False,
        n_neighbors=4,
        n_components=75,
        subsample_factor: float = 1.0,
        radius=None,
    ) -> None:
        self.deepc_dims = deepc_dims
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_u, h_y = HankelMatrixGenerator(
                deepc_dims.T_past, deepc_dims.T_fut
            ).generate_hankel_matrices(data)
        hankel_mat = np.vstack((h_u, h_y), dtype=np.float32)
        hankel_mat = resample(
            hankel_mat.T,
            replace=False,
            n_samples=int(hankel_mat.shape[-1] * subsample_factor),
            random_state=1,
        ).T
        print(f"data shape: {hankel_mat.shape}")
        self._n_neighbors = n_neighbors
        embedding = Isomap(
            n_neighbors=self._n_neighbors,
            radius=radius,
            n_components=n_components,
            p=1,
            n_jobs=-1,
        )
        # TODO: after fitting the data, need to hack n_neighbors in isomap
        self._embedding = make_pipeline(
            StandardScaler() if use_scaler else None,
            embedding,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._embedding.fit(hankel_mat.T)
        self._use_scaler = use_scaler

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        current_traj = np.append(input_traj, state_traj).reshape(1, -1)
        transformed_traj = self._embedding.transform(current_traj)
        # NOTE(@naefjo): embedding in (n_samples, n_components), so we want to reduce along axis 1
        norms = np.linalg.norm(
            self._embedding.steps[-1][-1].embedding_ - transformed_traj, ord=1, axis=1
        )

        idcs = np.argsort(norms)

        return idcs, norms

    def get_selector_name(self):
        return f"{'scaled_' if self._use_scaler else ''}isomap_embedder_nn{self._n_neighbors}"


class ClusteredIsoMapEmbeddedDistances:
    def __init__(
        self,
        data: TrajectoryData,
        deepc_dims: DeePCDims,
        use_scaler: bool = False,
        n_neighbors=4,
        n_components=75,
        subsample_factor: float = 1.0,
    ) -> None:
        self.deepc_dims = deepc_dims
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_u, h_y = HankelMatrixGenerator(
                deepc_dims.T_past, deepc_dims.T_fut
            ).generate_hankel_matrices(data, subsample_factor)
        hankel_mat = np.vstack((h_u, h_y))
        print(f"data shape: {hankel_mat.shape}")
        self._n_neighbors = n_neighbors
        embedding = Isomap(
            n_neighbors=self._n_neighbors,
            n_components=n_components,
            p=1,
            n_jobs=-1,
        )
        self.kmeans = KMeans(n_clusters=int(hankel_mat.shape[-1] / 10))

        self._embedding = make_pipeline(
            StandardScaler() if use_scaler else None,
            embedding,
        )
        self.kmeans.fit(hankel_mat.T)
        cluster_centers = self.kmeans.cluster_centers_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._embedding.fit(cluster_centers)
        self._use_scaler = use_scaler
        self._hankel_mat = hankel_mat

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        current_traj = np.append(input_traj, state_traj).reshape(1, -1)
        transformed_traj = self._embedding.transform(current_traj)
        # NOTE(@naefjo): embedding in (n_samples, n_components), so we want to reduce along axis 1
        cluster_norms = np.linalg.norm(
            self._embedding.steps[-1][-1].embedding_ - transformed_traj, ord=1, axis=1
        )

        idcs = np.argsort(cluster_norms)
        idcs_data = []
        for idx in idcs:
            idcs_data.extend((self.kmeans.labels_ == idx).nonzero()[0])

        return np.array(idcs_data), None

    def get_selector_name(self):
        return f"{'scaled_' if self._use_scaler else ''}isomap_embedder_nn{self._n_neighbors}"


class RandomSelector:
    def __init__(self) -> None:
        pass

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        return np.random.permutation(np.arange(0, H_y.shape[-1], dtype=int)), None

    def get_selector_name(self):
        return "random_selector"
