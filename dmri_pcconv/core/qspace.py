'''Q-Space Sampler'''
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


def get_shell_filter(bval: np.ndarray, shell: float, shell_var: float) -> np.ndarray:
    '''Applies shell filtering with given shell variance'''
    return (bval >= shell - shell_var) & (bval <= shell + shell_var)


def shell_pair_list(ints: List[int], multishell_only: bool = False):
    '''Create pairs of ints'''
    # Create an empty list to store the pairs
    pairs = set()
    # Iterate over the elements in the list
    for i in ints:
        # For each element, iterate over the remaining elements in the list
        for j in ints:
            if multishell_only and i == j:
                continue
            pairs.add((i, j))
    # Return the list of pairs
    return sorted(list(pairs))


def spherical_distances(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    '''Calculates pairwise spherical distance between x and y

    Args:
        x: array one -> shape (3, m)
        y: array two -> shape (3, n)
            Optional, if omitted will calculate distance
            matrix of `x`

    Returns:
        sph_dist: spherical distance matrix
            shape -> (m, n)
    '''
    x = x / np.linalg.norm(x, axis=0)
    if y is not None:
        y = y / np.linalg.norm(y, axis=0)
    else:
        y = x

    # Dot product
    dots = np.matmul(x.T, y)  # type: ignore

    # Ensure no values above 1 or below -1
    dots = np.clip(dots, -1.0, 1.0)

    # Calculate distance
    sph_dist = np.arccos(dots)

    return sph_dist


class QSpaceInfo:
    '''QSpace metadata object for distribution of Q-space sampling'''

    def __init__(
        self, q_in_max: int, q_in_min: int, q_out: int, shells: List[int], seed=None
    ) -> None:
        self.q_in_min = q_in_min
        self.q_in_max = q_in_max
        self.q_out = q_out
        self.candidate_shells = shells
        self.rng = np.random.default_rng(seed)
        self._range = np.arange(self.q_in_min, self.q_in_max + 1, dtype=int)
        self._deterministic_dist = 3
        self._set_deterministic_samples()
        self._q_in = 0
        self._shell_in = 0
        self._shell_out = 0

    def _set_deterministic_samples(self):
        q_ins = range(self.q_in_min, self.q_in_max + 1, self._deterministic_dist)
        shell_pairs = shell_pair_list(list(self.candidate_shells))
        self._deterministic_samples = []
        for shell_pair in shell_pairs:
            for q_in in q_ins:
                self._deterministic_samples.append((q_in, shell_pair[0], shell_pair[1]))

    @property
    def q_in(self):
        '''Current q_in size'''
        return self._q_in

    @property
    def qtotal(self):
        '''Current qtotal size'''
        return self.q_in + self.q_out

    @property
    def shells_in(self):
        '''Current shells_in'''
        return [self._shell_in]

    @property
    def shells_out(self):
        '''Current shells_out'''
        return [self._shell_out]

    @property
    def shells(self):
        '''Current shells'''
        return sorted(set(self.shells_in + self.shells_out))

    @property
    def in_total(self):
        '''Current in_total'''
        return self.q_in * len(self.shells_in)

    @property
    def out_total(self):
        '''Current out_total'''
        return self.q_out * len(self.shells_out)

    @property
    def total(self):
        '''Current total'''
        return self.in_total + self.out_total

    def deterministic_sample(self, idx):
        '''Non-random sampling'''
        true_index = idx % len(self._deterministic_samples)
        sample = self._deterministic_samples[true_index]
        self._q_in, self._shell_in, self._shell_out = sample

    def random_sample(self):
        '''Resets q_in and shell sample'''
        self._q_in = self.rng.choice(self._range)
        self._shell_in = self.rng.choice(self.candidate_shells)
        self._shell_out = self.rng.choice(self.candidate_shells)


class QSpaceSampler:
    '''QSpace Sampler'''

    def __init__(self, qinfo: QSpaceInfo, random=True, seed=None) -> None:
        self.qinfo = qinfo
        self.random = random
        self.rng = np.random.default_rng(seed)

    def choice_func(self, var) -> int:
        '''Random choice function, selects first entry if not random'''
        if self.random:
            return self.rng.choice(var)
        return 0

    def _get_bvec_sample_order(
        self,
        bvec: np.ndarray,
        sph_dists: np.ndarray,
        subset_size: int,
        prev_order: Optional[np.ndarray] = None,
        random_seed: Union[bool, int] = True,
    ) -> np.ndarray:
        '''Gets optimal order of subset given a subset_size and
            optionally an array of indices to be exluded from the
            search.

        Args:
            bvec: bvec array -> shape (3,b)
            sph_dists: spherical distances -> (b, b)
            subset_size: Size of subset
            prev_order (np.ndarray): Previous indices in set b that are excluded from search
                shape -> (p,).
                Default: `None` -> empty argmask

        Returns:
            order: order index array. shape -> (subset_size,)
        '''
        # Initialise array
        order = np.zeros(subset_size, dtype=int)

        # Mask previously selected co-ordinates
        avail_mask = np.ones(len(bvec.T), bool)
        if prev_order is not None:
            avail_mask[prev_order] = False

        # Get index mapping and apply mask
        idx_map = np.arange(len(bvec.T))[avail_mask]
        bvec = bvec[:, avail_mask]

        # Remove already used b-vectors from distances
        sph_dists = sph_dists[avail_mask, :]
        sph_dists = sph_dists[:, avail_mask]

        # Initial point
        if isinstance(random_seed, bool):
            if random_seed:
                start = self.rng.choice(len(bvec.T))
            else:
                start = 0
        elif isinstance(random_seed, (int, np.integer)):
            start = random_seed
        else:
            raise ValueError('random_seed must be either int or bool.')

        # Initialise pick mask
        pick_mask = np.zeros(len(idx_map), dtype=bool)

        order[0] = idx_map[start]
        pick_mask[start] = True

        # Compute other points
        for idx in range(1, subset_size):
            min_dists = np.amin(sph_dists[:, pick_mask], axis=1)
            new_dx = np.argmax(min_dists)

            # Get order and update pick mask
            order[idx] = idx_map[new_dx]
            pick_mask[new_dx] = True

        if prev_order is not None:
            order = np.concatenate([prev_order, order])

        return order

    def _get_subject_sample(
        self,
        bvec_shells: Dict[int, np.ndarray],
        shell_index: Dict[int, np.ndarray],
        sph_dists_shells: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        orders = {}
        q_in_order, q_out_order = [], []
        for shell in self.qinfo.shells:
            bvec = bvec_shells[shell]
            sph_dist = sph_dists_shells[shell]
            shell_rem = bvec.shape[-1]
            if shell in self.qinfo.shells_in:
                orders[shell] = self._get_bvec_sample_order(
                    bvec, sph_dist, self.qinfo.q_in, orders.get(shell), self.choice_func(shell_rem)
                )
                q_in_order.append(shell_index[shell][orders[shell][-self.qinfo.q_in :]])
                shell_rem -= self.qinfo.q_in
            if shell in self.qinfo.shells_out:
                orders[shell] = self._get_bvec_sample_order(
                    bvec, sph_dist, self.qinfo.q_out, orders.get(shell), self.choice_func(shell_rem)
                )
                q_out_order.append(shell_index[shell][orders[shell][-self.qinfo.q_out :]])

        return q_in_order, q_out_order

    def get_subject_sample(
        self,
        bvec_shells: Dict[int, np.ndarray],
        shell_index: Dict[int, np.ndarray],
        sph_dists_shells: np.ndarray,
    ) -> np.ndarray:
        '''Gets subject Q-space sample'''
        q_in_order, q_out_order = self._get_subject_sample(
            bvec_shells, shell_index, sph_dists_shells
        )
        subject_sample = np.concatenate(q_in_order + q_out_order)
        return subject_sample
