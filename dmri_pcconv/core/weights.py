'''Pretrained weights functions'''

import os

from urllib.request import urlretrieve
from dataclasses import dataclass

from tqdm import tqdm


LOCAL_DIR = os.environ.get('DMRI_PCCONV_DIR', os.path.join(os.path.expanduser('~'), '.dmri_pcconv'))


@dataclass
class ZenodoPath:
    '''Zenodo path dataclass'''

    record: int
    fname: str

    def __post_init__(self):
        self.root_url = f'https://zenodo.org/record/{self.record}/files'

    @property
    def url(self):
        '''URL of hosted file'''
        return f'{self.root_url}/{self.fname}'


WEIGHT_URLS = {
    'pccnn': ZenodoPath(10210758, 'pccnn_weights.pt'),
    'pccnn-bv': ZenodoPath(10210778, 'pccnn-bv_weights.pt'),
    'pccnn-sp': ZenodoPath(10210876, 'pccnn-sp_weights.pt'),
    'pccnn-bv-sp': ZenodoPath(10210894, 'pccnn-bv-sp_weights.pt'),
}


class DownloadProgressBar(tqdm):
    '''TQDM Download Progress Bar'''

    def update_to(self, b=1, bsize=1, tsize=None):
        '''Progress bar download hook'''
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    '''Downloads url and saves to file with a progress bar'''
    with DownloadProgressBar(
        unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
    ) as pbar:
        urlretrieve(url, filename=output_path, reporthook=pbar.update_to)


def get_weights(model_name: str) -> str:
    '''Gets weights given model parameters, will download if not present.

    Args:
        model_dim: Model dimensionality, either 1 or 3
        shell: dMRI shell, either provide int value or "all" or "all_norm"
            str to get model weights for combined model
        q_in: Number of input q-space samples
        combined: Return combined model if available

    Returns:
        weight_dir: Weight path to be used in model.load_weights method.
            Will raise a RuntimeError if not found.
    '''
    try:
        weight = WEIGHT_URLS[model_name]
    except KeyError as err:
        raise KeyError(f'No weights available for model {model_name}.') from err

    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    output_path = os.path.join(LOCAL_DIR, weight.fname)

    if not os.path.exists(output_path):
        download_url(weight.url, output_path)

    return output_path
