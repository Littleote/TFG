from ._GMM import GMM
from ._KDE import KDE

def VAE(device='auto', **kwargs):
    # Avoid importing more packages than necessary when possible
    from ._VAE import VAE as _VAE
    return _VAE(device, **kwargs)

__all__ = ['GMM', 'KDE', 'VAE']