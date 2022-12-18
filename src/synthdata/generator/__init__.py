from ._GMM import GMM
from ._KDE import KDE

def VAE(device: 'str | device' = 'auto',
        layers_sizes: 'list | None' = None, 
        latent_dimension: 'int | None' = None, layers: 'int | None' = None,
        epochs: 'int | None' = None, search_trials: 'int' = 10,
        batches: 'int | None' = None, batch_size: 'int | None' = 128,
        learning_rate: 'float' = 1e-1,
        reduce_learning_rate: 'bool' = True, min_learning_rate: 'float' = 1e-5,
        patience: 'int | None' = None, time_limit: 'float' = 10):
    # Avoid importing more packages than necessary when possible
    from ._VAE import VAE as _VAE
    return _VAE(device=device,
            layers_sizes=layers_sizes, 
            latent_dimension=latent_dimension,
            layers=layers,
            epochs=epochs,
            search_trials=search_trials,
            batches=batches, batch_size=batch_size,
            learning_rate=learning_rate,
            reduce_learning_rate=reduce_learning_rate,
            min_learning_rate=min_learning_rate,
            patience=patience,
            time_limit=time_limit)

__all__ = ['GMM', 'KDE', 'VAE']