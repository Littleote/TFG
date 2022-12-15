# Instal·lació

`pip install --upgrade synnthdata`

# Ús

- `datahub = synthdata.DataHub(cores: 'int' = 1, model: 'Model' = AutoSelector())` \
Classe per manejar les dades.
    + Cores (`int`): Nombre de cores a utilitzar quan s'especifica una variable objectiu.
    Si el valor és major a 0, el nombre total de cores.
    Si el valor és igual o menor a 0, el nombre de cores que deixar lliures respecte al màxim de l'ordinador.
    En el cas d'un únic core (1 o None) no es paral·lelitzarà.
    Aquest valor es pot cambiar amb `datahub.set_cores(cores: 'int' = 1)`.
    + Model (`Model`): Model a utilitzar per defecte per analitzar les dades.
    Aquest valor es pot cambiar amb `datahub.set_model(model: 'Model' = None)`.
   
- `datahub.load(self, data: 'pandas.DataFrame', encoders: 'dict of Encoders' = dict())` \
Funció per carregar les dades.
    + Data (`pandas.DataFrame`): Dataset per analitzar.
    + Encoders (`dict of Encoders`): Llista de codificadors utilitzats per les diferents columnes del conjunt de dades. 
    En cas de que una columna no es trobi en la llista, s'utilitzarà `synthdata.encoder.auto` per assignar-li un.
    
- `datahub.kfold_validation(self, folds: 'int' = None, train_samples: 'int | None' = None, validation_samples: 'int | None' = None, validation: 'str | Validator' = 'loglikelihood', model: 'Model | None' = None, target: 'str | None' = None, return_fit: 'bool' = False, return_time: 'bool' = True)` \
Funció per validar el funcionament d'un dels models.
    + Folds (`int`): Nombre de divisions i repeticons per fer.
    + Train_samples (`int` o `None`): Nombre de mostres que limitar a l'entrenament.
    Si és `None`, utilitzar el valor màxim.
    + Validation_samples (`int` o `None`): Nombre de mostres que limitar a la validació.
    Si és `None`, utilitzar el valor màxim.
    + Validation (`str` o `Validator`): Mètode de validació.
    Si és 'loglikelihood' utilitza logversemblança.
    Si és un `Validator` utilitzar el seu resultat donant-li les mostres originals (no vistes) contra les generades.
    + Model (`Model` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + Target (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.
    + Return_fit (`bool`): Retornar els valors de validació a l'entrenament.
    + Return_time (`bool`): Retornar el temps d'entrenament, i el temps de validació.
- `datahub.generate(self, n_samples: 'int', model: 'Model | None' = None, target: 'str | None' = None)` \
Funció per generar un dataset purament sintètic a partir de les dades.
    + N_samples (`int`): Nombre de mostres que tindra el dataset de sortida.
    + Model (`Model` o `None`): Model per sobresciure el per defecte per aquesta execució. Si és `None`, utilitzar el valor per defecte.
    + Target (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta. Si és `None`, aplicar a tots els valors alhora.
- `datahub.fill(self, model: 'Model | None' = None, target: 'str | None' = None)` \
Funció per omplir forats al dataset.
    + Model (`Model` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + Target (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.
- `datahub.extend(self, n_samples: 'int', max_sample: 'str | int' = 'n_samples', model: 'Model | None' = None, target: 'str | None' = None)` \
Funció per augmentar la mida del dataset amb noves dades sintétiques.
    + N_samples (`int`): Nombre mínim de mostres que tindra el dataset de sortida.
    + Max_samples (`str` o `int`): Nombre màxim de mostres que tindra el dataset de sortida.
    Si és 'n_samples', utilitzarà el valor `n_samples`.
    Si és 'max', no imposarà màxim`.
    + Model (`Model` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + Target (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.

# Informe

Enllaç de visualització del projecte: https://www.overleaf.com/read/grpvjrtvkksv

Enllaç de visualització del treball: https://www.overleaf.com/read/nmjnmbkfsbgh
