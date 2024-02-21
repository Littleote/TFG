# Instal·lació

`pip install synthdata`

# Documentació

## DataHub

- `datahub = synthdata.DataHub(cores: 'int' = 1, model: 'Generator' = AutoSelection(), **kwargs)` \
Classe per manejar les dades i encarregarse de cridar a totes les funcions necessaries.
    + **cores** (`int`): Nombre de cores a utilitzar quan s'especifica una variable objectiu.
    Si el valor és major a 0, el nombre total de cores.
    Si el valor és igual o menor a 0, el nombre de cores que deixar lliures respecte al màxim de l'ordinador.
    En el cas d'un únic core (1 o None) no es paral·lelitzarà.
    Aquest valor es pot cambiar amb `datahub.set_cores(cores: 'int' = 1)`.
    + **model** (`Generator`): Model a utilitzar per defecte per analitzar les dades.
    Aquest valor es pot cambiar amb `datahub.set_model(model: 'Generator' = None)`.
    + **kwargs**: Arguments pels pasos extra de la transformació.
        + **normalize** (`bool`): Normalitzar o no les dades. (`True` per defecte)
        + **remove_cov** (`bool`): Unificar dades amb alta covariancia o no. (`True` per defecte)
        + **whitening** (`bool`): Blanquejar o no les dades. (`False` per defecte)
   
- `datahub.load(data: 'DataFrame', encoders: 'dict[str, Encoder]' = dict())` \
Funció per carregar les dades i poder especificar el tipus i comportament de les dades.
    + **data** (`pandas.DataFrame`): Dataset per analitzar.
    + **encoders** (`dict[str, Encoder]`): Llista de codificadors utilitzats per les diferents columnes del conjunt de dades. 
    En cas de que una columna no es trobi en la llista, s'utilitzarà `synthdata.encoder.auto` per assignar-li un.
    Aquest valor es pot cambiar amb `datahub.add_encoders(self, encoders: 'dict[str, Encoder]')` o  `datahub.set_encoder(self, label: 'str', encoder: 'Encoder | str')`
    
- `datahub.kfold_validation(folds: 'int' = None, train_samples: 'int | None' = None, validation_samples: 'int | None' = None, validation: 'str | function' = 'loglikelihood', model: 'Generator | None' = None, target: 'str | None' = None, return_fit: 'bool' = False, return_time: 'bool' = True)` \
Funció per validar el correcte funcionament d'un dels models.
    + **folds** (`int`): Nombre de divisions i repeticons per fer.
    + **train_samples** (`int` o `None`): Nombre de mostres que limitar a l'entrenament.
    Si és `None`, utilitzar el valor màxim.
    + **validation_samples** (`int` o `None`): Nombre de mostres que limitar a la validació.
    Si és `None`, utilitzar el valor màxim.
    + **validation** (`str` o `function`): Mètode de validació.
    Si és 'loglikelihood' utilitza logversemblança.
    Si és una `function` utilitzar el seu resultat donant-li les mostres originals (no vistes) i les generades pel model.
    + **model** (`Generator` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + **target** (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.
    + **return_fit** (`bool`): Retornar els valors de validació a l'entrenament.
    + **return_time** (`bool`): Retornar el temps d'entrenament, i el temps de validació.
    
- `datahub.generate(n_samples: 'int', model: 'Generator | None' = None, target: 'str | None' = None)` \
Funció per generar un dataset purament sintètic a partir de les dades.
    + **n_samples** (`int`): Nombre de mostres que tindra el dataset de sortida.
    + **model** (`Generator` o `None`): Model per sobresciure el per defecte per aquesta execució. Si és `None`, utilitzar el valor per defecte.
    + **target** (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta. Si és `None`, aplicar a tots els valors alhora.
    
- `datahub.fill(model: 'Generator | None' = None, target: 'str | None' = None)` \
Funció per omplir forats o remplaçar dades invalides al dataset.
    + **model** (`Generator` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + **target** (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.
    
- `datahub.extend(self, n_samples: 'str | int', max_samples: 'str | int' = 'n_samples', on_empty: 'str' = 'ignore', model: 'Generator | None' = None, target: 'str | None' = None)` \
Funció per augmentar (o disminuir) la mida del dataset amb noves dades sintétiques.
    + **n_samples** (`str` o `int`): Nombre mínim de mostres que tindra el dataset de sortida.
    Si és 'min_target' o 'max_target', utilitzara el nombre de mostres de la categoria amb menys o més mostres respectivament.
    + **max_samples** (`str` o `int`): Nombre màxim de mostres que tindra el dataset de sortida.
    Si és 'n_samples', utilitzarà el valor `n_samples`.
    Si és 'max', no imposarà màxim`.
    + **on_empty** (`str`): Que fer pels casos on hi ha valor buits o invalids.
    + **model** (`Generator` o `None`): Model per sobresciure el per defecte per aquesta execució.
    Si és `None`, utilitzar el valor per defecte.
    + **target** (`str` o `None`): Nom d'una columna per si es vol aplicar la funció independentment a tots els diferents valors d'aquesta.
    Si és `None`, aplicar a tots els valors alhora.

## Generadors

- `generator = synthdata.generator.GMM(k: 'int | None' = None, k_max: 'int | None' = None, multivariate: 'bool' = True, iterations_limit: 'int' = 1000, time_limit: 'float' = 1, llh_tolerance: 'float' = 1e-3, attempts: 'int' = 3, criterion: 'str' = 'BIC')`
    + **k** (`int` o `None`): Nombre de gaussianes que s'utilitzaràn.
    Si és `None`, és farà una cerca del valor òptim
    + **k_max** (`int` o `None`): Límit de la cerca del valor de `k`.
    Si és `None`, no hi haurà límit
    + **multivariate** (`bool`): Utilitzar gaussianes multivariades (normals amb covariàncies) o univariades (normals sense covariaàcies).
    + **iterations_limit** (`int`): Nombre límit d'iteracions pel mètode d'_expectació-maximització_.
    + **time_limit** (`float`): Temps límit pel mètode d'_expectació-maximització_.
    + **llh_tolerance** (`float`): Tolerancia màxima de la logversemblança per finalitzar el mètode d'_expectació-maximització_.
    + **attempts** (`int`): Nombre d'execucions del mètode d'_expectació-maximització_.
    + **criterion** (`str`): Criteri per determinar el millor valor de `k`.
    Pot tenir el valor `Bayesian`, `b` o `BIC` pel criteri d'informació bayesia, el valor `Akaike`, `a` o `AIC` pel criteri d'informació d'Akaike i el valor `Cross-Validation` o `CV` pel criteri d'informació de validació creuada. (No importa majúscules ni minúscules)
    
- `generator = synthdata.generator.KDE(h: 'float | str | function' = 'tune')`
    + **h** (`float`, `str` o `function`): Valor d'`h` o funció per obtenir-lo.
    Si és un numero, s'utilitzarà directament.
    Si és `auto` o `tune` és fara una cerca pel millor valor i si és `constant` o `c` s'utilitzarà un valor general per les dimensions concretes.
    Si és una funció (`func(X: `ndarray`) -> float`), s'utilitzarà aquesta per assignar el valor d'`h`.
- `generator = synthdata.generator.VAE(device: 'str | device' = 'auto', layers_sizes: 'list | None' = None,  latent_dimension: 'int | None' = None, layers: 'int | None' = None, epochs: 'int | None' = None, search_trials: 'int' = 10, batches: 'int | None' = None, batch_size: 'int | None' = 128, learning_rate: 'float' = 1e-1, reduce_learning_rate: 'bool' = True, min_learning_rate: 'float' = 1e-5, patience: 'int | None' = None, time_limit: 'float' = 10')`
    + **device** (`str` o `device`): Lloc a on executar la xarxa neuronal. (p.e. 'cpu', 'cuda', ...)
    + **layers_sizes** (`list` o `None`): Mida de les capes de la xarxa.
    Si és `None`, generar-ne a partir dels valors de `latent_dimension` i `layers`
    + **latent_dimension** (`int` o `None`): Dimensió de l'espai latent a l'executar el model.
    Si `layers_sizes` té un valor, agafarà automàticament l'últim.
    Si és `None`, cercar el paràmetre al entrenar el model.
    + **layers** (`int` o `None`): Nombre de capes del model.
    Si `layers_sizes` té un valor, agafarà automàticament el nombre de capes que aquest indica.
    Si és `None`, cercar el paràmetre al entrenar el model.
    + **epochs** (`int` o `None`): Numero d'époques d'execució de la xarxa.
    Si és `None`, l'entrenament de la xarxa acabarà quan el ritme d'aprenentatge hagi arribat al mínim o quan la xarxa hagi deixat de millorar durant `patience` époques si `reduce_learning_rate` és fals.
    + **search_trials** (`int`): Nombre d'iteracions per fer la cerca dels parametres per la mida de la xarxa si aquests no s'han especificat.
    + **batches** (`int` o `None`): Nombre de tandes en que dividir les dades d'entrenament.
    Si és `None`,
    + **batch_size** (`int` o `None`): Mida de les tandes de les dades d'entrenament.
    Si és `None`,
    + **learning_rate** (`float`): Ritme d'aprenentatge de la xarxa.
    + **reduce_learning_rate** (`bool`): Decrementar o no el ritme d'aprenentatge quan la xarxa deixi de millorar
    + **min_learning_rate** (`float`): Mínim pel valor d'aprenentatge de la xarxa
    + **patience** (`int` o `None`): Límit d'iteracions per decrementar el ritme d'aprenentatge si la xarxa no millora i `reduce_learning_rate` és cert.
    Límit d'iteracions per finalitzar l'aprenentatge de la xarxa quan aquesta no millora si `reduce_learning_rate` és fals. (Per defecte 10 si es finalitza per époques, 5 si es finalitza pel ritme d'aprenentatge i 20 si es finalitza per estancar-se)
    + **time_limit** (`float`): Temps límit per entrenar la xarxa.

## Codificadors

- `encoder = synthdata.encoder.auto(data: 'DataFrame')`
    + **data** (`DataFrame`): Valors d'on extreure la informació necessaria per assignar-li el codificador corresponent.
- `encoder = synthdata.encoder.greater(value: 'float' = 0, include: 'bool' = True, influence: 'float' = 1)`
    + **value** (`float`): Valor mínim de la variable
    + **include** (`bool`): Indicar si el valor extrem està inclós entre els possibles de la variable.
    Si és fals, generarà cues a l'hora de transformar l'extrem per tal de que aquest valor no aparegui.
    + **influence** (`float`): Escala d'influcencia de la cua per casos on hi ha, quan més gran sigui, més abans apareixerà la cua.
    
- `encoder = synthdata.encoder.lower(value: 'float' = 0, include: 'bool' = True, influence: 'float' = 1)`
    + **value** (`float`): Valor màxim de la variable
    + **include** (`bool`): Indicar si el valor extrem està inclós entre els possibles de la variable.
    Si és fals, generarà cues a l'hora de transformar l'extrem per tal de que aquest valor no aparegui.
    + **influence** (`float`): Escala d'influcencia de la cua per casos on hi ha, quan més gran sigui, més abans apareixerà la cua.
    
- `encoder = synthdata.encoder.none()`

- `encoder = synthdata.encoder.discrete(minimum: 'int | None' = None, maximum: 'int | None' = None)`
    + **minimum** (`int` o `None`): Valor mínim de la variable.
    + **maximum** (`int` o `None`): Valor màxim de la variable.
    
- `encoder = synthdata.encoder.ignore(default = None)`
    + **default** (Qualsevol): Valor per posar a la variable al fer la transformació inversa.
    
- `encoder = synthdata.encoder.limit(lower: 'float | None' = None, upper: 'float | None' = None, tails: 'bool' = True, influence: 'float' = 1)`
    + **lower** (`float` o `None`): Valor mínim de la variable.
    + **upper** (`float` o `None`): Valor màxim de la variable.
    + **tails** (`bool`): Transformar els extrems en cues (per variables en intervals oberts) o no (per variables en intervals tancats).
    + **influence** (`float`): Escala d'influcencia de la cua per casos amb només un límit, quan més gran sigui, més abans apareixerà la cua.
    
- `encoder = synthdata.encoder.OHE(symbols: 'list')`
    + **symbols** (`list`): Llistat de símbols (noms, numeros, ...) que hi ha.
    
- `encoder = synthdata.encoder.scale(symbols: 'list')`
    + **symbols** (`list`): Llistat de símbols (noms, numeros, ...) que hi ha en el mateix ordre que en la escala que representen.

# Exemples

Omplir un dataset (`df`) amb forats:
```python
import synthdata as sd

dh = sd.DataHub()
dh.load(df)
df_filled = dh.fill()
```

Balancejar un dataset (`df`) perquè totes les categories de 'label' estiguin igual de representades.
```python
import synthdata as sd

dh = sd.DataHub()
dh.load(df)
df_balanced = dh.extend(n_samples='max_target', target='label')
```

Augmentar el nombre de dades d'un dataset (`df`).
```python
import synthdata as sd

dh = sd.DataHub()
dh.load(df)
df_augmented = dh.extend(n_samples=2*len(df))
```

# Informe

Enllaç de visualització del projecte: https://www.overleaf.com/read/grpvjrtvkksv

Enllaç de visualització del treball: https://www.overleaf.com/read/nmjnmbkfsbgh
