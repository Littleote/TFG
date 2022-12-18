# Datasets utilitzats per les diverses proves

## Iris dataset (csv)

Dataset per 

- `Sepal.Length`, `Sepal.Width`, `Petal.Length`, `Petal.Width`: Mesures de la planta en cm.
- `Species`: Tipus de planta `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`

## Ideal reduced dataset (csv)

Dataset de l'eficàcia de diversos algorítmes amb diversos nombres de cores agafant mesures de temps, configuració per compilar, registres, memoria utilitzada, nom dels algorítmes utilitzats, quantitat ideal de cores, etc.
Dataset reduit (l'original de prop de 0.5 GB) a [...] de cada categoria de nombre de cores óptim (`ideal`). 

## Sphere dataset (py)

Dataset de punts distribuits a una esfera amb una mica de soroll en `dimension` dimensions amb `N` mostres. (`sphere(N: 'int', dimension: 'int')`)  
Utilitzat per probar els models amb dades correlacionades no linealment.

- `dimension_{i}`: I-éssim valor de la cordenada de l'esfera.
- `zone`: 'inner' si es troba a l'interior del l'esfera unitat i 'outter' si es troba fora.


## Normal dataset (py)

Dataset d'un conjunt de normals independents (gaussiana univariada) amb paràmetres mu = 0, standard deviation = 1 en `dimension` dimensions amb `N` mostres. (`normal(N: 'int', dimension: 'int')`)  
Utilitzat per probar els models amb dades descorrelacionades.

- `dimension_{i}`: valor de la gaussiana en la i-éssima dimensió.
- `zone`: 'inner' si es troba a l'interior del cercle unitat i 'outter' si es troba fora.