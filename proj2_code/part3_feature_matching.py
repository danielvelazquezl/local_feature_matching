#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    Esta función calcula una lista de distancias desde cada feature en 
    una matriz a cada feature en otra. 

    Es necesario utilizar broadcasting de Numpy para mantener bajos 
    los requisitos de memoria. 

    Note: Usar un bucle for doble va a ser demasiado lento. 
    Un bucle for simple es el máximo posible. Se necesita vectorización. 
    Vea los detalles de broadcasting de numpy aquí: 
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: un array numpy de shape (n1,feat_dim) que representa el grupo
            de features donde feat_dim representa la dimensión del feature.
        features2: un array de shape (n2,feat_dim) que representa el segundo grupo
            de features (n1 no necesariamente es igual a n2)

    Returns:
        dists: Un array numpy de shape (n1,n2) que contiene las distancias (en el
            espacio de features) de cada feature en features1 a cada feature en features 2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `compute_feature_distances` debe implementarse en ' +
        '`part3_feature_matching.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Feature matching con ratio de distancia a Nearest-neighbor.

    Esta función no necesita ser simétrica (por ejemplo, puede producir diferentes
    número de matchings según el orden de los argumentos). 

    Para empezar, simplemente implemente el "ratio test", de la ecuación 7.18 en 
    la sección 7.1.3 de Szeliski. Hay muchos features repetitivos en estas imágenes,
    y todos sus descriptores se verán similares. La prueba de 'ratio test' nos ayuda
    a resuelver este problema (consulte también la Figura 11 del artículo IJCV de 
    David Lowe). 

    Debes llamar a `compute_feature_distances()` en esta función, y luego procesar 
    la salida.

    Args:
        features1: Un array numpy de shape (n1,feat_dim) representando un grupo de features
            donde feat_dim es la dimensión del feature (128 cuando sea SIFT).
        features2: Un array numpy de shape (n2, feat_dim) que representa al segundo 
            grupo de features (n1 no necesariamente es igual a n2)

    Returns:
        matches: Un array numpy de shape (k,2), donde k es el número de matches.
            La primera columna es un index en features1, y la segunda columna
            es un index en features2
        confidences: Un array numpy de shape (k,) con el valor real de la confidencia
            para cada match. Con el valor del ratio.

    'matches' y 'confidencias' pueden ser vacíos, ejemplo (0x2) y (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError(' La función `match_features_ratio_test` debe ser implementada en ' +
        '`part3_feature_matching.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
