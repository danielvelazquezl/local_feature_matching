#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: float, Y: float, feature_width: int
) -> np.ndarray:
    """Cree features locales usando parches normalizados.

    Normalizar la intensidad de la imagen en una ventana local centrada en 
    el punto de interes a un vector cuya norma es la unidad. Este feature local
    es simple de codificar y funciona Ok.
    
    Elegir la celda top-left para centrar la ventana de tamaño par.

    Args:
        image_bw: array de shape (M,N) representando una imagen en escala de grises
        X: array de shape (K,) representando las coordenadas en X de los puntos de interés.
        Y: array de shape (K,) representando las coordenadas en Y de los puntos de interés.
        feature_width: tamaño de la ventana cuadrada.

    Returns:
        fvs: array de shape (K,D) representando los descriptores de features.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `compute_normalized_patch_descriptors` debe implementarse en' +
        'function in`part2_patch_descriptor.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
