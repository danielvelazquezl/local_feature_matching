#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from proj2_code.part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple


"""
Autores: Vijay Upadhya, John Lambert, Cusuh Ham, Patsorn Sangkloy, Samarth
Brahmbhatt, Frank Dellaert, James Hays, January 2021.

Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Su implementación no coincidirá exactamente con la referencia de SIFT. Por ejemplo,
excluiremos la invariancia de rotación y escala. 
 
No es necesario realizar la interpolación en la que cada gradiente
contribuye a múltiples contenedores de orientación en múltiples celdas. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esta función devolverá las magnitudes y orientaciones de las
    gradientes en cada ubicación de píxel. 

    Args:
        Ix: array de shape (m,n), representando las gradientes en x de la imagen
        Iy: array de shape (m,n), representando las gradientes en y de la imagen
    Returns:
        magnitudes: Un array numpy de shape (m,n), representando las magnitudes de 
            las gradientes en cada posición de un pixel.
        orientations: Un array numpy de shape (m,n), representando los ángulos de las
            gradientes en cada pixel. Los angulos deben tener un rango de -PI a PI.
    """
    magnitudes = []  # placeholder
    orientations = []  # placeholder

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_magnitudes_and_orientations()` debe implementarse en ' +
        'in `part4_sift_descriptor.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Dado un parche de 16x16, forme un vector de histogramas de gradiente de 128D 

    Propiedades clave para implementar: 
    (1) una cuadrícula de celdas de 4x4, cada feature_width/4. 
        Es simplemente la terminología utilizada en la literatura de features 
        para describir los contenedores donde se describirán distribuciones de las 
        gradientes. La grilla se va a extender feature_with/2 hacia la izquierda del
        centro, y feature_with/2 - 1 hacia la derecha.

    (2) cada celda debe tener un histograma de la distribución local de gradientes
        en 8 orientaciones. Concatenando estos histogramas nos va a dar un vector
        de 4x4x8 = 128 dimensiones. Los centros de loscontendores para el histograma 
        deben ir desde -7pi/8, -5pi/8 .... 5pi/8, 7pi/8. Los histogramas deben ser agregados
        al vector de features de izquierda a derecha y luego fila por fila (como leerían
        normalmente)
    No normalice aquí el histograma a la norma de la unidad; conserve el histograma
    con sus valores. Una función útil para usar sería np.histogram. 

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_gradient_histogram_vec_from_patch` debe implementarse en' +
        'function in `part4_sift_descriptor.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
    x: float,
    y: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    Esta función devuelve el vector de features para un punto de interés 
    específico. Para empezar, es posible que desee utilizar simplemente parches 
    normalizados como su feature local. Esto es muy simple de codificar y funciona bien. 
    Sin embargo, para tener el puntaje correcto, necesitará implementar el descriptor SIFT 
    más efectivo (ver Szeliski 7.1.2 o las publicaciones originales en 
    http://www.cs.ubc.ca/~lowe/keypoints/). Tu implementación no necesita ser idéntica
    a la de las referencias.

    Su descriptor (de referencia) debe tener: 
    (1) Cada feature debe normalizarse a la longitud de la unidad. 
    (2) Cada feature debe elevarse a la potencia de 2 (square-root SIFT)
        (leer https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

    Para nuestras pruebas, no es necesario realizar la interpolación en la que cada
    magnitud de gradiente contribuye a múltiples contenedores de orientación en múltiples
    celdas. Como se describe en Szeliski, una sola magnitude de gradiente crea una
    contribución ponderada a las 4 celdas más cercanas y los 2 contenedores de orientaciones más 
    cercanos dentro de cada celda, para 8 contribuciones totales. 

    Args:
        x: a float, the x-coordinate of the interest point
        y: A float, the y-coordinate of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            Puede asumir que feature_width será un múltiplo de 4 (es decir,
            cada celda de feature local SIFT tendrá un número entero
            para la anchura y altura). Este es el tamaño de ventana inicial que 
            examinamos alrededor de cada punto de interés. 
    Returns:
        fv: Un array numpy de shape (feat_dim,1) representado un feature vector.
            "feat_dim" es la dimension del feature (ejemplo., 128 para SIFT estandar). 
            Estos son los features desriptores.
    """

    fv = []  # placeholder

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_feat_vec` debe implementarse en ' +
        '`student_sift.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fv


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    Esta función devuelve los features SIFT de 128-d calculados en cada  uno de los putnos.
    Implementar el descriptor SIFT más efectivo  (ver Szeliski 4.1.2 o
    las publicaciones originales en http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer que representa el width del feature en pixels.
            Puede asumir que feature_width será un múltiplo de 4 (es decir,
            cada celda de feature local SIFT tendrá un número entero
            para la anchura y altura). Este es el tamaño de ventana inicial que 
            examinamos alrededor de cada punto de interés. 
    Returns:
        fvs: Un array numpy de shape (k, feat_dim) respresentando todos los feature 
        vectors. "feat_dim" es la dimensión del vector. 128 en nuestro caso para
        el SIFT estandar. Estos son los descriptores de los features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_SIFT_descriptors` debe implementarse en' +
        '`part4_sift_descriptor.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


### ----------------- OPCIONAL (ABAJO) ------------------------------------

## PUNTAJE EXTRA

def get_sift_features_vectorized(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_SIFT_features_vectorized` debe implementarse en' +
        '`part4_sift_descriptor.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
