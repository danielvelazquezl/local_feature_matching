#!/usr/bin/python3

import numpy as np
import torch

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Utilice la convolución con filtros Sobel para calcular el gradiente de la imagen en cada
     píxel. 

    Args:
        image_bw: una matriz numpy de shape (M, N) que contiene la imagen en escala de grises 

    Returns:
        Ix: Array de shape (M,N) representando la derivada de la imagen con respecto de x
        Iy:  Array de shape (M,N) representando la derivada de la imagen con respecto de y
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `compute_image_gradients` necesita ser implementada en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return Ix, Iy


def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Crear un Tensor de python que represente a un kernel Gaussiano en 2d

    Args:
        ksize: dimensión del kernel cuadrado
        sigma: la deviación estandar del kernel

    Returns:
        kernel: Tensor de shape (ksize,ksize) representando a un kernel Gaussiano 2D

    Acá deberían reusar su código del proyecto 1
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_gaussian_kernel_2D_pytorch` necesita ser implementada en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Calcula los elementos de la matriz de segundo momento M.
    
    Calcula las gradientes de la imagen Ix e Iy en cada pixel, y luego los
    elementos de la matriz de segundo momento sx2, sxsy, sy2 en cada pixel, 
    utilizando una convolución con el filtro Gaussiano.

    Args:
        image_bw: array de shape (M,N) que contiene la imagen en escala de grises
        ksize: tamaño del filtro Gaussiano 2D.
        sigma: la deviación estandar del Filtro Gaussiano.

    Returns:
        sx2: array de shape (M,N) que cotiene el segundo momento en la dirección x
        sy2: array de shape (M,N) que cotiene el segundo momento en la dirección y
        sxsy: array de dim (M,N) conteniendo el segundo momento en la dirección x 
        y luego en la dirección y
    """

    sx2, sy2, sxsy = None, None, None
    ###########################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                     #
    ###########################################################################

    raise NotImplementedError('La función `second_moments` necesita ser implementada en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Calcule la puntuación de  Harris Corners en cada píxel (ver Szeliski 7.1.1) 

    Recordar que R = det(M) - alpha * (trace(M))^2
    donde M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    y * es la operación de convolución sobre un filtro Gaussiano de tamaño (k ,k).

    Para entender como funciona el proceso de convolución pueden ver el siguiente enlace:
        http://cs231n.github.io/convolutional-networks/

    Ix, Iy son solamente derivadas de las imagenes en las direcciones x e y respectivamente.
    Puedes usar la función Pytorch nn.Conv2d().
    

    Args:
        image_bw: array de shape (M,N) que contiene la imagen en escala de grises.
            ksize: tamaño del filtro Gaussiano 2D
        sigma: deviación estandar del filtro gaussiano
        alpha: término escalar en el cálculo del Harris Response Score .

    Returns:
        R: array de shape (M,N), indicando el score de esquinas en cada pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `compute_harris_response_map` debe ser implementada en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implementar el operador maxpool 2d de tamaño (ksize, ksize).
    

    Note: la implementación es identica a my_conv2d_numpy() excepto que replazamos el producto punto
    por la operación max().

    Args:
        R: array de shape (M,N) que representa el mapa de respuesta/score de 
        Harris detector. Es un mapa 2d.

    Returns:
        maxpooled_R: arreglo de shape(M,N) que representa el resultado de aplicar maxpooling 2d 
        al mapa de respuesta/score de harris.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `maxpool_numpy` debe ser implementada en' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Obtenga los k principales puntos de interés que superan los máximos locales 
    sobre un vecindario (ksize, ksize).

    HINT: Una forma sencilla de hacer non-maximum suppression es simplemente elegir 
    un máximo local sobre un tamaño de ventana (u, v). Esto puede lograrse usando
    nn.MaxPool2d. Tenga en cuenta que esto nos daría todos los máximos locales incluso 
    cuando tengan una puntuación muy baja en comparación con otros máximos locales.  
    Podría ser útil establecer el umbral de la puntuación de valor bajo antes de realizar 
    pooling (torch.median puede ser útil aquí).

    Ver https://pytorch.org/docs/stable/nn.html#maxpool2d para entender como funciona
    max pooling antes de utilizarlo.

    Aplicar un Threshold global para descartar todo lo que sea menor a la mediana hasta cero,
    luego aplicar MaxPool usando un kernel de 7x7. 
    Esto llenará todas las entradas de las subcuadrículas con el valor máximo de cada vecindario.
    Binarice la imagen según ubicaciones que sean iguales a su máximo. 1 en los máximos, 0 en el resto.
    Multiplique esta imagen binaria por los valores de R. Vamos a testear una imagen a la vez.   

    Args:
        R: mapa de score de respuestas de shape (M,N)
        k: cantidad de puntos de interes (se topa los primeros K basados en el confidence score)
        ksize: el tamaño del kernel del operador max-pooling.

    Returns:
        x: array de shape (k,) conteniendo las coordenadas en X de los puntos de interés.
        y: array de shape (k,) conteniendo las coordenadas en Y de los puntos de interés.
        c: array de shape (k,) conteniendo el puntaje de coincidencia de los K puntos de interés.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `nms_maxpool_pytorch` debe implementarse en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, confidences


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remover puntos de interés muy cercanos a un borde de la imagen para permitir a SIFT extraer features.
    Asegúrate de remover todos los puntos donde no pueda formarse una ventana de 16x16 sobre
    ese punto. 

    Args:
        img: array de shape (M,N) conteniendo la imagen en escala de grises.
        x: array de shape (k,) representando los puntos de interes en coordenadas X
        y: array de shape (k,)  representando los puntos de interes en coordenadas Y
        c: array de shape (k,) representando el puntaje de confidencia de los puntos de interés.

    Returns:
        x: array de shape (p,), donde p <= k (menor o igual a k después del recorde de puntos)
        y: array de shape (p,)
        c: array de shape (p,)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError(' La función `remove_border_vals` debe implementarse en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementar el detector Harris. Vas a usar las siguientes funciones probablemente
    compute_harris_response_map(), nms_maxpool_pytorch(), y
    remove_border_vals().
    Asegúrate de ordenar los puntos de interés en orden de confidencia.

    Args:
        image_bw: array de shape (M,N) conteniendo la imagen en escala de grises
        k: cantidad máxima de puntos de interés a retornar.

    Returns:
        x: array de shape (p,) conteniendo las coordenadas en X de los puntos de interés.
        y: array de shape (p,) conteniendo las coordenadas en Y de los puntos de interés.
        c: array de dim (p,) conteniendo el puntaje de confidencia de cada punto de 
            interest done p<= k
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('La función `get_harris_interest_points` debe implementarse en ' +
        '`part1_harris_corner.py`')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c
