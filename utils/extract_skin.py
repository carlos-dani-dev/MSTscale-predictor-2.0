"""
Segmentação de pele facial via Mediapipe Face Landmarker (Tasks API).

Diferente da API antiga (mp.solutions.face_mesh, em descontinuação), a
Tasks API exige um modelo .task baixado à parte -- este módulo cuida do
download (uma única vez, com cache em disco) e da inferência.

A "segmentação de pele" aqui é uma aproximação via landmarks, não um
segmentador semântico dedicado (a Tasks API não oferece um pronto para
pele): usamos o contorno externo do rosto (hull convexo de um subconjunto
de landmarks) e removemos de dentro dele as regiões de olhos, sobrancelhas
e lábios (também via hull convexo de seus landmarks). O resultado não é
perfeito nas bordas, mas mascara a maior parte das áreas não-pele do rosto.
"""

import os
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DIR = "./"
MODEL_PATH = "face_landmarker.task"

# Índices de landmarks do FaceMesh (topologia de 478 pontos) usados para
# aproximar o contorno externo do rosto (testa, laterais, queixo).
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78,
]

NON_SKIN_REGIONS = (LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS)

SKIN_MASK_ERODE_KERNEL = "5"


_landmarker = None


def _ensure_model_downloaded():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"[skin_extractor] baixando modelo Face Landmarker -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[skin_extractor] download concluído.")
    return MODEL_PATH


def _get_landmarker():

    global _landmarker
    if _landmarker is None:
        model_path = _ensure_model_downloaded()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def _landmarks_to_points(landmarks, w, h):

    return np.array(
        [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
        dtype=np.int32,
    )


def _build_skin_mask(points, w, h, erode_kernel=SKIN_MASK_ERODE_KERNEL):

    mask = np.zeros((h, w), dtype=np.uint8)

    face_hull = cv.convexHull(points[FACE_OVAL])
    cv.fillConvexPoly(mask, face_hull, 255)

    for region in NON_SKIN_REGIONS:
        region_hull = cv.convexHull(points[region])
        cv.fillConvexPoly(mask, region_hull, 0)

    if erode_kernel > 1:
        if erode_kernel % 2 == 0:
            erode_kernel += 1

        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (erode_kernel, erode_kernel),
        )

        mask = cv.erode(
            mask,
            kernel,
            iterations=1,
        )

    return mask


def extract_skin(face_rgb):

    skin_mask = extract_skin_mask(face_rgb)

    if skin_mask is None:
        return None

    return cv.bitwise_and(
        face_rgb,
        face_rgb,
        mask=(skin_mask.astype(np.uint8) * 255),
    )


def extract_skin_mask(face_rgb):

    h, w = face_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=face_rgb,
    )

    landmarker = _get_landmarker()
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    points = _landmarks_to_points(
        result.face_landmarks[0],
        w,
        h,
    )

    mask = _build_skin_mask(points, w, h)

    return mask > 0