# Introducción al Procesamiento de Imágenes

## Fundamentos de Imágenes Digitales

### Representación de Imágenes

Una imagen digital es una matriz bidimensional de píxeles. Cada píxel contiene valores numéricos que representan la intensidad de luz o color.

```
Imagen Grayscale (1 canal):
┌─────────────────────────────┐
│  0   50  100  150  200  255 │  ← Valores 0-255
│  25  75  125  175  225  250 │    0 = Negro
│  50  100 150  200  250  255 │    255 = Blanco
└─────────────────────────────┘
     Height × Width × 1

Imagen RGB (3 canales):
┌─────────────────────────────┐
│ [R,G,B] [R,G,B] [R,G,B] ... │
│ [R,G,B] [R,G,B] [R,G,B] ... │  → Height × Width × 3
│ [R,G,B] [R,G,B] [R,G,B] ... │
└─────────────────────────────┘
```

### Espacios de Color

```python
import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ColorSpaceConverter:
    """Conversión entre espacios de color."""

    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convierte RGB a escala de grises.
        Fórmula: Y = 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
        """
        Convierte RGB a HSV (Hue, Saturation, Value).
        Útil para segmentación por color.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    @staticmethod
    def rgb_to_lab(image: np.ndarray) -> np.ndarray:
        """
        Convierte RGB a LAB.
        L = Luminosidad, A = verde-rojo, B = azul-amarillo.
        Perceptualmente uniforme.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """OpenCV usa BGR por defecto, no RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Ejemplo de uso
image_bgr = cv2.imread("imagen.jpg")  # OpenCV lee en BGR
converter = ColorSpaceConverter()

image_rgb = converter.bgr_to_rgb(image_bgr)
image_gray = converter.rgb_to_grayscale(image_rgb)
image_hsv = converter.rgb_to_hsv(image_rgb)
```

## Operaciones Básicas

### Transformaciones Geométricas

```python
class GeometricTransforms:
    """Transformaciones geométricas de imágenes."""

    @staticmethod
    def resize(
        image: np.ndarray,
        size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Redimensiona imagen.

        Interpolaciones:
        - INTER_NEAREST: Más rápido, pixelado
        - INTER_LINEAR: Balance velocidad/calidad
        - INTER_CUBIC: Mejor calidad, más lento
        - INTER_LANCZOS4: Mejor para downscaling
        """
        return cv2.resize(image, size, interpolation=interpolation)

    @staticmethod
    def rotate(
        image: np.ndarray,
        angle: float,
        center: Tuple[int, int] = None
    ) -> np.ndarray:
        """Rota imagen alrededor de un centro."""
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def flip(image: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """
        Voltea imagen.
        mode: 'horizontal', 'vertical', 'both'
        """
        flip_codes = {
            "horizontal": 1,
            "vertical": 0,
            "both": -1
        }
        return cv2.flip(image, flip_codes[mode])

    @staticmethod
    def crop(
        image: np.ndarray,
        x: int, y: int,
        width: int, height: int
    ) -> np.ndarray:
        """Recorta región de interés (ROI)."""
        return image[y:y+height, x:x+width]

    @staticmethod
    def pad(
        image: np.ndarray,
        padding: Tuple[int, int, int, int],
        mode: str = "constant",
        value: int = 0
    ) -> np.ndarray:
        """
        Añade padding a imagen.
        padding: (top, bottom, left, right)
        """
        top, bottom, left, right = padding
        border_type = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "replicate": cv2.BORDER_REPLICATE
        }
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            border_type[mode], value=value
        )
```

### Normalización

```python
class ImageNormalizer:
    """Técnicas de normalización para imágenes."""

    @staticmethod
    def min_max_normalize(image: np.ndarray) -> np.ndarray:
        """Normaliza a rango [0, 1]."""
        img_float = image.astype(np.float32)
        return (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

    @staticmethod
    def standardize(
        image: np.ndarray,
        mean: Tuple[float, ...] = None,
        std: Tuple[float, ...] = None
    ) -> np.ndarray:
        """
        Estandarización Z-score.
        Si no se proveen mean/std, calcula de la imagen.

        ImageNet stats:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        """
        img_float = image.astype(np.float32) / 255.0

        if mean is None:
            mean = img_float.mean(axis=(0, 1))
        if std is None:
            std = img_float.std(axis=(0, 1))

        mean = np.array(mean)
        std = np.array(std)

        return (img_float - mean) / (std + 1e-8)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Ecualización de histograma.
        Mejora contraste distribuyendo intensidades uniformemente.
        """
        if len(image.shape) == 3:
            # Para imágenes color, aplicar en canal V de HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return cv2.equalizeHist(image)

    @staticmethod
    def clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization.
        Mejor que ecualización global para imágenes con
        variaciones locales de iluminación.
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )

        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(image)
```

## Filtros y Convolución

### Operación de Convolución

```
Convolución 2D:
                    Kernel 3×3
                   ┌─────────┐
Input    *        │-1 -1 -1 │        Output
Image              │-1  8 -1 │   =    (Edge
                   │-1 -1 -1 │        Detection)
                   └─────────┘

Operación:
┌───┬───┬───┐     ┌───┬───┬───┐
│ a │ b │ c │     │k1 │k2 │k3 │
├───┼───┼───┤  *  ├───┼───┼───┤  = Σ(elemento × kernel)
│ d │ e │ f │     │k4 │k5 │k6 │
├───┼───┼───┤     ├───┼───┼───┤
│ g │ h │ i │     │k7 │k8 │k9 │
└───┴───┴───┘     └───┴───┴───┘
```

### Filtros Comunes

```python
class ImageFilters:
    """Filtros de imagen mediante convolución."""

    # Kernels predefinidos
    KERNELS = {
        "blur_3x3": np.ones((3, 3)) / 9,
        "blur_5x5": np.ones((5, 5)) / 25,
        "sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        "edge_laplacian": np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]),
        "edge_sobel_x": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        "edge_sobel_y": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        "emboss": np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
    }

    @staticmethod
    def apply_kernel(
        image: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """Aplica convolución con kernel dado."""
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 0
    ) -> np.ndarray:
        """
        Blur gaussiano.
        Reduce ruido preservando bordes mejor que blur uniforme.
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)

    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Blur de mediana.
        Excelente para eliminar ruido salt-and-pepper.
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def bilateral_filter(
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Filtro bilateral.
        Suaviza preservando bordes (edge-preserving smoothing).
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# Ejemplo: Pipeline de preprocesamiento
def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Pipeline completo de preprocesamiento."""
    # Cargar imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reducir ruido
    image = ImageFilters.bilateral_filter(image)

    # Mejorar contraste
    normalizer = ImageNormalizer()
    image = normalizer.clahe(image)

    # Redimensionar
    transforms = GeometricTransforms()
    image = transforms.resize(image, target_size)

    # Normalizar para red neuronal
    image = normalizer.standardize(
        image,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    return image
```

## Detección de Bordes

### Algoritmo de Canny

```python
class EdgeDetector:
    """Detectores de bordes."""

    @staticmethod
    def canny(
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detector de bordes Canny.

        Pasos internos:
        1. Suavizado gaussiano (reduce ruido)
        2. Gradientes con Sobel (magnitud y dirección)
        3. Non-maximum suppression (adelgaza bordes)
        4. Hysteresis thresholding (conecta bordes)

        Args:
            low_threshold: Umbral inferior para hysteresis
            high_threshold: Umbral superior para hysteresis
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(image, low_threshold, high_threshold)

    @staticmethod
    def sobel(
        image: np.ndarray,
        direction: str = "both"
    ) -> np.ndarray:
        """
        Detector Sobel.
        Calcula gradientes en X, Y o magnitud combinada.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        if direction == "x":
            return np.abs(sobel_x).astype(np.uint8)
        elif direction == "y":
            return np.abs(sobel_y).astype(np.uint8)
        else:  # "both" - magnitud
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def laplacian(image: np.ndarray) -> np.ndarray:
        """
        Detector Laplaciano.
        Segunda derivada - detecta cambios rápidos de intensidad.
        Sensible a ruido.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.abs(laplacian).astype(np.uint8)
```

## Detección de Características

### Keypoints y Descriptores

```python
class FeatureDetector:
    """Detección de características locales."""

    def __init__(self, method: str = "orb"):
        """
        Args:
            method: 'sift', 'orb', 'akaze'
            - SIFT: Robusto, patentado hasta 2020
            - ORB: Rápido, libre, bueno para tiempo real
            - AKAZE: Balance entre velocidad y robustez
        """
        self.method = method
        self.detector = self._create_detector()

    def _create_detector(self):
        if self.method == "sift":
            return cv2.SIFT_create()
        elif self.method == "orb":
            return cv2.ORB_create(nfeatures=1000)
        elif self.method == "akaze":
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Método desconocido: {self.method}")

    def detect_and_compute(
        self,
        image: np.ndarray
    ) -> Tuple[list, np.ndarray]:
        """
        Detecta keypoints y calcula descriptores.

        Returns:
            keypoints: Lista de puntos de interés
            descriptors: Matriz de descriptores (N × D)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> list:
        """
        Empareja características entre dos imágenes.
        Usa Lowe's ratio test para filtrar malos matches.
        """
        if self.method == "orb":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = bf.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        return good_matches

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: list
    ) -> np.ndarray:
        """Visualiza keypoints en imagen."""
        return cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )


# Ejemplo: Matching de imágenes
def match_images(img1_path: str, img2_path: str) -> int:
    """Encuentra correspondencias entre dos imágenes."""
    detector = FeatureDetector(method="orb")

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)

    matches = detector.match_features(desc1, desc2)

    print(f"Keypoints imagen 1: {len(kp1)}")
    print(f"Keypoints imagen 2: {len(kp2)}")
    print(f"Matches buenos: {len(matches)}")

    return len(matches)
```

## Data Augmentation

### Técnicas de Aumento

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmentor:
    """Pipeline de data augmentation para Computer Vision."""

    @staticmethod
    def get_train_transforms(image_size: int = 224):
        """
        Transformaciones para entrenamiento.
        Aumenta diversidad del dataset.
        """
        return A.Compose([
            # Redimensionar
            A.Resize(image_size, image_size),

            # Geométricas
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),

            # Color/Brillo
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20
                ),
            ], p=0.5),

            # Ruido y blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),

            # Oclusiones (simula objetos parcialmente visibles)
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    fill_value=0
                ),
                A.GridDropout(ratio=0.3),
            ], p=0.2),

            # Normalización
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms(image_size: int = 224):
        """Transformaciones para validación (determinísticas)."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_test_time_augmentation(image_size: int = 224):
        """
        TTA: Test Time Augmentation.
        Aplica múltiples transformaciones en inferencia
        y promedia predicciones.
        """
        return [
            A.Compose([A.Resize(image_size, image_size)]),
            A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0)]),
            A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=15, p=1.0)]),
            A.Compose([
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.CenterCrop(image_size, image_size)
            ]),
        ]


# Ejemplo de uso
def create_augmented_dataset(
    images: list,
    labels: list,
    augmentor: DataAugmentor,
    augmentations_per_image: int = 5
) -> Tuple[list, list]:
    """Crea dataset aumentado."""
    train_transform = augmentor.get_train_transforms()

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # Imagen original
        augmented_images.append(img)
        augmented_labels.append(label)

        # Augmentaciones
        for _ in range(augmentations_per_image):
            transformed = train_transform(image=img)
            augmented_images.append(transformed['image'])
            augmented_labels.append(label)

    return augmented_images, augmented_labels
```

## Morfología Matemática

### Operaciones Morfológicas

```python
class MorphologicalOps:
    """Operaciones morfológicas para imágenes binarias."""

    @staticmethod
    def get_kernel(
        shape: str = "rect",
        size: Tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        """
        Crea elemento estructurante.
        shape: 'rect', 'ellipse', 'cross'
        """
        shapes = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS
        }
        return cv2.getStructuringElement(shapes[shape], size)

    @staticmethod
    def erosion(
        image: np.ndarray,
        kernel: np.ndarray,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Erosión: Encoge regiones blancas.
        Elimina ruido pequeño, separa objetos conectados.
        """
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def dilation(
        image: np.ndarray,
        kernel: np.ndarray,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Dilatación: Expande regiones blancas.
        Rellena huecos pequeños, conecta regiones cercanas.
        """
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Opening: Erosión + Dilatación.
        Elimina ruido preservando tamaño de objetos.
        """
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Closing: Dilatación + Erosión.
        Rellena huecos preservando tamaño de objetos.
        """
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def gradient(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Gradiente morfológico: Dilatación - Erosión.
        Detecta contornos de objetos.
        """
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


# Pipeline para limpiar imagen binaria
def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Limpia máscara binaria eliminando ruido."""
    morph = MorphologicalOps()
    kernel = morph.get_kernel("ellipse", (5, 5))

    # Eliminar ruido pequeño
    cleaned = morph.opening(mask, kernel)

    # Rellenar huecos
    cleaned = morph.closing(cleaned, kernel)

    return cleaned
```

## Aplicaciones en Ciberseguridad

### Análisis Visual de Capturas

```python
class SecurityImageAnalyzer:
    """Análisis de imágenes para seguridad."""

    def __init__(self):
        self.normalizer = ImageNormalizer()
        self.edge_detector = EdgeDetector()
        self.feature_detector = FeatureDetector(method="orb")

    def detect_screenshot_tampering(
        self,
        original: np.ndarray,
        suspect: np.ndarray,
        threshold: float = 0.8
    ) -> dict:
        """
        Detecta si una captura de pantalla ha sido manipulada.
        Compara características entre original y sospechosa.
        """
        # Extraer características
        _, desc1 = self.feature_detector.detect_and_compute(original)
        _, desc2 = self.feature_detector.detect_and_compute(suspect)

        # Emparejar
        matches = self.feature_detector.match_features(desc1, desc2)

        # Calcular similitud
        max_matches = min(len(desc1), len(desc2)) if desc1 is not None and desc2 is not None else 0
        similarity = len(matches) / max_matches if max_matches > 0 else 0

        return {
            "is_tampered": similarity < threshold,
            "similarity_score": similarity,
            "matches_found": len(matches),
            "confidence": abs(similarity - threshold) / threshold
        }

    def analyze_phishing_screenshot(
        self,
        image: np.ndarray,
        reference_logo: np.ndarray
    ) -> dict:
        """
        Analiza captura de página sospechosa de phishing.
        Busca logos falsificados comparando con referencia.
        """
        # Detectar logo en screenshot
        _, desc_ref = self.feature_detector.detect_and_compute(reference_logo)
        _, desc_img = self.feature_detector.detect_and_compute(image)

        if desc_ref is None or desc_img is None:
            return {"logo_detected": False, "confidence": 0}

        matches = self.feature_detector.match_features(desc_ref, desc_img)

        # Si hay suficientes matches, el logo está presente
        logo_detected = len(matches) > 10

        return {
            "logo_detected": logo_detected,
            "matches": len(matches),
            "warning": "Posible phishing - logo detectado en contexto sospechoso" if logo_detected else None
        }

    def extract_text_regions(
        self,
        image: np.ndarray
    ) -> list:
        """
        Detecta regiones de texto para posterior OCR.
        Usa MSER (Maximally Stable Extremal Regions).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # MSER para detectar regiones de texto
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Obtener bounding boxes
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            # Filtrar por aspect ratio típico de texto
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
                bboxes.append((x, y, w, h))

        return bboxes


# Ejemplo de uso
def analyze_suspicious_screenshot(image_path: str) -> dict:
    """Analiza captura sospechosa."""
    analyzer = SecurityImageAnalyzer()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar regiones de texto (posibles credenciales)
    text_regions = analyzer.extract_text_regions(image)

    # Analizar bordes (detectar ediciones)
    edges = analyzer.edge_detector.canny(image)
    edge_density = np.sum(edges > 0) / edges.size

    return {
        "text_regions_found": len(text_regions),
        "edge_density": edge_density,
        "suspicious_editing": edge_density > 0.15,  # Umbral heurístico
        "recommendations": [
            "Verificar texto extraído con OCR",
            "Comparar con capturas conocidas legítimas"
        ]
    }
```

## Resumen

| Técnica | Uso | Función OpenCV |
|---------|-----|----------------|
| Convolución | Filtros, detección bordes | `cv2.filter2D()` |
| Gaussian Blur | Reducir ruido | `cv2.GaussianBlur()` |
| Canny | Detección bordes | `cv2.Canny()` |
| SIFT/ORB | Keypoints y matching | `cv2.SIFT_create()` |
| Histogram Eq. | Mejorar contraste | `cv2.equalizeHist()` |
| Morfología | Limpiar máscaras | `cv2.morphologyEx()` |

### Checklist de Preprocesamiento

```
□ Cargar imagen en formato correcto (BGR → RGB)
□ Redimensionar a tamaño consistente
□ Normalizar (ImageNet stats si usas transfer learning)
□ Aplicar data augmentation en entrenamiento
□ NO aplicar augmentation en validación/test
□ Verificar rango de valores (0-1 o 0-255)
```

## Referencias

- OpenCV Documentation: https://docs.opencv.org/
- Albumentations: https://albumentations.ai/docs/
- Digital Image Processing - Gonzalez & Woods
