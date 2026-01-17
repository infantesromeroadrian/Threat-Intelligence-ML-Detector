# Reducción de Dimensionalidad en Ciberseguridad

## 1. Panorama General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              REDUCCIÓN DE DIMENSIONALIDAD EN SEGURIDAD                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │    MALWARE      │    │      IDS        │    │     LOGS        │        │
│   │   ANALYSIS      │    │   NETWORK       │    │   ANALYSIS      │        │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘        │
│            │                      │                      │                  │
│            ▼                      ▼                      ▼                  │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                 ALTA DIMENSIONALIDAD                             │      │
│   │  • PE Features: 100-1000+    • Flow features: 50-200            │      │
│   │  • API calls: 1000+          • Packet features: 100+            │      │
│   │  • Strings: 10000+           • Log events: 500+                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │              TÉCNICAS DE REDUCCIÓN                               │      │
│   │                                                                  │      │
│   │  Feature Selection          Feature Extraction                  │      │
│   │  ├─ Mutual Information     ├─ PCA                              │      │
│   │  ├─ RF Importance          ├─ t-SNE (visualización)            │      │
│   │  ├─ L1 Regularization      ├─ UMAP                             │      │
│   │  └─ Chi-squared            └─ Autoencoders                      │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    BENEFICIOS                                    │      │
│   │  ✓ Menor tiempo de inferencia (crítico en tiempo real)          │      │
│   │  ✓ Menor costo de extracción de features                        │      │
│   │  ✓ Modelos más interpretables                                   │      │
│   │  ✓ Reducción de ruido y overfitting                             │      │
│   │  ✓ Visualización para análisis humano                           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Análisis de Malware

### 2.1 Features de Ejecutables PE

Un ejecutable Windows PE puede tener cientos de features extraíbles.

```python
import numpy as np
import pefile
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PEFeatureExtractor:
    """
    Extractor de features de ejecutables PE.
    Genera 100+ features que luego necesitan reducción.
    """

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extrae features de un PE."""
        features = {}

        try:
            pe = pefile.PE(file_path)

            # === Features de cabecera (20+) ===
            features.update(self._extract_header_features(pe))

            # === Features de secciones (30+) ===
            features.update(self._extract_section_features(pe))

            # === Features de imports (20+) ===
            features.update(self._extract_import_features(pe))

            # === Features de exports (10+) ===
            features.update(self._extract_export_features(pe))

            # === Features de recursos (10+) ===
            features.update(self._extract_resource_features(pe))

            # === Features de entropía (10+) ===
            features.update(self._extract_entropy_features(pe))

            pe.close()

        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            return {}

        return features

    def _extract_header_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features del DOS/PE header."""
        h = pe.FILE_HEADER
        oh = pe.OPTIONAL_HEADER

        return {
            'machine': h.Machine,
            'num_sections': h.NumberOfSections,
            'timestamp': h.TimeDateStamp,
            'pointer_symbol_table': h.PointerToSymbolTable,
            'num_symbols': h.NumberOfSymbols,
            'size_optional_header': h.SizeOfOptionalHeader,
            'characteristics': h.Characteristics,
            # Optional Header
            'magic': oh.Magic,
            'major_linker_version': oh.MajorLinkerVersion,
            'minor_linker_version': oh.MinorLinkerVersion,
            'size_of_code': oh.SizeOfCode,
            'size_of_initialized_data': oh.SizeOfInitializedData,
            'size_of_uninitialized_data': oh.SizeOfUninitializedData,
            'entry_point': oh.AddressOfEntryPoint,
            'base_of_code': oh.BaseOfCode,
            'image_base': oh.ImageBase,
            'section_alignment': oh.SectionAlignment,
            'file_alignment': oh.FileAlignment,
            'size_of_image': oh.SizeOfImage,
            'size_of_headers': oh.SizeOfHeaders,
            'checksum': oh.CheckSum,
            'subsystem': oh.Subsystem,
            'dll_characteristics': oh.DllCharacteristics,
            'size_of_stack_reserve': oh.SizeOfStackReserve,
            'size_of_heap_reserve': oh.SizeOfHeapReserve,
            'num_rva_and_sizes': oh.NumberOfRvaAndSizes,
        }

    def _extract_section_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features agregadas de secciones."""
        sections = pe.sections

        if not sections:
            return {f'section_{k}': 0 for k in [
                'count', 'total_raw_size', 'total_virtual_size',
                'entropy_mean', 'entropy_max', 'entropy_min',
                'executable_count', 'writable_count'
            ]}

        raw_sizes = [s.SizeOfRawData for s in sections]
        virtual_sizes = [s.Misc_VirtualSize for s in sections]
        entropies = [s.get_entropy() for s in sections]

        executable = sum(1 for s in sections
                        if s.Characteristics & 0x20000000)  # IMAGE_SCN_MEM_EXECUTE
        writable = sum(1 for s in sections
                      if s.Characteristics & 0x80000000)  # IMAGE_SCN_MEM_WRITE

        return {
            'section_count': len(sections),
            'section_total_raw_size': sum(raw_sizes),
            'section_total_virtual_size': sum(virtual_sizes),
            'section_entropy_mean': np.mean(entropies),
            'section_entropy_max': max(entropies),
            'section_entropy_min': min(entropies),
            'section_entropy_std': np.std(entropies),
            'section_executable_count': executable,
            'section_writable_count': writable,
            'section_raw_size_max': max(raw_sizes),
            'section_raw_size_min': min(raw_sizes),
            'section_size_ratio': max(raw_sizes) / (min(raw_sizes) + 1),
        }

    def _extract_import_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features de imports (APIs usadas)."""
        features = {
            'imports_total': 0,
            'imports_dll_count': 0,
            'imports_suspicious_dll_count': 0,
        }

        # DLLs sospechosas comunes en malware
        suspicious_dlls = {
            'ntdll.dll', 'kernel32.dll', 'advapi32.dll',
            'ws2_32.dll', 'wininet.dll', 'urlmon.dll',
            'crypt32.dll', 'shell32.dll'
        }

        # APIs sospechosas por categoría
        suspicious_apis = {
            'file': ['CreateFile', 'WriteFile', 'DeleteFile', 'CopyFile', 'MoveFile'],
            'process': ['CreateProcess', 'OpenProcess', 'TerminateProcess',
                       'VirtualAlloc', 'VirtualProtect', 'WriteProcessMemory'],
            'registry': ['RegSetValue', 'RegCreateKey', 'RegDeleteKey'],
            'network': ['socket', 'connect', 'send', 'recv', 'InternetOpen',
                       'URLDownloadToFile', 'HttpOpenRequest'],
            'crypto': ['CryptEncrypt', 'CryptDecrypt', 'CryptGenKey'],
            'injection': ['NtCreateThreadEx', 'RtlCreateUserThread',
                         'SetWindowsHookEx', 'CreateRemoteThread'],
        }

        api_counts = {cat: 0 for cat in suspicious_apis}

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            features['imports_dll_count'] = len(pe.DIRECTORY_ENTRY_IMPORT)

            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', errors='ignore').lower()

                if dll_name in suspicious_dlls:
                    features['imports_suspicious_dll_count'] += 1

                if hasattr(entry, 'imports'):
                    for imp in entry.imports:
                        features['imports_total'] += 1

                        if imp.name:
                            api_name = imp.name.decode('utf-8', errors='ignore')
                            for cat, apis in suspicious_apis.items():
                                if any(api in api_name for api in apis):
                                    api_counts[cat] += 1

        # Añadir counts por categoría
        for cat, count in api_counts.items():
            features[f'imports_{cat}_count'] = count

        return features

    def _extract_export_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features de exports."""
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            exports = pe.DIRECTORY_ENTRY_EXPORT.symbols
            return {
                'exports_count': len(exports),
                'exports_has_name': sum(1 for e in exports if e.name),
            }
        return {'exports_count': 0, 'exports_has_name': 0}

    def _extract_resource_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features de recursos embebidos."""
        features = {'resource_count': 0, 'resource_total_size': 0}

        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            def count_resources(entry, depth=0):
                count = 0
                size = 0
                if hasattr(entry, 'directory'):
                    for e in entry.directory.entries:
                        c, s = count_resources(e, depth + 1)
                        count += c
                        size += s
                elif hasattr(entry, 'data'):
                    count = 1
                    size = entry.data.struct.Size
                return count, size

            for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                c, s = count_resources(entry)
                features['resource_count'] += c
                features['resource_total_size'] += s

        return features

    def _extract_entropy_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Features de entropía (indicador de packing/cifrado)."""
        import math

        def calculate_entropy(data: bytes) -> float:
            if not data:
                return 0.0
            entropy = 0
            for x in range(256):
                p_x = data.count(bytes([x])) / len(data)
                if p_x > 0:
                    entropy -= p_x * math.log2(p_x)
            return entropy

        # Entropía del archivo completo
        with open(pe.path, 'rb') as f:
            data = f.read()

        file_entropy = calculate_entropy(data)

        # Entropía de primeros 1KB (headers)
        header_entropy = calculate_entropy(data[:1024])

        # Entropía de overlay (datos después del PE)
        overlay_offset = pe.get_overlay_data_start_offset() or len(data)
        overlay_entropy = calculate_entropy(data[overlay_offset:]) if overlay_offset < len(data) else 0

        return {
            'file_entropy': file_entropy,
            'header_entropy': header_entropy,
            'overlay_entropy': overlay_entropy,
            'overlay_size': len(data) - overlay_offset if overlay_offset < len(data) else 0,
            'is_likely_packed': 1 if file_entropy > 7.0 else 0,
        }


# === Reducción de Dimensionalidad para Malware ===

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
import pandas as pd

class MalwareFeatureReducer:
    """
    Pipeline de reducción de dimensionalidad para análisis de malware.

    Estrategia:
    1. Eliminar features constantes
    2. Selección por importancia (RF + MI)
    3. PCA opcional para las restantes
    """

    def __init__(self,
                 n_features_select: int = 50,
                 use_pca: bool = False,
                 pca_variance: float = 0.95):
        self.n_features_select = n_features_select
        self.use_pca = use_pca
        self.pca_variance = pca_variance

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'MalwareFeatureReducer':
        """
        Ajusta el reductor.
        """
        self.feature_names_ = feature_names

        # 1. Eliminar features constantes
        variances = np.var(X, axis=0)
        self.variance_mask_ = variances > 1e-10

        X_var = X[:, self.variance_mask_]
        names_var = [n for n, m in zip(feature_names, self.variance_mask_) if m]

        print(f"Después de eliminar constantes: {X_var.shape[1]} features")

        # 2. Calcular importancias combinadas
        # 2a. Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_var, y)
        rf_importance = rf.feature_importances_

        # 2b. Mutual Information
        mi_importance = mutual_info_classif(X_var, y, random_state=42)

        # 2c. Combinar (promedio normalizado)
        rf_norm = rf_importance / (rf_importance.max() + 1e-10)
        mi_norm = mi_importance / (mi_importance.max() + 1e-10)
        combined_importance = (rf_norm + mi_norm) / 2

        # 3. Seleccionar top features
        n_select = min(self.n_features_select, X_var.shape[1])
        top_indices = np.argsort(combined_importance)[-n_select:]

        self.selected_indices_ = top_indices
        self.selected_names_ = [names_var[i] for i in top_indices]

        # Guardar importancias para análisis
        self.importance_report_ = pd.DataFrame({
            'feature': names_var,
            'rf_importance': rf_importance,
            'mi_importance': mi_importance,
            'combined': combined_importance,
            'selected': [i in top_indices for i in range(len(names_var))]
        }).sort_values('combined', ascending=False)

        # 4. PCA opcional
        if self.use_pca:
            X_selected = X_var[:, top_indices]
            self.pca_ = PCA(n_components=self.pca_variance)
            self.pca_.fit(X_selected)
            print(f"PCA: {self.pca_.n_components_} componentes "
                  f"({self.pca_variance*100}% varianza)")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la reducción."""
        # Aplicar variance mask
        X_var = X[:, self.variance_mask_]

        # Seleccionar features
        X_selected = X_var[:, self.selected_indices_]

        # PCA opcional
        if self.use_pca:
            X_selected = self.pca_.transform(X_selected)

        return X_selected

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Retorna top N features más importantes."""
        return self.importance_report_.head(n)
```

### 2.2 Visualización de Familias de Malware

```python
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List

class MalwareFamilyVisualizer:
    """
    Visualización de familias de malware usando UMAP/t-SNE.

    Uso típico:
    - Identificar clusters de variantes
    - Detectar nuevas familias
    - Analizar evolución de malware
    """

    def __init__(self,
                 method: str = 'umap',
                 n_components: int = 2,
                 random_state: int = 42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray,
                      labels: np.ndarray = None,
                      label_names: Dict[int, str] = None) -> np.ndarray:
        """
        Reduce dimensionalidad para visualización.

        Args:
            X: Features de malware (n_samples, n_features)
            labels: Etiquetas de familia (opcional)
            label_names: Mapeo id -> nombre de familia
        """
        # Escalar
        X_scaled = self.scaler.fit_transform(X)

        # Reducir
        if self.method == 'umap':
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=self.random_state
            )
        else:  # tsne
            reducer = TSNE(
                n_components=self.n_components,
                perplexity=30,
                learning_rate='auto',
                init='pca',
                random_state=self.random_state
            )

        self.embedding_ = reducer.fit_transform(X_scaled)
        self.labels_ = labels
        self.label_names_ = label_names

        return self.embedding_

    def plot(self, figsize=(12, 10), point_size=20, alpha=0.6,
             highlight_samples: List[int] = None):
        """
        Genera visualización del embedding.

        Args:
            highlight_samples: Índices de muestras a destacar (ej: nuevas muestras)
        """
        fig, ax = plt.subplots(figsize=figsize)

        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = self.labels_ == label
                name = self.label_names_.get(label, f"Family {label}") \
                       if self.label_names_ else f"Family {label}"

                ax.scatter(
                    self.embedding_[mask, 0],
                    self.embedding_[mask, 1],
                    c=[colors[i]],
                    label=name,
                    s=point_size,
                    alpha=alpha
                )

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(
                self.embedding_[:, 0],
                self.embedding_[:, 1],
                s=point_size,
                alpha=alpha
            )

        # Destacar muestras específicas
        if highlight_samples:
            ax.scatter(
                self.embedding_[highlight_samples, 0],
                self.embedding_[highlight_samples, 1],
                c='red',
                s=point_size * 3,
                marker='*',
                label='Nuevas muestras',
                edgecolors='black',
                linewidths=1
            )

        ax.set_xlabel(f'{self.method.upper()} 1')
        ax.set_ylabel(f'{self.method.upper()} 2')
        ax.set_title(f'Visualización de Familias de Malware ({self.method.upper()})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def find_similar_samples(self, query_idx: int, n_neighbors: int = 5) -> List[int]:
        """
        Encuentra muestras similares en el espacio reducido.
        Útil para análisis de variantes.
        """
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(self.embedding_)

        distances, indices = nn.kneighbors([self.embedding_[query_idx]])

        # Excluir la propia muestra
        return indices[0][1:].tolist()


# Ejemplo de uso
np.random.seed(42)

# Simular dataset de malware con familias
n_samples = 500
n_features = 100

# Generar clusters para diferentes familias
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=6,  # 6 familias
    cluster_std=[1.5, 2.0, 1.0, 2.5, 1.8, 1.2],
    random_state=42
)

family_names = {
    0: 'Emotet',
    1: 'TrickBot',
    2: 'Ryuk',
    3: 'Dridex',
    4: 'QakBot',
    5: 'IcedID'
}

# Visualizar
visualizer = MalwareFamilyVisualizer(method='umap')
embedding = visualizer.fit_transform(X, labels=y, label_names=family_names)

# Simular nuevas muestras a analizar
new_sample_indices = [10, 50, 100]  # Muestras a investigar

fig = visualizer.plot(highlight_samples=new_sample_indices)
plt.show()

# Encontrar variantes similares
for idx in new_sample_indices:
    similar = visualizer.find_similar_samples(idx)
    print(f"\nMuestra {idx} ({family_names[y[idx]]}) similar a:")
    for s in similar:
        print(f"  - Muestra {s} ({family_names[y[s]]})")
```

---

## 3. Detección de Intrusiones (IDS/IPS)

### 3.1 Reducción para Network Flow Features

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from typing import List, Tuple

class NetworkFlowReducer:
    """
    Reducción de dimensionalidad para features de flujo de red.

    Datasets típicos:
    - CICIDS2017: ~80 features
    - NSL-KDD: ~41 features
    - UNSW-NB15: ~49 features

    Objetivo: Reducir a 15-25 features manteniendo detección.
    """

    # Features típicas de flujo de red
    FLOW_FEATURES = [
        # Básicas
        'duration', 'protocol_type', 'service', 'flag',
        # Bytes
        'src_bytes', 'dst_bytes', 'total_bytes', 'bytes_per_second',
        # Paquetes
        'src_pkts', 'dst_pkts', 'total_pkts', 'pkts_per_second',
        # Estadísticas de paquetes
        'pkt_size_avg', 'pkt_size_std', 'pkt_size_max', 'pkt_size_min',
        # Flags TCP
        'syn_count', 'ack_count', 'fin_count', 'rst_count', 'psh_count', 'urg_count',
        # Timing
        'iat_mean', 'iat_std', 'iat_max', 'iat_min',  # Inter-arrival time
        # Ventana TCP
        'init_win_bytes_fwd', 'init_win_bytes_bwd',
        # Actividad
        'active_mean', 'active_std', 'idle_mean', 'idle_std',
        # Ratio
        'down_up_ratio', 'fwd_bwd_ratio',
        # Agregadas por conexión
        'count_same_srv', 'count_same_host', 'serror_rate', 'rerror_rate',
    ]

    def __init__(self,
                 n_features_select: int = 20,
                 method: str = 'hybrid'):  # 'filter', 'pca', 'hybrid'
        self.n_features_select = n_features_select
        self.method = method
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'NetworkFlowReducer':
        """
        Ajusta el reductor para datos de red.
        """
        self.feature_names_ = feature_names
        n_features = X.shape[1]

        # Manejar NaN e infinitos
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Escalar
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        if self.method == 'filter':
            # Solo Mutual Information
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            top_indices = np.argsort(mi_scores)[-self.n_features_select:]

            self.selected_indices_ = top_indices
            self.selected_names_ = [feature_names[i] for i in top_indices]
            self.pca_ = None

        elif self.method == 'pca':
            # Solo PCA
            self.pca_ = PCA(n_components=self.n_features_select)
            self.pca_.fit(X_scaled)
            self.selected_indices_ = None
            self.selected_names_ = [f'PC{i+1}' for i in range(self.n_features_select)]

        else:  # hybrid
            # Seleccionar top 30 por MI, luego PCA a n_features_select
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            n_mi = min(30, n_features)
            top_mi_indices = np.argsort(mi_scores)[-n_mi:]

            self.mi_indices_ = top_mi_indices
            X_mi = X_scaled[:, top_mi_indices]

            # PCA en el subconjunto
            n_pca = min(self.n_features_select, n_mi)
            self.pca_ = PCA(n_components=n_pca)
            self.pca_.fit(X_mi)

            self.selected_names_ = [feature_names[i] for i in top_mi_indices]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica reducción."""
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        if self.method == 'filter':
            return X_scaled[:, self.selected_indices_]
        elif self.method == 'pca':
            return self.pca_.transform(X_scaled)
        else:  # hybrid
            X_mi = X_scaled[:, self.mi_indices_]
            return self.pca_.transform(X_mi)

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importancia de features originales."""
        if self.method == 'pca':
            # Importancia basada en loadings de PCA
            loadings = np.abs(self.pca_.components_).sum(axis=0)
            importance = loadings / loadings.sum()
        elif self.method == 'hybrid':
            # Combinar MI selection + PCA loadings
            loadings = np.abs(self.pca_.components_).sum(axis=0)
            importance = np.zeros(len(self.feature_names_))
            for i, idx in enumerate(self.mi_indices_):
                importance[idx] = loadings[i]
            importance = importance / (importance.sum() + 1e-10)
        else:
            # Solo MI
            importance = np.zeros(len(self.feature_names_))
            importance[self.selected_indices_] = 1.0 / len(self.selected_indices_)

        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)


# Pipeline completo para IDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class IDSPipeline:
    """
    Pipeline completo de detección de intrusiones con reducción.
    """

    def __init__(self, n_features: int = 20):
        self.n_features = n_features

    def build_pipeline(self):
        """Construye pipeline sklearn."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(
                score_func=mutual_info_classif,
                k=self.n_features
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])

    def evaluate_feature_reduction(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_counts: List[int] = None) -> pd.DataFrame:
        """
        Evalúa rendimiento con diferentes números de features.
        """
        if feature_counts is None:
            feature_counts = [5, 10, 15, 20, 30, 40, 50, X.shape[1]]

        results = []

        for n in feature_counts:
            if n > X.shape[1]:
                continue

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=mutual_info_classif, k=n)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])

            scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')

            results.append({
                'n_features': n,
                'f1_mean': scores.mean(),
                'f1_std': scores.std()
            })
            print(f"Features: {n:3d} | F1: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        return pd.DataFrame(results)


# Ejemplo con dataset sintético tipo NSL-KDD
np.random.seed(42)

# Simular 41 features de red
n_samples = 2000
n_features = 41

# Features informativas vs ruido
X = np.random.randn(n_samples, n_features)

# Hacer algunas features informativas
y = ((X[:, 0] > 0) & (X[:, 5] < 0) | (X[:, 10] > 1)).astype(int)

# Añadir ruido
X[:, 20:] = np.random.randn(n_samples, n_features - 20) * 2

feature_names = [f'feature_{i}' for i in range(n_features)]

# Evaluar reducción
print("Evaluación de reducción de features para IDS:")
print("=" * 60)

ids_pipeline = IDSPipeline()
results = ids_pipeline.evaluate_feature_reduction(X, y)
```

### 3.2 Detección de Anomalías en Red con PCA

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class PCANetworkAnomalyDetector:
    """
    Detector de anomalías de red usando PCA.

    Método:
    1. Entrenar PCA en tráfico normal
    2. Para nueva muestra: proyectar y reconstruir
    3. Error de reconstrucción alto = anomalía

    Ventajas:
    - No necesita ejemplos de ataques (semi-supervisado)
    - Interpetable: qué features contribuyen al error
    """

    def __init__(self,
                 n_components: float = 0.95,  # Varianza a retener
                 contamination: float = 0.01):  # % esperado de anomalías
        self.n_components = n_components
        self.contamination = contamination
        self.scaler = StandardScaler()

    def fit(self, X_normal: np.ndarray) -> 'PCANetworkAnomalyDetector':
        """
        Entrena con tráfico NORMAL únicamente.
        """
        # Escalar
        self.scaler.fit(X_normal)
        X_scaled = self.scaler.transform(X_normal)

        # PCA
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_scaled)

        # Calcular errores de reconstrucción en training
        X_reconstructed = self.pca_.inverse_transform(
            self.pca_.transform(X_scaled)
        )
        errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

        # Umbral basado en percentil
        self.threshold_ = np.percentile(errors, 100 * (1 - self.contamination))

        self.train_errors_ = errors

        print(f"PCA: {self.pca_.n_components_} componentes")
        print(f"Varianza explicada: {self.pca_.explained_variance_ratio_.sum():.2%}")
        print(f"Umbral de anomalía: {self.threshold_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice anomalías.

        Returns:
            labels: 0=normal, 1=anomalía
            scores: Error de reconstrucción
        """
        X_scaled = self.scaler.transform(X)

        # Reconstruir
        X_reconstructed = self.pca_.inverse_transform(
            self.pca_.transform(X_scaled)
        )

        # Error de reconstrucción
        errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

        # Clasificar
        labels = (errors > self.threshold_).astype(int)

        return labels, errors

    def get_anomaly_contribution(self, X: np.ndarray,
                                  feature_names: list) -> pd.DataFrame:
        """
        Para cada anomalía, identifica qué features contribuyen más al error.
        Útil para análisis forense.
        """
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.pca_.inverse_transform(
            self.pca_.transform(X_scaled)
        )

        # Error por feature
        feature_errors = (X_scaled - X_reconstructed) ** 2

        contributions = []
        for i in range(len(X)):
            total_error = feature_errors[i].sum()
            if total_error > self.threshold_:
                # Top features que contribuyen al error
                top_indices = np.argsort(feature_errors[i])[-5:][::-1]
                top_features = [(feature_names[j], feature_errors[i, j])
                               for j in top_indices]
                contributions.append({
                    'sample_idx': i,
                    'total_error': total_error,
                    'top_features': top_features
                })

        return pd.DataFrame(contributions)


# Ejemplo
np.random.seed(42)

# Tráfico normal
n_normal = 1000
X_normal = np.random.randn(n_normal, 30)

# Tráfico con anomalías (diferente distribución)
n_test = 200
X_test_normal = np.random.randn(n_test - 20, 30)
X_test_anomaly = np.random.randn(20, 30) * 3 + 2  # Distribución diferente
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.array([0] * (n_test - 20) + [1] * 20)

# Entrenar y evaluar
detector = PCANetworkAnomalyDetector(n_components=0.95, contamination=0.05)
detector.fit(X_normal)

predictions, scores = detector.predict(X_test)

# Métricas
from sklearn.metrics import classification_report, roc_auc_score

print("\nResultados:")
print(classification_report(y_test, predictions, target_names=['Normal', 'Anomalía']))
print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")
```

---

## 4. Análisis de Logs

### 4.1 Reducción para Log Events

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from typing import List, Tuple

class LogFeatureReducer:
    """
    Reducción de dimensionalidad para análisis de logs.

    Logs tienen alta dimensionalidad:
    - TF-IDF de tokens: 10,000+ dimensiones
    - Event types: 500+ tipos
    - Features temporales, IP, usuarios, etc.

    Técnicas:
    - LSA (TruncatedSVD) para reducir TF-IDF
    - LDA para topic modeling
    - Autoencoders para representación aprendida
    """

    def __init__(self,
                 method: str = 'lsa',  # 'lsa', 'lda'
                 n_components: int = 50,
                 max_features: int = 5000):
        self.method = method
        self.n_components = n_components
        self.max_features = max_features

    def fit_transform_text(self, log_messages: List[str]) -> Tuple[np.ndarray, object]:
        """
        Transforma logs de texto a representación reducida.

        Args:
            log_messages: Lista de mensajes de log

        Returns:
            X_reduced: Matriz reducida
            vectorizer: Para transformar nuevos logs
        """
        # TF-IDF
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z_0-9]+\b'  # Solo palabras válidas
        )

        X_tfidf = self.vectorizer_.fit_transform(log_messages)

        print(f"TF-IDF shape: {X_tfidf.shape}")

        # Reducir
        if self.method == 'lsa':
            self.reducer_ = TruncatedSVD(
                n_components=self.n_components,
                random_state=42
            )
            X_reduced = self.reducer_.fit_transform(X_tfidf)

            explained_var = self.reducer_.explained_variance_ratio_.sum()
            print(f"LSA: {self.n_components} componentes, "
                  f"{explained_var:.2%} varianza explicada")

        else:  # LDA
            self.reducer_ = LatentDirichletAllocation(
                n_components=self.n_components,
                random_state=42,
                n_jobs=-1
            )
            X_reduced = self.reducer_.fit_transform(X_tfidf)

            print(f"LDA: {self.n_components} topics")

        return X_reduced

    def transform(self, log_messages: List[str]) -> np.ndarray:
        """Transforma nuevos logs."""
        X_tfidf = self.vectorizer_.transform(log_messages)
        return self.reducer_.transform(X_tfidf)

    def get_top_terms_per_component(self, n_terms: int = 10) -> pd.DataFrame:
        """
        Muestra los términos más importantes por componente/topic.
        Útil para interpretar qué captura cada dimensión.
        """
        feature_names = self.vectorizer_.get_feature_names_out()

        if self.method == 'lsa':
            components = self.reducer_.components_
        else:
            components = self.reducer_.components_

        rows = []
        for i, component in enumerate(components):
            top_indices = component.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[j] for j in top_indices]
            top_weights = [component[j] for j in top_indices]

            rows.append({
                'component': i,
                'top_terms': ', '.join(top_terms[:5]),
                'all_terms': list(zip(top_terms, top_weights))
            })

        return pd.DataFrame(rows)


# Ejemplo: Logs de seguridad
sample_logs = [
    "Failed password for root from 192.168.1.100 port 22 ssh2",
    "Failed password for admin from 10.0.0.50 port 22 ssh2",
    "Accepted password for user1 from 192.168.1.10 port 22 ssh2",
    "Connection closed by 192.168.1.100 port 22",
    "Invalid user test from 192.168.1.100 port 22",
    "Failed password for invalid user admin from 10.0.0.99",
    "Accepted publickey for deploy from 10.0.0.1 port 22",
    "session opened for user root by (uid=0)",
    "session closed for user root",
    "sudo: user1 : TTY=pts/0 ; PWD=/home/user1 ; USER=root ; COMMAND=/bin/bash",
    "PAM unix authentication failure for root",
    "TCP connection attempt blocked from 203.0.113.50 to port 445",
    "Firewall: DROP IN=eth0 SRC=192.168.1.200 DST=10.0.0.5 PROTO=TCP DPT=3389",
    "HTTP 403 Forbidden GET /admin/config.php from 192.168.1.150",
    "SQL injection attempt detected in parameter id",
    "XSS attack blocked in user input field",
] * 100  # Repetir para tener suficientes muestras

# Reducir
log_reducer = LogFeatureReducer(method='lsa', n_components=10)
X_reduced = log_reducer.fit_transform_text(sample_logs)

print(f"\nShape reducido: {X_reduced.shape}")

# Ver qué captura cada componente
print("\nInterpretación de componentes:")
terms_df = log_reducer.get_top_terms_per_component()
for _, row in terms_df.iterrows():
    print(f"  Componente {row['component']}: {row['top_terms']}")
```

### 4.2 Detección de Anomalías en Logs con Autoencoders

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class LogAutoencoder(nn.Module):
    """
    Autoencoder para detección de anomalías en logs.

    Arquitectura:
    Input (n_features) -> Encoder -> Latent (n_latent) -> Decoder -> Output (n_features)

    Anomalías: Alto error de reconstrucción
    """

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_latent(self, x):
        """Obtiene representación latente (reducida)."""
        return self.encoder(x)


class LogAnomalyDetector:
    """
    Detector de anomalías en logs usando Autoencoder.
    """

    def __init__(self,
                 latent_dim: int = 16,
                 epochs: int = 50,
                 batch_size: int = 64,
                 contamination: float = 0.05):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X_normal: np.ndarray) -> 'LogAnomalyDetector':
        """Entrena con logs normales."""
        # Escalar
        X_scaled = self.scaler.fit_transform(X_normal)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Dataset
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Modelo
        self.model_ = LogAutoencoder(X_normal.shape[1], self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Entrenar
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]

                reconstructed = self.model_(x)
                loss = criterion(reconstructed, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")

        # Calcular umbral
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        self.threshold_ = np.percentile(errors, 100 * (1 - self.contamination))
        print(f"Umbral de anomalía: {self.threshold_:.6f}")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice anomalías."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        labels = (errors > self.threshold_).astype(int)
        return labels, errors

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Obtiene representación reducida para visualización."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            latent = self.model_.get_latent(X_tensor).cpu().numpy()

        return latent


# Ejemplo de uso
np.random.seed(42)

# Simular features de logs (después de TF-IDF + reducción inicial)
n_normal = 1000
n_anomaly = 50
n_features = 50

# Logs normales: distribución típica
X_normal = np.random.randn(n_normal, n_features) * 0.5

# Logs anómalos: patrón diferente (ej: brute force, exfiltration)
X_anomaly = np.random.randn(n_anomaly, n_features) * 2 + 1

# Test set
X_test = np.vstack([
    np.random.randn(100, n_features) * 0.5,  # Normal
    X_anomaly[:20]  # Anomalías
])
y_test = np.array([0] * 100 + [1] * 20)

# Entrenar detector
detector = LogAnomalyDetector(latent_dim=8, epochs=30)
detector.fit(X_normal)

# Evaluar
predictions, scores = detector.predict(X_test)

from sklearn.metrics import classification_report, roc_auc_score
print("\nResultados:")
print(classification_report(y_test, predictions, target_names=['Normal', 'Anomalía']))
print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")

# Visualizar espacio latente
latent = detector.get_latent_representation(X_test)
print(f"\nDimensionalidad reducida: {n_features} -> {latent.shape[1]}")
```

---

## 5. Casos de Uso Específicos

### 5.1 Threat Hunting: Reducción para Investigación

```python
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from typing import List, Dict

class ThreatHuntingPipeline:
    """
    Pipeline de Threat Hunting con reducción de dimensionalidad.

    Flujo:
    1. Recolectar eventos de múltiples fuentes
    2. Extraer features
    3. Reducir con UMAP para visualización
    4. Clustering para identificar comportamientos anómalos
    5. Drill-down en clusters sospechosos
    """

    def __init__(self,
                 n_neighbors: int = 15,
                 min_cluster_size: int = 10):
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size

    def analyze(self, X: np.ndarray,
                metadata: pd.DataFrame) -> Dict:
        """
        Analiza eventos para threat hunting.

        Args:
            X: Features de eventos
            metadata: Info adicional (timestamp, user, host, etc.)

        Returns:
            Resultados del análisis con clusters y anomalías
        """
        # 1. Reducir a 2D para visualización
        self.reducer_ = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=0.1,
            random_state=42
        )
        embedding_2d = self.reducer_.fit_transform(X)

        # 2. Clustering
        self.clusterer_ = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5
        )
        clusters = self.clusterer_.fit_predict(embedding_2d)

        # 3. Identificar outliers (cluster -1)
        outlier_mask = clusters == -1
        n_outliers = outlier_mask.sum()

        # 4. Analizar clusters
        cluster_stats = []
        unique_clusters = [c for c in np.unique(clusters) if c != -1]

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_meta = metadata[mask]

            stats = {
                'cluster_id': cluster_id,
                'size': mask.sum(),
                'unique_users': cluster_meta['user'].nunique() if 'user' in cluster_meta else 0,
                'unique_hosts': cluster_meta['host'].nunique() if 'host' in cluster_meta else 0,
                'time_span': None
            }

            if 'timestamp' in cluster_meta:
                stats['time_span'] = (
                    cluster_meta['timestamp'].max() -
                    cluster_meta['timestamp'].min()
                ).total_seconds() / 3600  # horas

            cluster_stats.append(stats)

        # 5. Scoring de sospecha
        # Clusters pequeños con pocos usuarios/hosts son más sospechosos
        for stats in cluster_stats:
            suspicion_score = 0
            if stats['size'] < 20:
                suspicion_score += 2
            if stats['unique_users'] == 1:
                suspicion_score += 3
            if stats['unique_hosts'] == 1:
                suspicion_score += 2
            stats['suspicion_score'] = suspicion_score

        return {
            'embedding': embedding_2d,
            'clusters': clusters,
            'n_outliers': n_outliers,
            'outlier_indices': np.where(outlier_mask)[0],
            'cluster_stats': sorted(cluster_stats,
                                   key=lambda x: x['suspicion_score'],
                                   reverse=True)
        }

    def get_suspicious_events(self,
                             results: Dict,
                             metadata: pd.DataFrame,
                             top_n: int = 10) -> pd.DataFrame:
        """
        Retorna los eventos más sospechosos para investigación.
        """
        suspicious_indices = []

        # 1. Todos los outliers
        suspicious_indices.extend(results['outlier_indices'].tolist())

        # 2. Eventos de clusters sospechosos
        for stats in results['cluster_stats'][:3]:  # Top 3 clusters sospechosos
            if stats['suspicion_score'] >= 3:
                cluster_mask = results['clusters'] == stats['cluster_id']
                suspicious_indices.extend(np.where(cluster_mask)[0].tolist())

        suspicious_indices = list(set(suspicious_indices))[:top_n]

        return metadata.iloc[suspicious_indices]


# Ejemplo
np.random.seed(42)

# Simular eventos de seguridad
n_events = 500

# Features de eventos (reducidas previamente)
X = np.random.randn(n_events, 20)

# Añadir un cluster de comportamiento anómalo (posible threat)
X[480:500] = np.random.randn(20, 20) * 0.3 + np.array([2] * 20)

# Metadata
metadata = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n_events, freq='1min'),
    'user': np.random.choice(['user1', 'user2', 'user3', 'admin', 'service'], n_events),
    'host': np.random.choice(['srv1', 'srv2', 'ws1', 'ws2', 'ws3'], n_events),
    'event_type': np.random.choice(['login', 'file_access', 'network', 'process'], n_events)
})

# El cluster anómalo es de un solo usuario en un solo host
metadata.loc[480:499, 'user'] = 'suspicious_user'
metadata.loc[480:499, 'host'] = 'compromised_host'

# Analizar
pipeline = ThreatHuntingPipeline()
results = pipeline.analyze(X, metadata)

print("=== Threat Hunting Analysis ===")
print(f"Total eventos: {n_events}")
print(f"Outliers detectados: {results['n_outliers']}")
print(f"Clusters encontrados: {len(results['cluster_stats'])}")

print("\n--- Clusters Sospechosos ---")
for stats in results['cluster_stats'][:5]:
    print(f"Cluster {stats['cluster_id']}: "
          f"size={stats['size']}, "
          f"users={stats['unique_users']}, "
          f"hosts={stats['unique_hosts']}, "
          f"suspicion={stats['suspicion_score']}")

print("\n--- Eventos para Investigar ---")
suspicious = pipeline.get_suspicious_events(results, metadata)
print(suspicious[['timestamp', 'user', 'host', 'event_type']])
```

---

## 6. Best Practices para Seguridad

### 6.1 Consideraciones de Producción

```python
"""
BEST PRACTICES: Reducción de Dimensionalidad en Seguridad
"""

# 1. CONSISTENCIA ENTRE TRAINING Y INFERENCE
# -----------------------------------------
# SIEMPRE guardar y reutilizar los transformadores

import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def save_reduction_pipeline(pipeline, path: str):
    """Guarda pipeline completo para producción."""
    joblib.dump(pipeline, path)

def load_reduction_pipeline(path: str):
    """Carga pipeline para inference."""
    return joblib.load(path)


# 2. MANEJO DE FEATURES FALTANTES
# -------------------------------
# En producción, algunas features pueden no estar disponibles

import numpy as np

def handle_missing_features(X: np.ndarray,
                           expected_features: int,
                           feature_order: list,
                           available_features: list) -> np.ndarray:
    """
    Maneja features faltantes rellenando con valores por defecto.
    """
    X_complete = np.zeros((X.shape[0], expected_features))

    for i, feat in enumerate(available_features):
        if feat in feature_order:
            idx = feature_order.index(feat)
            X_complete[:, idx] = X[:, i]

    return X_complete


# 3. MONITOREO DE DRIFT
# ---------------------
# La distribución de features puede cambiar (concept drift)

from scipy import stats

def detect_feature_drift(X_reference: np.ndarray,
                        X_new: np.ndarray,
                        threshold: float = 0.05) -> dict:
    """
    Detecta drift en la distribución de features.
    Usa Kolmogorov-Smirnov test.
    """
    drifted_features = []

    for i in range(X_reference.shape[1]):
        statistic, p_value = stats.ks_2samp(
            X_reference[:, i],
            X_new[:, i]
        )

        if p_value < threshold:
            drifted_features.append({
                'feature_idx': i,
                'ks_statistic': statistic,
                'p_value': p_value
            })

    return {
        'drift_detected': len(drifted_features) > 0,
        'drifted_features': drifted_features,
        'drift_ratio': len(drifted_features) / X_reference.shape[1]
    }


# 4. LATENCIA EN TIEMPO REAL
# --------------------------
# Para IDS/IPS, la latencia es crítica

import time
from functools import wraps

def measure_latency(func):
    """Decorator para medir latencia de inference."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        print(f"Latency: {latency_ms:.2f}ms")
        return result
    return wrapper


class FastReducer:
    """
    Reductor optimizado para baja latencia.
    - Pre-computa lo posible
    - Usa operaciones vectorizadas
    - Evita copias innecesarias
    """

    def __init__(self, pca_components: np.ndarray,
                 mean: np.ndarray,
                 scale: np.ndarray):
        # Pre-computar: components ya escalados
        self.projection_matrix = pca_components / scale
        self.mean = mean
        self.scale = scale

    @measure_latency
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform optimizado."""
        # Una sola operación matricial
        return (X - self.mean) @ self.projection_matrix.T


# 5. INTERPRETABILIDAD
# --------------------
# En seguridad, necesitamos explicar las decisiones

def explain_pca_anomaly(X_sample: np.ndarray,
                       pca,
                       scaler,
                       feature_names: list,
                       n_top: int = 5) -> dict:
    """
    Explica por qué una muestra es anómala según PCA.
    """
    # Reconstruir
    X_scaled = scaler.transform(X_sample.reshape(1, -1))
    X_reconstructed = pca.inverse_transform(pca.transform(X_scaled))

    # Error por feature
    errors = (X_scaled - X_reconstructed).ravel() ** 2

    # Top features con mayor error
    top_indices = np.argsort(errors)[-n_top:][::-1]

    explanation = {
        'total_error': errors.sum(),
        'top_features': [
            {
                'name': feature_names[i],
                'error_contribution': errors[i],
                'original_value': X_sample[i],
                'reconstructed_value': scaler.inverse_transform(X_reconstructed)[0, i]
            }
            for i in top_indices
        ]
    }

    return explanation
```

---

## 7. Resumen

### Matriz de Decisión

| Caso de Uso | Técnica Recomendada | Razón |
|-------------|---------------------|-------|
| Clasificación de Malware | Feature Selection (MI + RF) | Interpretabilidad, features costosas |
| Visualización de Familias | UMAP | Preserva clusters, rápido |
| IDS en tiempo real | PCA pre-entrenado | Baja latencia |
| Detección de anomalías | Autoencoder / PCA | Semi-supervisado |
| Análisis de logs | LSA / LDA | Maneja texto sparse |
| Threat Hunting | UMAP + HDBSCAN | Exploración interactiva |

### Checklist de Implementación

```
□ Feature Selection antes de Extraction cuando interpretabilidad importa
□ Pipeline guardado y versionado
□ Manejo de features faltantes en producción
□ Monitoreo de drift configurado
□ Latencia medida y aceptable
□ Explicabilidad para analistas
□ Tests de regresión con datos reales
□ Documentación de features seleccionadas
```
