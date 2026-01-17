# Audio Multimodal: Speech, Text y Vision

## Introduccion

Los sistemas multimodales de audio integran audio/speech con texto y vision para tareas como: speech recognition, speaker verification, audio-visual speech recognition, multimodal sentiment analysis, y deteccion de deepfakes. Esta combinacion es crucial para aplicaciones de seguridad.

```
Landscape de Audio Multimodal:

┌────────────────────────────────────────────────────────────────────┐
│                     AUDIO MULTIMODAL TASKS                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  AUDIO + TEXT                                                      │
│  ├── Speech Recognition (ASR)                                      │
│  ├── Text-to-Speech (TTS)                                          │
│  ├── Audio Captioning                                              │
│  └── Spoken Language Understanding                                 │
│                                                                    │
│  AUDIO + VISION                                                    │
│  ├── Audio-Visual Speech Recognition (lip reading + audio)        │
│  ├── Speaker Diarization with Face                                 │
│  ├── Audio-Visual Event Detection                                  │
│  └── Video Understanding with Audio                                │
│                                                                    │
│  AUDIO + TEXT + VISION                                             │
│  ├── Multimodal Sentiment Analysis                                 │
│  ├── Deepfake Detection (audio-visual inconsistency)              │
│  ├── Video Captioning with Audio                                   │
│  └── Multimodal Meeting Analysis                                   │
│                                                                    │
│  SECURITY APPLICATIONS                                             │
│  ├── Voice Authentication / Speaker Verification                   │
│  ├── Audio Deepfake Detection                                      │
│  ├── Synthetic Voice Detection                                     │
│  └── Fraud Detection in Call Centers                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Fundamentos de Audio Processing

### Representaciones de Audio

```
Representaciones de Audio para Deep Learning:

1. WAVEFORM (Raw Audio)
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │    Amplitude                                                   │
   │       ↑                                                        │
   │       │    /\      /\                                          │
   │       │   /  \    /  \    /\                                   │
   │     0 ├──/────\──/────\──/──\──────────────────→ Time         │
   │       │        \/      \/    \  /\                             │
   │       │                       \/  \/                           │
   │                                                                │
   │    Sample rate: 16000 Hz (16K muestras/segundo)               │
   │    1 segundo audio = tensor de 16000 valores                  │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘

2. SPECTROGRAM (Time-Frequency)
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │    Frequency                                                   │
   │       ↑                                                        │
   │   8kHz│ ░░░░████░░░░░░░░████░░░░                              │
   │       │ ░░████████░░░░████████░░                               │
   │   4kHz│ ████████████░░████████████                             │
   │       │ ████████████████████████████                           │
   │   0Hz └──────────────────────────────→ Time                   │
   │                                                                │
   │    STFT: Short-Time Fourier Transform                         │
   │    Shape: (num_freq_bins, num_time_frames)                    │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘

3. MEL SPECTROGRAM
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │    Escala Mel imita percepcion humana:                        │
   │    - Mas resolucion en frecuencias bajas                      │
   │    - Menos en altas (logaritmico)                             │
   │                                                                │
   │    Mel(f) = 2595 * log10(1 + f/700)                           │
   │                                                                │
   │    Shape tipico: (80 mel bins, time_frames)                   │
   │    n_fft=400, hop_length=160 para 16kHz                       │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘

4. MFCC (Mel-Frequency Cepstral Coefficients)
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │    Pipeline:                                                   │
   │    Audio → STFT → Mel Filter Bank → Log → DCT → MFCC         │
   │                                                                │
   │    Comprime informacion espectral a ~13-40 coeficientes       │
   │    Util para speaker recognition tradicional                  │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Dict, List, Optional, Tuple
import numpy as np


class AudioFeatureExtractor:
    """
    Extractor de features de audio para diferentes representaciones.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mfcc: int = 40
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm='slaney',
            mel_scale='slaney'
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )

        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def load_audio(
        self,
        audio_path: str,
        target_sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Carga audio y resamplea si es necesario.

        Returns:
            (channels, samples) tensor
        """
        waveform, sr = torchaudio.load(audio_path)

        target_sr = target_sr or self.sample_rate

        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Convertir a mono si es stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def extract_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extrae mel spectrogram.

        Args:
            waveform: (1, samples) audio tensor

        Returns:
            (1, n_mels, time_frames) mel spectrogram en dB
        """
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        if normalize:
            # Normalizar a media 0, std 1
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db

    def extract_mfcc(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Extrae MFCC features.

        Returns:
            (1, n_mfcc, time_frames) MFCC tensor
        """
        return self.mfcc_transform(waveform)

    def extract_all_features(
        self,
        audio_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Extrae todas las representaciones.
        """
        waveform = self.load_audio(audio_path)

        return {
            'waveform': waveform,
            'mel_spectrogram': self.extract_mel_spectrogram(waveform),
            'mfcc': self.extract_mfcc(waveform),
            'duration_seconds': waveform.shape[1] / self.sample_rate
        }
```

## Whisper: Speech Recognition

Whisper de OpenAI es un modelo robusto de speech recognition entrenado en 680K horas de audio multilingue.

```
Whisper Architecture:

┌────────────────────────────────────────────────────────────────────┐
│                           WHISPER                                   │
│                                                                    │
│   Input: 30s audio (padded/truncated)                              │
│          ↓                                                         │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │              Log-Mel Spectrogram (80 bins)                  │  │
│   │              Shape: (80, 3000) for 30s                      │  │
│   └────────────────────────────────────────────────────────────┘  │
│          ↓                                                         │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                   Audio Encoder                             │  │
│   │  ┌──────────────────────────────────────────────────────┐  │  │
│   │  │  Conv1D (80 → 512/768/1024/1280) + GELU              │  │  │
│   │  │  Conv1D (stride=2 for downsampling)                  │  │  │
│   │  │  + Sinusoidal Positional Encoding                    │  │  │
│   │  │                     ↓                                 │  │  │
│   │  │  Transformer Encoder Blocks × N                      │  │  │
│   │  │  (N = 4/6/12/24/32 depending on model size)          │  │  │
│   │  └──────────────────────────────────────────────────────┘  │  │
│   └────────────────────────────────────────────────────────────┘  │
│          ↓                                                         │
│   Audio Features: (1500, dim)                                      │
│          ↓                                                         │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                   Text Decoder                              │  │
│   │  ┌──────────────────────────────────────────────────────┐  │  │
│   │  │  Token Embedding + Positional Embedding              │  │  │
│   │  │                     ↓                                 │  │  │
│   │  │  Transformer Decoder Blocks × N                      │  │  │
│   │  │  (Causal Self-Attention + Cross-Attention to Audio)  │  │  │
│   │  │                     ↓                                 │  │  │
│   │  │  Linear → Softmax → Next Token                       │  │  │
│   │  └──────────────────────────────────────────────────────┘  │  │
│   └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   Output: "<|startoftranscript|><|en|><|transcribe|>Hello..."    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Modelo Sizes:
┌────────────┬─────────────┬─────────┬───────────────┐
│ Model      │ Parameters  │ Layers  │ Multilingual  │
├────────────┼─────────────┼─────────┼───────────────┤
│ tiny       │ 39M         │ 4       │ Yes           │
│ base       │ 74M         │ 6       │ Yes           │
│ small      │ 244M        │ 12      │ Yes           │
│ medium     │ 769M        │ 24      │ Yes           │
│ large      │ 1550M       │ 32      │ Yes           │
│ large-v2   │ 1550M       │ 32      │ Yes (better)  │
│ large-v3   │ 1550M       │ 32      │ Yes (best)    │
└────────────┴─────────────┴─────────┴───────────────┘
```

```python
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Union
import torch


class WhisperWrapper:
    """
    Wrapper para Whisper con diferentes backends.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_transformers: bool = False
    ):
        """
        Args:
            model_name: tiny, base, small, medium, large, large-v2, large-v3
            device: cuda o cpu
            use_transformers: usar HuggingFace transformers (mas flexible)
        """
        self.device = device
        self.use_transformers = use_transformers

        if use_transformers:
            model_id = f"openai/whisper-{model_name}"
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
            self.model.to(device)
        else:
            self.model = whisper.load_model(model_name, device=device)
            self.processor = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",  # "transcribe" or "translate"
        return_timestamps: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe audio a texto.

        Args:
            audio_path: Path al archivo de audio
            language: Codigo de idioma (None para auto-detect)
            task: "transcribe" (mismo idioma) o "translate" (a ingles)
            return_timestamps: Incluir timestamps por segmento

        Returns:
            Dict con transcripcion y metadata
        """
        if self.use_transformers:
            return self._transcribe_transformers(
                audio_path, language, task, return_timestamps
            )
        else:
            return self._transcribe_openai(
                audio_path, language, task, return_timestamps
            )

    def _transcribe_openai(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        return_timestamps: bool
    ) -> Dict:
        """Transcripcion usando libreria openai-whisper."""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=return_timestamps
        )

        output = {
            'text': result['text'],
            'language': result['language'],
            'segments': result['segments'] if return_timestamps else None
        }

        return output

    def _transcribe_transformers(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        return_timestamps: bool
    ) -> Dict:
        """Transcripcion usando HuggingFace transformers."""
        # Cargar audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze().numpy()

        # Procesar
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # Configurar generacion
        generate_kwargs = {
            'task': task,
            'return_timestamps': return_timestamps
        }
        if language:
            generate_kwargs['language'] = language

        # Generar
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                **generate_kwargs
            )

        # Decodificar
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return {
            'text': transcription,
            'language': language or 'auto'
        }

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        audio_path: str
    ) -> torch.Tensor:
        """
        Extrae embeddings del encoder de Whisper.
        Utiles para tareas downstream.

        Returns:
            (1, seq_len, hidden_dim) embeddings
        """
        if not self.use_transformers:
            # Cargar y procesar audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.device)

            # Encode
            with torch.no_grad():
                embeddings = self.model.encoder(mel.unsqueeze(0))

            return embeddings
        else:
            # Cargar audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze().numpy()

            # Procesar
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)

            # Get encoder outputs
            encoder_outputs = self.model.model.encoder(
                inputs.input_features
            )

            return encoder_outputs.last_hidden_state


class WhisperSecurityAnalyzer:
    """
    Analisis de seguridad usando Whisper.
    """

    def __init__(self, whisper_wrapper: WhisperWrapper):
        self.whisper = whisper_wrapper

        # Keywords sospechosas
        self.suspicious_keywords = {
            'phishing': ['password', 'credit card', 'social security', 'verify your account', 'urgent action'],
            'scam': ['lottery', 'prize', 'wire transfer', 'western union', 'gift card'],
            'threat': ['bomb', 'attack', 'kill', 'weapon', 'explosive']
        }

    def analyze_call_recording(
        self,
        audio_path: str
    ) -> Dict[str, any]:
        """
        Analiza grabacion de llamada para seguridad.
        Detecta intentos de phishing, scam, etc.
        """
        # Transcribir
        result = self.whisper.transcribe(
            audio_path,
            return_timestamps=True
        )

        text = result['text'].lower()

        # Detectar keywords sospechosas
        detections = {}
        for category, keywords in self.suspicious_keywords.items():
            found = [kw for kw in keywords if kw in text]
            if found:
                detections[category] = found

        # Calcular risk score
        risk_score = len(detections) * 0.3
        risk_score = min(risk_score, 1.0)

        risk_level = "low"
        if risk_score > 0.3:
            risk_level = "medium"
        if risk_score > 0.6:
            risk_level = "high"

        return {
            'transcription': result['text'],
            'language': result['language'],
            'detections': detections,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'segments': result.get('segments')
        }
```

## Audio Embeddings y Speaker Verification

```
Speaker Verification Pipeline:

┌────────────────────────────────────────────────────────────────────┐
│                   SPEAKER VERIFICATION                              │
│                                                                    │
│   Enrollment Phase:                                                │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │  Voice       │    │   Speaker    │    │   Store      │        │
│   │  Samples     │───▶│   Encoder    │───▶│   Embedding  │        │
│   │  (User X)    │    │              │    │   (voiceprint)│        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                    │
│   Verification Phase:                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │  New Voice   │    │   Speaker    │    │   Compare    │        │
│   │  Sample      │───▶│   Encoder    │───▶│   with       │───▶ Accept/Reject
│   │              │    │              │    │   Stored     │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                    │
│   Similarity: cosine(emb_new, emb_stored) > threshold              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Modelos de Speaker Embedding:
┌─────────────────────┬──────────────────────────────────────────────┐
│ Modelo              │ Descripcion                                  │
├─────────────────────┼──────────────────────────────────────────────┤
│ X-Vector (TDNN)     │ Time-Delay NN, clasico, robusto             │
│ ECAPA-TDNN          │ SOTA speaker verification                    │
│ Wav2Vec2-XLSR       │ Self-supervised, multilingue                 │
│ WavLM               │ Microsoft, excelente para speaker           │
│ TitaNet             │ NVIDIA, alta precision                       │
│ SpeechBrain models  │ Framework completo speaker recognition       │
└─────────────────────┴──────────────────────────────────────────────┘
```

```python
from speechbrain.inference.speaker import EncoderClassifier
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class SpeakerVerificationSystem:
    """
    Sistema de verificacion de speaker usando SpeechBrain.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Modelo de SpeechBrain Hub
                - "speechbrain/spkrec-ecapa-voxceleb" (ECAPA-TDNN)
                - "speechbrain/spkrec-xvect-voxceleb" (X-Vector)
        """
        self.device = device

        # Cargar modelo
        self.encoder = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device}
        )

        # Database de voiceprints
        self.voiceprints: Dict[str, torch.Tensor] = {}

    def extract_embedding(
        self,
        audio_path: str
    ) -> torch.Tensor:
        """
        Extrae embedding de speaker de un audio.

        Returns:
            (1, 192) embedding tensor (ECAPA-TDNN)
        """
        embedding = self.encoder.encode_batch(
            self.encoder.load_audio(audio_path)
        )
        return embedding.squeeze(0)

    def enroll_speaker(
        self,
        speaker_id: str,
        audio_paths: List[str]
    ) -> None:
        """
        Registra un speaker con multiples muestras de voz.
        Promedia embeddings para mejor representacion.
        """
        embeddings = []
        for path in audio_paths:
            emb = self.extract_embedding(path)
            embeddings.append(emb)

        # Promediar embeddings
        avg_embedding = torch.stack(embeddings).mean(dim=0)

        # Normalizar
        avg_embedding = F.normalize(avg_embedding, dim=-1)

        self.voiceprints[speaker_id] = avg_embedding

    def verify_speaker(
        self,
        audio_path: str,
        claimed_speaker_id: str,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Verifica si el audio pertenece al speaker reclamado.

        Returns:
            Dict con decision, score, y confidence
        """
        if claimed_speaker_id not in self.voiceprints:
            return {
                'verified': False,
                'error': f"Speaker {claimed_speaker_id} not enrolled"
            }

        # Extraer embedding
        test_embedding = self.extract_embedding(audio_path)
        test_embedding = F.normalize(test_embedding, dim=-1)

        # Comparar con voiceprint almacenado
        stored_embedding = self.voiceprints[claimed_speaker_id]

        # Similitud coseno
        similarity = F.cosine_similarity(
            test_embedding.unsqueeze(0),
            stored_embedding.unsqueeze(0)
        ).item()

        # Decision
        verified = similarity >= threshold

        return {
            'verified': verified,
            'similarity_score': similarity,
            'threshold': threshold,
            'claimed_speaker': claimed_speaker_id,
            'confidence': abs(similarity - threshold)
        }

    def identify_speaker(
        self,
        audio_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Identifica el speaker entre los enrollados.

        Returns:
            Lista de (speaker_id, similarity) ordenada
        """
        if not self.voiceprints:
            return []

        # Extraer embedding
        test_embedding = self.extract_embedding(audio_path)
        test_embedding = F.normalize(test_embedding, dim=-1)

        # Comparar con todos los voiceprints
        scores = []
        for speaker_id, voiceprint in self.voiceprints.items():
            similarity = F.cosine_similarity(
                test_embedding.unsqueeze(0),
                voiceprint.unsqueeze(0)
            ).item()
            scores.append((speaker_id, similarity))

        # Ordenar por score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class VoiceAuthenticator:
    """
    Sistema de autenticacion por voz para seguridad.
    """

    def __init__(
        self,
        speaker_system: SpeakerVerificationSystem,
        liveness_threshold: float = 0.7,
        verification_threshold: float = 0.6
    ):
        self.speaker = speaker_system
        self.liveness_threshold = liveness_threshold
        self.verification_threshold = verification_threshold

    def authenticate(
        self,
        audio_path: str,
        claimed_identity: str,
        passphrase: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Autenticacion completa con verificacion de speaker.

        Args:
            audio_path: Audio de la persona intentando autenticar
            claimed_identity: ID del usuario que dice ser
            passphrase: Frase secreta esperada (opcional)
        """
        result = {
            'authenticated': False,
            'checks_passed': [],
            'checks_failed': [],
            'details': {}
        }

        # 1. Verificacion de speaker
        speaker_result = self.speaker.verify_speaker(
            audio_path,
            claimed_identity,
            threshold=self.verification_threshold
        )

        result['details']['speaker_verification'] = speaker_result

        if speaker_result.get('verified'):
            result['checks_passed'].append('speaker_verification')
        else:
            result['checks_failed'].append('speaker_verification')

        # 2. Verificacion de passphrase (si se proporciona)
        if passphrase:
            # Usar Whisper para transcribir
            from . import WhisperWrapper  # Importar localmente

            whisper = WhisperWrapper(model_name="base")
            transcription = whisper.transcribe(audio_path)

            spoken_text = transcription['text'].lower().strip()
            expected_text = passphrase.lower().strip()

            # Comparacion simple (en produccion: usar distancia de edicion)
            passphrase_match = spoken_text == expected_text

            result['details']['passphrase'] = {
                'expected': expected_text,
                'spoken': spoken_text,
                'match': passphrase_match
            }

            if passphrase_match:
                result['checks_passed'].append('passphrase')
            else:
                result['checks_failed'].append('passphrase')

        # Decision final
        required_checks = ['speaker_verification']
        if passphrase:
            required_checks.append('passphrase')

        result['authenticated'] = all(
            check in result['checks_passed'] for check in required_checks
        )

        return result
```

## Audio Deepfake Detection

```
Audio Deepfake Detection:

Tipos de Audio Deepfakes:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  1. Text-to-Speech (TTS) Synthesis                                │
│     - Genera voz sintetica desde texto                            │
│     - VITS, Tacotron, FastSpeech                                  │
│                                                                    │
│  2. Voice Conversion (VC)                                          │
│     - Convierte voz de A a sonar como B                           │
│     - So-VITS-SVC, RVC                                            │
│                                                                    │
│  3. Voice Cloning                                                  │
│     - Clona voz con pocas muestras                                │
│     - VALL-E, Bark, Tortoise                                      │
│                                                                    │
│  4. Adversarial Audio                                              │
│     - Audio modificado para enganar sistemas                       │
│     - Pequenas perturbaciones imperceptibles                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Arquitectura de Deteccion:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   Audio Input                                                      │
│        │                                                           │
│        ▼                                                           │
│   ┌──────────────────────────────────────────┐                    │
│   │        Feature Extraction                 │                    │
│   │  - Mel Spectrogram                        │                    │
│   │  - Linear Spectrogram                     │                    │
│   │  - Raw Waveform (RawNet)                  │                    │
│   │  - CQT (Constant-Q Transform)             │                    │
│   └──────────────────────────────────────────┘                    │
│        │                                                           │
│        ▼                                                           │
│   ┌──────────────────────────────────────────┐                    │
│   │          Backend Classifier               │                    │
│   │  - ResNet / SE-ResNet                     │                    │
│   │  - LCNN (Light CNN)                       │                    │
│   │  - RawNet2/3                              │                    │
│   │  - Wav2Vec2 + Classifier                  │                    │
│   │  - AASIST (sota)                          │                    │
│   └──────────────────────────────────────────┘                    │
│        │                                                           │
│        ▼                                                           │
│   Binary Output: Real (0) vs Fake (1)                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import torch.nn as nn
import torchaudio


class AudioDeepfakeDetector:
    """
    Detector de audio deepfakes/synthetic voice.
    """

    def __init__(
        self,
        model_name: str = "Harf/wav2vec2-large-xlsr-53-deepfake-detection",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        # Cargar modelo pre-entrenado para deepfake detection
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def load_audio(
        self,
        audio_path: str,
        target_sr: int = 16000
    ) -> torch.Tensor:
        """Carga y preprocesa audio."""
        waveform, sr = torchaudio.load(audio_path)

        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze(0)

    @torch.no_grad()
    def detect(
        self,
        audio_path: str
    ) -> Dict[str, any]:
        """
        Detecta si el audio es real o sintetico.

        Returns:
            Dict con prediccion, probabilidades, y confidence
        """
        # Cargar audio
        waveform = self.load_audio(audio_path)

        # Procesar
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # Inferencia
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Probabilidades
        probs = torch.softmax(logits, dim=-1).squeeze()

        # Clases: 0=real, 1=fake (verificar con modelo especifico)
        prob_real = probs[0].item()
        prob_fake = probs[1].item()

        prediction = "fake" if prob_fake > prob_real else "real"
        confidence = max(prob_real, prob_fake)

        return {
            'prediction': prediction,
            'is_fake': prob_fake > prob_real,
            'probability_real': prob_real,
            'probability_fake': prob_fake,
            'confidence': confidence,
            'risk_level': self._assess_risk(prob_fake)
        }

    def _assess_risk(self, prob_fake: float) -> str:
        """Evalua nivel de riesgo."""
        if prob_fake < 0.3:
            return "low"
        elif prob_fake < 0.6:
            return "medium"
        elif prob_fake < 0.8:
            return "high"
        else:
            return "critical"


class LCNNDeepfakeDetector(nn.Module):
    """
    Light CNN para deteccion de deepfakes en audio.
    Arquitectura ligera y efectiva.
    """

    def __init__(
        self,
        num_classes: int = 2
    ):
        super().__init__()

        # Max-Feature-Map activation
        self.mfm = lambda x: torch.max(
            x[:, :x.shape[1]//2, :, :],
            x[:, x.shape[1]//2:, :, :]
        )[0]

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 96, kernel_size=1)
        self.conv5 = nn.Conv2d(48, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(32 * 8 * 8, 160)  # Ajustar segun input size
        self.fc2 = nn.Linear(80, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, freq, time) spectrogram

        Returns:
            (batch, num_classes) logits
        """
        # Conv blocks con MFM activation
        x = self.mfm(self.conv1(x))
        x = self.pool(x)

        x = self.mfm(self.conv2(x))
        x = self.mfm(self.conv3(x))
        x = self.pool(x)

        x = self.mfm(self.conv4(x))
        x = self.mfm(self.conv5(x))
        x = self.pool(x)

        x = self.mfm(self.conv6(x))
        x = self.mfm(self.conv7(x))
        x = self.pool(x)

        x = self.mfm(self.conv8(x))
        x = self.mfm(self.conv9(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.mfm(self.fc1(x).view(-1, 160, 1, 1).squeeze(-1).squeeze(-1).unsqueeze(1))
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```

## Multimodal Sentiment Analysis

```
Multimodal Sentiment Analysis:

┌────────────────────────────────────────────────────────────────────┐
│           Audio-Visual-Text Sentiment Analysis                      │
│                                                                    │
│   Video con Audio                                                  │
│        │                                                           │
│   ┌────┴────┬───────────────────────┐                             │
│   │         │                       │                              │
│   ▼         ▼                       ▼                              │
│ ┌──────┐ ┌──────┐              ┌──────────┐                       │
│ │Video │ │Audio │              │ Transcr. │                        │
│ │Frames│ │Wavefrm│             │  (ASR)   │                        │
│ └──┬───┘ └──┬───┘              └────┬─────┘                        │
│    │        │                       │                              │
│    ▼        ▼                       ▼                              │
│ ┌──────┐ ┌──────┐              ┌──────────┐                       │
│ │Vision│ │Audio │              │  Text    │                        │
│ │Encoder│ │Encoder│             │ Encoder  │                        │
│ │(ViT) │ │(Wav2V)│             │ (BERT)   │                        │
│ └──┬───┘ └──┬───┘              └────┬─────┘                        │
│    │        │                       │                              │
│    ▼        ▼                       ▼                              │
│   V_emb   A_emb                   T_emb                           │
│    │        │                       │                              │
│    └────────┴───────────┬───────────┘                             │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                              │
│              │   Multimodal Fusion │                              │
│              │   (Cross-Attention) │                              │
│              └──────────┬──────────┘                              │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                              │
│              │   Sentiment Head    │                              │
│              │  (Positive/Negative/│                              │
│              │   Neutral)          │                              │
│              └─────────────────────┘                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Datasets:
- CMU-MOSEI: 23,453 clips, sentiment + emotion
- CMU-MOSI: 2,199 clips, sentiment
- IEMOCAP: Emotion recognition dialogues
- MELD: Friends TV show emotions
```

```python
from transformers import Wav2Vec2Model, BertModel, ViTModel
import torch
import torch.nn as nn


class MultimodalSentimentAnalyzer(nn.Module):
    """
    Analisis de sentimiento multimodal: Audio + Video + Text.
    """

    def __init__(
        self,
        audio_model: str = "facebook/wav2vec2-base",
        text_model: str = "bert-base-uncased",
        vision_model: str = "google/vit-base-patch16-224",
        hidden_dim: int = 256,
        num_classes: int = 3  # positive, negative, neutral
    ):
        super().__init__()

        # Encoders (frozen o fine-tuned)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model)
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.vision_encoder = ViTModel.from_pretrained(vision_model)

        # Dimensiones de cada encoder
        audio_dim = self.audio_encoder.config.hidden_size  # 768
        text_dim = self.text_encoder.config.hidden_size    # 768
        vision_dim = self.vision_encoder.config.hidden_size # 768

        # Proyecciones a espacio comun
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Clasificador
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        audio_input: torch.Tensor,
        text_input: Dict[str, torch.Tensor],
        vision_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_input: (batch, audio_len) waveform
            text_input: Dict con input_ids, attention_mask
            vision_input: (batch, 3, 224, 224) frames

        Returns:
            Dict con logits y features intermedias
        """
        # Encode audio
        audio_outputs = self.audio_encoder(audio_input)
        audio_feats = audio_outputs.last_hidden_state.mean(dim=1)  # (batch, 768)
        audio_feats = self.audio_proj(audio_feats)

        # Encode text
        text_outputs = self.text_encoder(**text_input)
        text_feats = text_outputs.pooler_output  # (batch, 768)
        text_feats = self.text_proj(text_feats)

        # Encode vision
        vision_outputs = self.vision_encoder(vision_input)
        vision_feats = vision_outputs.pooler_output  # (batch, 768)
        vision_feats = self.vision_proj(vision_feats)

        # Stack para cross-attention
        # Shape: (batch, 3, hidden_dim)
        multimodal_feats = torch.stack([audio_feats, text_feats, vision_feats], dim=1)

        # Self-attention entre modalidades
        attended_feats, attention_weights = self.cross_attention(
            multimodal_feats, multimodal_feats, multimodal_feats
        )

        # Flatten y fusion
        fused = attended_feats.view(attended_feats.size(0), -1)
        fused = self.fusion(fused)

        # Clasificacion
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'audio_features': audio_feats,
            'text_features': text_feats,
            'vision_features': vision_feats,
            'attention_weights': attention_weights
        }


class AudioVisualSyncChecker(nn.Module):
    """
    Verifica sincronizacion audio-visual.
    Util para detectar videos manipulados/deepfakes.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        video_dim: int = 768,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        # Temporal alignment module
        self.temporal_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)

        # Sync classifier
        self.sync_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)  # synced / not_synced
        )

    def forward(
        self,
        audio_feats: torch.Tensor,
        video_feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_feats: (batch, time, audio_dim)
            video_feats: (batch, time, video_dim)

        Returns:
            Dict con prediccion de sincronizacion
        """
        # Proyectar
        audio = self.audio_proj(audio_feats)  # (batch, time, hidden)
        video = self.video_proj(video_feats)  # (batch, time, hidden)

        # Concatenar temporalmente
        concat = torch.cat([audio, video], dim=-1)  # (batch, time, hidden*2)

        # Conv temporal
        concat = concat.transpose(1, 2)  # (batch, hidden*2, time)
        temporal = self.temporal_conv(concat)  # (batch, hidden, time)
        temporal = temporal.transpose(1, 2)  # (batch, time, hidden)

        # Pool temporal
        pooled = temporal.mean(dim=1)  # (batch, hidden)

        # Clasificar
        logits = self.sync_head(pooled)

        probs = torch.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'sync_probability': probs[:, 1],
            'is_synced': probs[:, 1] > 0.5
        }
```

## Audio-Visual Speech Recognition

```python
class AudioVisualASR(nn.Module):
    """
    Speech Recognition combinando audio y video (lip reading).
    Mejora robustez en ambientes ruidosos.
    """

    def __init__(
        self,
        whisper_encoder,
        lip_encoder,
        vocab_size: int = 50000,
        hidden_dim: int = 512
    ):
        super().__init__()

        self.audio_encoder = whisper_encoder  # Pre-trained Whisper encoder
        self.lip_encoder = lip_encoder  # Pre-trained lip reading model

        # Fusion
        self.audio_proj = nn.Linear(512, hidden_dim)
        self.lip_proj = nn.Linear(512, hidden_dim)

        # Cross-modal attention
        self.av_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        audio_mel: torch.Tensor,
        lip_video: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            audio_mel: (batch, 80, time) mel spectrogram
            lip_video: (batch, frames, 3, H, W) lip region video
            target_ids: (batch, seq_len) target token ids for training

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Encode audio
        audio_feats = self.audio_encoder(audio_mel)
        audio_feats = self.audio_proj(audio_feats)

        # Encode lips
        batch, frames = lip_video.shape[:2]
        lip_flat = lip_video.view(batch * frames, *lip_video.shape[2:])
        lip_feats = self.lip_encoder(lip_flat)
        lip_feats = lip_feats.view(batch, frames, -1)
        lip_feats = self.lip_proj(lip_feats)

        # Audio-visual fusion via cross-attention
        # Audio attends to lips
        fused_feats, _ = self.av_attention(
            query=audio_feats,
            key=lip_feats,
            value=lip_feats
        )

        # Combine with residual
        encoder_output = audio_feats + fused_feats

        # Decode (en inferencia, usar autoregressive generation)
        if target_ids is not None:
            # Teacher forcing durante training
            tgt_emb = self.embed_tokens(target_ids)
            decoded = self.decoder(tgt_emb, encoder_output)
            logits = self.output_proj(decoded)
            return logits
        else:
            return encoder_output  # Para generacion autoregresiva
```

## Aplicaciones de Seguridad Integradas

```python
class MultimodalSecuritySystem:
    """
    Sistema de seguridad multimodal integrado.
    Combina multiples modalidades para deteccion robusta.
    """

    def __init__(
        self,
        speaker_verifier: SpeakerVerificationSystem,
        deepfake_detector: AudioDeepfakeDetector,
        whisper: WhisperWrapper
    ):
        self.speaker = speaker_verifier
        self.deepfake = deepfake_detector
        self.whisper = whisper

    def full_voice_authentication(
        self,
        audio_path: str,
        claimed_identity: str,
        expected_passphrase: Optional[str] = None,
        liveness_check: bool = True
    ) -> Dict[str, any]:
        """
        Autenticacion de voz completa con multiples verificaciones.

        Returns:
            Dict con resultado de autenticacion y detalles
        """
        result = {
            'authenticated': False,
            'checks': {},
            'risk_score': 0.0,
            'recommendation': ''
        }

        # 1. Deepfake check (liveness)
        if liveness_check:
            deepfake_result = self.deepfake.detect(audio_path)
            result['checks']['deepfake_detection'] = deepfake_result

            if deepfake_result['is_fake']:
                result['recommendation'] = "REJECT: Synthetic voice detected"
                result['risk_score'] = 1.0
                return result

            # Penalizacion por incertidumbre
            if deepfake_result['probability_fake'] > 0.3:
                result['risk_score'] += 0.3

        # 2. Speaker verification
        speaker_result = self.speaker.verify_speaker(
            audio_path,
            claimed_identity,
            threshold=0.6
        )
        result['checks']['speaker_verification'] = speaker_result

        if not speaker_result.get('verified', False):
            result['recommendation'] = "REJECT: Voice does not match enrolled speaker"
            result['risk_score'] = 0.8
            return result

        # 3. Passphrase verification (optional)
        if expected_passphrase:
            transcription = self.whisper.transcribe(audio_path)
            spoken_text = transcription['text'].lower().strip()
            expected = expected_passphrase.lower().strip()

            # Fuzzy matching
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, spoken_text, expected).ratio()

            passphrase_result = {
                'expected': expected,
                'spoken': spoken_text,
                'similarity': similarity,
                'match': similarity > 0.8
            }
            result['checks']['passphrase'] = passphrase_result

            if not passphrase_result['match']:
                result['recommendation'] = "REJECT: Passphrase does not match"
                result['risk_score'] = 0.7
                return result

        # Si todos los checks pasan
        result['authenticated'] = True
        result['recommendation'] = "ACCEPT: All verification checks passed"
        result['risk_score'] = max(0, result['risk_score'])

        return result

    def analyze_suspicious_call(
        self,
        audio_path: str
    ) -> Dict[str, any]:
        """
        Analisis de llamada sospechosa.
        Detecta phishing, scam, amenazas, etc.
        """
        analysis = {}

        # 1. Transcribir
        transcription = self.whisper.transcribe(
            audio_path,
            return_timestamps=True
        )
        analysis['transcription'] = transcription['text']
        analysis['language'] = transcription['language']

        # 2. Analisis de contenido
        text = transcription['text'].lower()

        # Patrones sospechosos
        patterns = {
            'urgency': ['immediately', 'urgent', 'right now', 'today only', 'limited time'],
            'authority': ['irs', 'fbi', 'police', 'government', 'bank official'],
            'threat': ['arrest', 'lawsuit', 'legal action', 'warrant'],
            'financial': ['wire transfer', 'gift card', 'bitcoin', 'cryptocurrency'],
            'personal_info': ['social security', 'credit card', 'bank account', 'password']
        }

        detected_patterns = {}
        for category, keywords in patterns.items():
            found = [kw for kw in keywords if kw in text]
            if found:
                detected_patterns[category] = found

        analysis['suspicious_patterns'] = detected_patterns

        # 3. Deepfake check
        deepfake_result = self.deepfake.detect(audio_path)
        analysis['synthetic_voice'] = deepfake_result

        # 4. Risk assessment
        risk_score = 0.0

        # Penalizacion por patrones
        if 'urgency' in detected_patterns:
            risk_score += 0.2
        if 'authority' in detected_patterns:
            risk_score += 0.3
        if 'threat' in detected_patterns:
            risk_score += 0.3
        if 'financial' in detected_patterns:
            risk_score += 0.4
        if 'personal_info' in detected_patterns:
            risk_score += 0.4

        # Penalizacion por voz sintetica
        if deepfake_result['is_fake']:
            risk_score += 0.5

        risk_score = min(risk_score, 1.0)

        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        analysis['risk_score'] = risk_score
        analysis['risk_level'] = risk_level

        # Recomendaciones
        recommendations = []
        if 'financial' in detected_patterns or 'personal_info' in detected_patterns:
            recommendations.append("DO NOT provide any financial or personal information")
        if 'authority' in detected_patterns:
            recommendations.append("Verify caller identity through official channels")
        if deepfake_result['is_fake']:
            recommendations.append("Voice appears synthetic - high likelihood of scam")
        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append("Recommend ending call and reporting to authorities")

        analysis['recommendations'] = recommendations

        return analysis
```

## Resumen

```
AUDIO MULTIMODAL - KEY TAKEAWAYS:

┌─────────────────────────────────────────────────────────────────┐
│                     AUDIO REPRESENTATIONS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  - Waveform: Raw audio, 16kHz sample rate                      │
│  - Mel Spectrogram: Time-frequency, perceptual scale           │
│  - MFCC: Compressed spectral features                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     MODELOS CLAVE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. WHISPER (OpenAI)                                            │
│     ├── ASR multilingue robusto                                │
│     ├── Encoder-decoder transformer                            │
│     └── 680K horas de entrenamiento                            │
│                                                                 │
│  2. SPEAKER VERIFICATION                                        │
│     ├── ECAPA-TDNN (SpeechBrain)                               │
│     ├── X-Vector                                               │
│     └── Voiceprint embeddings para autenticacion               │
│                                                                 │
│  3. DEEPFAKE DETECTION                                          │
│     ├── LCNN, RawNet2                                          │
│     ├── Wav2Vec2 + clasificador                                │
│     └── Detecta TTS, voice conversion                          │
│                                                                 │
│  4. MULTIMODAL SENTIMENT                                        │
│     ├── Audio + Video + Text                                   │
│     ├── Cross-modal attention                                  │
│     └── Fusion de modalidades                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                APLICACIONES SEGURIDAD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  - Voice Authentication: Verificacion biometrica               │
│  - Audio Deepfake Detection: TTS/VC detection                  │
│  - Call Fraud Detection: Phishing/scam en llamadas             │
│  - Audio-Visual Sync: Detectar videos manipulados              │
│  - Liveness Detection: Anti-spoofing                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

MEJORES PRACTICAS:
- Whisper para transcripcion robusta
- ECAPA-TDNN para speaker verification
- Multiples checks para autenticacion (layered security)
- Combinar audio + text + vision para robustez
- Threshold tuning segun FAR/FRR requeridos
```

## Referencias

1. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper) - Radford et al., 2022
2. "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation" - Desplanques et al., 2020
3. "ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge" - Yamagishi et al., 2021
4. "Audio-Visual Speech Recognition" - Afouras et al., 2018
5. "Multimodal Sentiment Analysis: A Survey" - Poria et al., 2020
6. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks" - Jung et al., 2022
