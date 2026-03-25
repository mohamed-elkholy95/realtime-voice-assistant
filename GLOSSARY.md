# 📖 Glossary — Speech Processing & Voice Assistant Terms

A quick-reference guide to the key concepts used in this project, organized by domain.

---

## Audio & Signal Processing

| Term | Definition |
|------|-----------|
| **Sample Rate** | Number of audio samples captured per second (Hz). 16 kHz is standard for speech (covers frequencies up to 8 kHz per the Nyquist theorem). |
| **Nyquist Frequency** | The maximum frequency that can be represented at a given sample rate, equal to sample_rate / 2. Frequencies above this cause aliasing. |
| **Pre-emphasis** | A first-order high-pass filter (y[n] = x[n] − α·x[n−1]) that boosts high frequencies to compensate for the natural spectral tilt of voiced speech. |
| **Framing** | Splitting a continuous audio signal into short overlapping segments (typically 25 ms with 10 ms hop) for analysis. Speech is approximately stationary within a single frame. |
| **Windowing** | Multiplying each frame by a smooth taper function (e.g., Hanning window) to reduce spectral leakage caused by the abrupt edges of framing. |
| **FFT (Fast Fourier Transform)** | An efficient algorithm to compute the Discrete Fourier Transform, converting a time-domain signal into its frequency-domain representation. |
| **STFT (Short-Time Fourier Transform)** | FFT applied to each windowed frame, producing a time-frequency representation (spectrogram). |
| **Spectrogram** | A 2D visualization showing how the frequency content of a signal evolves over time. X-axis = time, Y-axis = frequency, color = magnitude. |
| **SNR (Signal-to-Noise Ratio)** | Ratio of signal power to noise power in decibels. Higher SNR = cleaner audio. SNR = 10·log₁₀(P_signal / P_noise). |
| **Clipping** | Distortion that occurs when a signal exceeds the representable amplitude range (e.g., ±32767 for int16). |

## Feature Extraction

| Term | Definition |
|------|-----------|
| **Mel Scale** | A perceptual frequency scale where equal distances sound equally spaced to human listeners. mel = 2595·log₁₀(1 + f/700). |
| **Mel Filter Bank** | A set of triangular bandpass filters spaced on the Mel scale, used to convert linear frequency spectra into a perceptually-relevant representation. |
| **MFCC (Mel-Frequency Cepstral Coefficients)** | The dominant feature representation for speech. Computed via: pre-emphasis → framing → FFT → Mel filterbank → log → DCT. Typically 13 coefficients per frame. |
| **Delta Features (Δ)** | Temporal derivatives of MFCCs capturing how features change over time (velocity). Computed via regression over a window of ±N frames. |
| **Delta-Delta Features (ΔΔ)** | Second-order temporal derivatives (acceleration). Delta of delta features. The standard 39-D feature vector is [13 static + 13 Δ + 13 ΔΔ]. |
| **DCT (Discrete Cosine Transform)** | Transform that decorrelates the log Mel energies and compresses information into fewer coefficients. Lower coefficients capture the vocal tract shape. |
| **Log-Mel Spectrogram** | Mel filterbank energies after log compression. Used directly by modern models like Whisper instead of MFCCs. |
| **Cepstrum** | The "spectrum of a spectrum" — the inverse Fourier transform of the log magnitude spectrum. MFCCs are a special case using the Mel scale + DCT. |

## Speech Recognition (ASR / STT)

| Term | Definition |
|------|-----------|
| **ASR (Automatic Speech Recognition)** | The task of converting spoken audio into written text. Also called STT (Speech-to-Text). |
| **Whisper** | OpenAI's encoder-decoder Transformer model (2022) trained on 680K hours of multilingual audio. Handles ASR, translation, and language detection end-to-end. |
| **WER (Word Error Rate)** | Standard ASR metric: (substitutions + deletions + insertions) / reference_words. 0% = perfect. Human-level ≈ 4–5% on conversational speech. |
| **CER (Character Error Rate)** | Character-level equivalent of WER. Useful for languages without clear word boundaries (Chinese, Japanese). |
| **Levenshtein Distance** | The minimum number of single-element edits (insert, delete, substitute) to transform one sequence into another. Computed via dynamic programming in O(m·n). |
| **VAD (Voice Activity Detection)** | Identifying which portions of audio contain speech vs. silence. Used for endpointing, noise reduction, and preprocessing before ASR. |
| **Endpointing** | Detecting when a speaker starts and stops talking — critical for determining when to run ASR on a streaming audio input. |
| **Language Model** | A probabilistic model of word sequences that helps ASR resolve ambiguities (e.g., "recognize speech" vs. "wreck a nice beach"). |

## Natural Language Understanding (NLU)

| Term | Definition |
|------|-----------|
| **Intent Classification** | Determining WHAT a user wants to do from their utterance (e.g., "play jazz" → music intent, "what time is it" → time intent). |
| **Entity Extraction (NER)** | Identifying and classifying named entities in text (e.g., "Set a timer for 5 minutes" → entity: duration="5 minutes"). |
| **Slot Filling** | Extracting specific values required by an intent (e.g., music intent needs slots: genre, artist, playlist). |
| **Fuzzy Matching** | Approximate string matching that tolerates typos and variations. The Ratcliff/Obershelp algorithm compares the ratio of matching characters. |
| **Confidence Score** | A value (0–1) indicating how certain the classifier is about its prediction. Low confidence may trigger a clarification question. |
| **Confusion Matrix** | A table showing how often each true class was predicted as each possible class. Reveals systematic misclassification patterns. |
| **Precision** | Of all predictions for a class, what fraction were correct? TP / (TP + FP). High precision = few false alarms. |
| **Recall** | Of all actual instances of a class, what fraction did we find? TP / (TP + FN). High recall = few missed detections. |
| **F1 Score** | Harmonic mean of precision and recall: 2·P·R / (P + R). Balances both concerns into a single metric. |

## Text-to-Speech (TTS)

| Term | Definition |
|------|-----------|
| **TTS (Text-to-Speech)** | Converting written text into spoken audio. Modern systems use neural networks for natural-sounding synthesis. |
| **Vocoder** | The component that converts acoustic features (mel spectrogram) into an audio waveform. Examples: WaveGlow, HiFi-GAN, WaveNet. |
| **Tacotron** | A sequence-to-sequence model that generates mel spectrograms from text. Tacotron 2 is widely used with an attention mechanism for alignment. |
| **SSML (Speech Synthesis Markup Language)** | An XML-based markup for fine-grained TTS control: rate, pitch, pauses, emphasis, pronunciation. |
| **Prosody** | The rhythm, stress, and intonation patterns in speech. Controlling prosody makes synthesized speech sound more natural. |
| **Pitch Scaling** | Changing the perceived pitch of audio without altering duration. Requires a phase vocoder for artifact-free results. |
| **Speech Rate** | The speed of speech output. Controlled by time-stretching (WSOLA) or resampling during synthesis. |

## API & System Design

| Term | Definition |
|------|-----------|
| **REST API** | An architectural style for web services using standard HTTP methods (GET, POST, PUT, DELETE) with resource-oriented URLs. |
| **Token Bucket** | A rate-limiting algorithm that allows short bursts while enforcing a sustained request rate. Tokens refill at a constant rate up to a bucket capacity. |
| **Rate Limiting** | Restricting the number of API requests a client can make in a time window to prevent abuse and ensure fair usage. |
| **CORS (Cross-Origin Resource Sharing)** | Browser security mechanism that controls which domains can call an API. The server sets `Access-Control-Allow-Origin` headers. |
| **Pydantic** | Python library for data validation using type annotations. FastAPI uses it for automatic request/response validation and OpenAPI schema generation. |

---

*This glossary covers terms used in the project source code, tests, and documentation. For deeper exploration of any concept, check the inline docstrings in the corresponding source module.*
