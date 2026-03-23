import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("📚 Learn: Voice Assistant Concepts")
st.markdown("Educational deep-dives into the technologies behind voice assistants.")

# ── What is a Voice Assistant? ───────────────────────────────────────────

st.header("🎙️ What is a Voice Assistant?")

st.markdown("""
A **voice assistant** is a software system that processes spoken language to understand user
requests and provide spoken (or visual) responses. Think Siri, Alexa, or Google Assistant.

The core challenge is converting between two very different representations:
- **Speech**: Continuous analog signal with infinite possible utterances
- **Meaning**: Discrete intent + entities that trigger specific actions

This requires a pipeline of specialized models, each solving one piece of the puzzle.
""")

# ── Speech-to-Text ──────────────────────────────────────────────────────

st.header("📝 Speech-to-Text (ASR)")

st.markdown("""
### How Automatic Speech Recognition Works

**Goal:** Convert an audio waveform into a text transcript.

**Step 1: Feature Extraction**
Raw audio (waveform) contains too much data. We compress it into a compact representation:

1. **Framing**: Split audio into 25ms windows with 10ms overlap
   - Short enough to be "stationary" (spectral content doesn't change much)
   - Long enough for good frequency resolution

2. **Windowing**: Apply a Hanning window to each frame
   - Tapers frame edges to zero, reducing "spectral leakage"
   - Spectral leakage = energy from one frequency bleeding into adjacent bins

3. **FFT**: Compute the Discrete Fourier Transform of each windowed frame
   - Converts time-domain signal → frequency-domain spectrum
   - Shows which frequencies are present and their energies

4. **Mel Filter Bank**: Apply triangular filters on the Mel scale
   - Compresses the frequency axis to match human hearing
   - Humans are better at distinguishing low frequencies than high ones
   - The Mel scale is quasi-logarithmic: mel = 2595 × log₁₀(1 + f/700)

5. **Log Compression**: Take the log of filter bank energies
   - Models human loudness perception (Weber-Fechner law)
   - log(loudness) ∝ perceived loudness

**Step 2: Acoustic Modeling** (Whisper does this end-to-end)
- Maps features → phonemes (or directly to text)
- Uses an encoder-decoder Transformer architecture
- Encoder: compresses the spectrogram into a latent representation
- Decoder: generates text tokens autoregressively

**Step 3: Language Modeling**
- Applies linguistic knowledge to resolve ambiguities
- "recognize speech" vs "wreck a nice beach" (same sound, different meaning)
- Integrated into modern end-to-end models
""")

with st.expander("🔧 Technical: Whisper Architecture"):
    st.markdown("""
**OpenAI Whisper** (2022) is trained on 680K hours of multilingual audio:

- **Encoder**: 24-layer Transformer that processes the audio spectrogram
- **Decoder**: 24-layer Transformer that generates text autoregressively
- **Multitask**: Can do transcription, translation, and language identification
- **Robust**: Works across accents, noise levels, and languages

| Model | Params | Speed | Quality |
|-------|--------|-------|---------|
| Tiny  | 39M    | ~10x  | Fair    |
| Base  | 74M    | ~5x   | Good    |
| Small | 244M   | ~2x   | Very Good |
| Medium| 769M   | ~1x   | Excellent |
| Large | 1550M  | ~0.5x | Best    |
    """)

# ── MFCCs Explained ─────────────────────────────────────────────────────

st.header("🌀 Mel-Frequency Cepstral Coefficients (MFCCs)")

st.markdown("""
### What are MFCCs?

MFCCs are the most widely used features in speech processing. They capture
the **shape of the vocal tract** — which is what distinguishes different sounds.

Think of it this way:
- Your **vocal cords** produce a buzz (the "source") — determines pitch
- Your **vocal tract** (throat, mouth, tongue position) filters it — determines the sound
- MFCCs isolate the vocal tract filter by removing the source information

### How MFCCs are Computed

```
Audio → Pre-emphasis → Framing → FFT → Mel Filter Bank → Log → DCT → MFCCs
```

1. **Pre-emphasis** (y[n] = x[n] - 0.97 × x[n-1])
   - Boosts high frequencies to compensate for natural -12dB/octave falloff

2. **FFT** → Power Spectrum (energy at each frequency)

3. **Mel Filter Bank** (26 triangular filters)
   - Converts linear frequency to perceptual Mel frequency
   - Each filter covers a Mel-spaced frequency band

4. **Log Compression** (log of filter bank energies)
   - Human loudness perception is logarithmic

5. **DCT** (Discrete Cosine Transform)
   - **This is the key step!** DCT decorrelates the Mel energies
   - Lower coefficients (MFCC 1-12) capture the spectral envelope (vocal tract shape)
   - Higher coefficients capture fine details (often discarded)
   - Cepstral analysis literally separates source from filter!
""")

with st.expander("🧮 Why DCT gives us Cepstral Coefficients"):
    st.markdown("""
The term "cepstral" is literally "spectral" with the first four letters reversed.
This isn't a joke — it reflects the mathematical duality:

- **Spectrum**: frequency domain of a TIME signal
- **Cepstrum**: "frequency domain" of a FREQUENCY signal

In speech:
- The power spectrum has two components multiplied together:
  - Source spectrum (vocal cord buzz — harmonics of the fundamental frequency)
  - Filter spectrum (vocal tract — smooth spectral envelope)

- In the LOG domain, multiplication becomes ADDITION:
  `log(power spectrum) = log(source) + log(filter)`

- The DCT then acts as a "frequency analysis" of this log spectrum:
  - **Low "quefrencies"** → slow variations → the spectral envelope (vocal tract) → **MFCCs**
  - **High "quefrencies"** → rapid variations → the harmonic structure (pitch)

By keeping only the low-order DCT coefficients, we isolate the vocal tract
characteristics and discard pitch information. This is brilliant because:
- Same vowel at different pitches → same MFCCs (good for recognition)
- Different vowels at same pitch → different MFCCs (good for discrimination)
    """)

# ── Intent Classification ───────────────────────────────────────────────

st.header("🧠 Intent Classification")

st.markdown("""
### What is Intent Classification?

Given a user's text, determine **WHAT they want to do**:

| Utterance | Intent | Entities |
|-----------|--------|----------|
| "What's the weather?" | weather | — |
| "Play jazz music" | music | genre=jazz |
| "Remind me at 5 PM" | reminder | time=17:00 |
| "Calculate 15% of 200" | calculation | expr=15%×200 |

### Approaches

1. **Rule-based** (this project)
   - Keyword matching + regex patterns
   - Fast, deterministic, easy to debug
   - Limited to pre-defined patterns
   - Good for demonstration and narrow domains

2. **Supervised ML**
   - Train a classifier on labeled examples
   - SVM, Random Forest, or neural networks
   - Requires labeled dataset (can be expensive)

3. **Deep Learning** (production)
   - BERT, RoBERTa, DistilBERT fine-tuned for intent classification
   - Joint Intent Classification + Entity Extraction (NLU)
   - Handles unseen utterances via transfer learning
   - Requires GPU for training, but CPU inference is feasible

4. **Few-shot / Zero-shot** (frontier)
   - GPT-style models with prompt engineering
   - No training data needed (just a description of each intent)
   - Can handle complex, multi-intent utterances
""")

# ── Text-to-Speech ──────────────────────────────────────────────────────

st.header("🔊 Text-to-Speech (TTS)")

st.markdown("""
### How TTS Works

**Goal:** Convert text into natural-sounding speech.

**Step 1: Text Analysis (Frontend)**
- Text normalization: "15" → "fifteen", "Dr." → "Doctor"
- Phonemization: Convert words to phonemes
  - "hello" → /h ə l oʊ/
- Prosody prediction: Stress, intonation, rhythm

**Step 2: Acoustic Model**
- Neural network generates a mel spectrogram from phonemes
- Tacotron 2: encoder-decoder with attention
- FastSpeech: non-autoregressive (faster, parallel generation)

**Step 3: Vocoder**
- Converts mel spectrogram → audio waveform
- WaveNet: autoregressive, high quality, slow
- HiFi-GAN: GAN-based, fast, high quality
- Griffin-Lim: no neural network needed, lower quality

### Speech Rate & Pitch Control

- **Speech Rate**: Speed up by removing frames, slow down by duplicating
  - Professional systems use WSOLA (Waveform Similarity Overlap-Add)
- **Pitch Scaling**: Change perceived pitch without changing duration
  - Requires a phase vocoder (frequency-domain processing)
""")

# ── WER Explained ────────────────────────────────────────────────────────

st.header("📏 Word Error Rate (WER)")

st.markdown("""
### The Standard ASR Metric

**WER** measures how many edits are needed to transform the hypothesis
into the reference:

```
WER = (Substitutions + Deletions + Insertions) / Reference Words
```

**Example:**
```
Reference:    the cat sat on the mat
Hypothesis:   the bat sat on the rug

Operations:   cat → bat (1 substitution), mat → rug (1 substitution)
WER = 2 / 6 = 33.3%
```

### Using Dynamic Programming (Levenshtein Distance)

We use a 2D table to find the minimum-cost alignment:

|     |     | the | bat | sat | on  | the | mat |
|-----|-----|-----|-----|-----|-----|-----|-----|
|     |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
| the |  1  |  0  |  1  |  2  |  3  |  4  |  5  |
| cat |  2  |  1  |  1  |  2  |  3  |  4  |  5  |
| sat |  3  |  2  |  2  |  1  |  2  |  3  |  4  |
| on  |  4  |  3  |  3  |  2  |  1  |  2  |  3  |
| the |  5  |  4  |  4  |  3  |  2  |  1  |  2  |
| mat |  6  |  5  |  5  |  4  |  3  |  2  |  2*  |

*But "cat" → "bat" adds another substitution → actual edit distance is 2.

### WER Benchmarks

| System | Condition | WER |
|--------|-----------|-----|
| Human | Clean | 4-5% |
| Human | Noisy | 10-15% |
| Whisper Large | Clean | ~3% |
| Whisper Tiny | Clean | ~10% |
| Traditional ASR | Clean | 5-15% |
| Traditional ASR | Noisy | 15-30% |
""")

# ── Further Reading ──────────────────────────────────────────────────────

st.header("📖 Further Reading")

st.markdown("""
- [Whisper Paper (OpenAI, 2022)](https://cdn.openai.com/papers/whisper.pdf)
- [Davis & Mermelstein (1980) — Original MFCC paper](https://ieeexplore.ieee.org/document/1163420)
- [MFCCs Explained (Practical Cryptography)](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- [Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Tacotron 2 (Google, 2017)](https://arxiv.org/abs/1712.05884)
- [Coqui TTS — Open Source TTS](https://github.com/coqui-ai/TTS)
""")
