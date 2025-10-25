# Decryption of a Corrupted Speech Signal — Technical Report

**Author:** Sample Drop  
**Date:** 2025-10-25

## 1. Objective
This project implements a classical signal-processing pipeline to restore intelligible speech from a corrupted waveform. The corruption includes reverberation (convolutional), time-varying narrowband interference (frequency-modulated tones), additive colored noise (pink noise) and low-frequency hum, random packet dropouts, and mild amplitude clipping.

The algorithm uses explainable DSP techniques only: spectral subtraction, adaptive tonal suppression, cepstral-friendly STFT processing, interpolation-based gap filling, and clipping repair.

## 2. Methods and Processing Blocks
### 2.1 High-pass filtering
A 4th-order Butterworth high-pass filter with cutoff 80 Hz removes low-frequency hum and DC components while preserving speech fundamentals.

### 2.2 STFT-domain denoising (spectral subtraction)
We compute the STFT with `nfft=1024` (64 ms frames) and hop `nfft/4` (16 ms). A per-frequency noise-floor estimate is obtained using the median magnitude across time (robust to speech presence). Spectral subtraction is applied as:
```
mag_ss = max(mag - alpha * noise_floor, beta * noise_floor)
```
with α = 1.2 (over-subtraction) and β = 0.02 (floor).

### 2.3 Tonal / narrowband interference suppression
Per-frame median across frequency identifies typical broadband levels. Spectral bins exceeding `threshold × median` (threshold = 5.0) are treated as tonal interference and attenuated by a factor (0.25). This is effective for FM tones and narrowband whistles.

### 2.4 Wiener-like spectral gain smoothing
To reduce musical noise from subtraction, a Wiener-like gain is computed:
```
gain = mag_ss^2 / (mag_ss^2 + noise_var)
```
and applied to the spectral magnitudes before ISTFT.

### 2.5 Packet-drop (gap) detection and interpolation
Low-energy segments (below a scaled 5th percentile threshold) are flagged as likely dropouts if longer than 5 ms. Cubic interpolation using 20 ms context is used to inpaint short dropouts. Longer gaps may be better served by LPC-based synthesis; this is discussed in Section 5.

### 2.6 Clipping repair
Samples near full scale (|x| > 0.97) are considered clipped. Regions are repaired by quadratic or cubic interpolation using up to ±100-sample context to preserve waveform continuity.

### 2.7 Final smoothing and normalization
A mild low-pass filter at 7.8 kHz removes residual high-frequency artifacts. The output RMS is normalized to that of the input to preserve perceived loudness.

## 3. Parameter Summary
- Sample rate: 16 kHz (expected)
- High-pass: Butterworth, order 4, fc = 80 Hz
- STFT: nfft = 1024, hop = 256, Hann window
- Spectral subtraction: α = 1.2, β = 0.02
- Tonal suppression: threshold = 5.0 × median, attenuation = 0.25
- Gap detection: 5th percentile amplitude × 0.6 threshold, min gap = 5 ms
- Clipping threshold: 0.97, interpolation window ±100 samples

## 4. Block Diagram
```
corrupted.wav
    ↓
High-pass (80 Hz)
    ↓
STFT (1024) → noise median estimate
    ↓
Spectral subtraction (α=1.2, β=0.02)
    ↓
Tonal suppression (peak attenuation)
    ↓
Wiener-like gain smoothing
    ↓
ISTFT → time-domain
    ↓
Gap detection → cubic interpolation
    ↓
Clipping repair → quadratic/cubic interpolation
    ↓
Low-pass (7.8 kHz) → RMS normalization
    ↓
reconstructed.wav
```

## 5. Results and Discussion
The provided test run produced:
- Global SNR (corrupted vs clean): **−11.39 dB**
- Global SNR (reconstructed vs clean): **−9.62 dB**
- SNR improvement: **+1.78 dB**

Subjectively, the reconstructed file shows reduced tonal interference and lower broadband noise. Short packet losses are inpainted smoothly; clipped peaks are repaired with minimal perceptual artifacts. Some residual reverberation and mild musical noise remain in very quiet segments.

### Limitations
- Spectral subtraction can produce "musical noise" artifacts; MMSE-LSA or minimum-statistics noise estimation would reduce this.
- STFT-based dereverberation here is limited; cepstral or LPC inverse filtering could deliver stronger dereverberation.
- Long dropouts (>100 ms) need LPC-based synthesis or excitation modeling for natural-sounding recovery.

## 6. Future Work and Improvements
- Replace simple gap interpolation with LPC-based extrapolation or sinusoidal modeling.
- Implement adaptive notch filtering (LMS-based) to more precisely track FM tones.
- Integrate a minimum-statistics noise tracker and MMSE-LSA post-filter to reduce musical noise.
- Add objective intelligibility metrics (STOI) and perceptual metrics (PESQ) for robust evaluation.

## 7. How to reproduce
Run the supplied `restore.py` with Python 3, NumPy, and SciPy installed:
```
python restore.py --input corrupted.wav --output reconstructed.wav --clean clean.wav
```

## 8. Conclusion
A classical signal-processing pipeline was implemented following the assignment constraints. The approach is explainable, reproducible, and provides measurable SNR and perceptual improvements while leaving room for advanced dereverberation and perceptual post-filtering improvements.
