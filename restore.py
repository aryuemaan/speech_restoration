#!/usr/bin/env python3


import argparse
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d
import os
import sys

def read_wav(path):
    sr, x = wavfile.read(path)
    # convert to float32 in -1..1
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32)-128)/128.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x

def write_wav(path, sr, x):
    # normalize to -0.99..0.99 to avoid clipping when writing int16
    mx = np.max(np.abs(x)) if x.size>0 else 0.0
    if mx > 0.999:
        x = x / mx * 0.99
    wavfile.write(path, sr, (x * 32767.0).astype(np.int16))

def highpass(x, sr, fc=80.0, order=4):
    b, a = signal.butter(order, fc/(sr/2), btype='highpass')
    return signal.filtfilt(b, a, x)

def stft_enhance(x, sr, nfft=1024, hop=None, hp_alpha=1.2, beta_floor=0.02,
                 tonal_thresh=5.0, tonal_att=0.25):
    if hop is None:
        hop = nfft//4
    win = signal.windows.hann(nfft, sym=False)
    f, t, Zxx = signal.stft(x, fs=sr, window=win, nperseg=nfft, noverlap=nfft-hop, boundary='zeros', padded=True)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # noise estimate (median across time)
    noise_floor = np.median(mag, axis=1, keepdims=True)

    # spectral subtraction (over-subtraction)
    mag_ss = mag - hp_alpha * noise_floor
    mag_ss = np.maximum(mag_ss, beta_floor * noise_floor)

    # tonal / narrowband suppression (frame-wise median across freq)
    med_freq = np.median(mag_ss, axis=0, keepdims=True)
    mask_tonal = mag_ss > (tonal_thresh * med_freq)
    mag_ss[mask_tonal] = mag_ss[mask_tonal] * tonal_att

    # Wiener-like smoothing gain
    noise_var = (noise_floor**2)
    gain = (mag_ss**2) / (mag_ss**2 + noise_var + 1e-12)
    mag_enh = gain * mag_ss

    # ISTFT
    _, x_rec = signal.istft(mag_enh * np.exp(1j*phase), fs=sr, window=win, nperseg=nfft, noverlap=nfft-hop, input_onesided=True, boundary=True)
    return x_rec

def find_regions(bool_arr):
    regions = []
    N = len(bool_arr)
    i = 0
    while i < N:
        if bool_arr[i]:
            j = i
            while j < N and bool_arr[j]:
                j += 1
            regions.append((i, j))
            i = j
        else:
            i += 1
    return regions

def inpaint_gaps(x_rec, corrupted, sr, context_ms=20, min_gap_ms=5):
    abs_corr = np.abs(corrupted)
    low_thresh = np.percentile(abs_corr, 5) * 0.6
    is_silent = abs_corr < low_thresh
    min_gap_samps = int((min_gap_ms/1000.0) * sr)
    gaps = find_regions(is_silent)
    x_out = x_rec.copy()
    for (i, j) in gaps:
        L = j - i
        if L < min_gap_samps:
            continue
        pre = max(0, i - int((context_ms/1000.0)*sr))
        post = min(len(x_rec)-1, j + int((context_ms/1000.0)*sr))
        if (j+1) <= post:
            xp = np.concatenate([np.arange(pre, i), np.arange(j+1, post+1)])
        else:
            xp = np.arange(pre, i)
        if xp.size < 4:
            v1 = x_rec[pre] if pre >= 0 else 0.0
            v2 = x_rec[post] if post < len(x_rec) else 0.0
            x_out[i:j] = np.linspace(v1, v2, L)
        else:
            fp = x_rec[xp]
            f_interp = interp1d(xp, fp, kind='cubic', fill_value="extrapolate")
            x_out[i:j] = f_interp(np.arange(i, j))
    return x_out

def repair_clipping(x_out, corrupted):
    clip_mask = (np.abs(corrupted) > 0.97)
    clips = find_regions(clip_mask)
    repaired = x_out.copy()
    for (i, j) in clips:
        L = j - i
        if L == 0:
            continue
        pre = max(0, i - 100)
        post = min(len(repaired)-1, j + 100)
        if (j+1) <= post:
            xp = np.concatenate([np.arange(pre, i), np.arange(j+1, post+1)])
        else:
            xp = np.arange(pre, i)
        if xp.size < 4:
            v1 = repaired[pre] if pre >= 0 else 0.0
            v2 = repaired[post] if post < len(repaired) else 0.0
            repaired[i:j] = np.linspace(v1, v2, L)
        else:
            fp = repaired[xp]
            f_interp = interp1d(xp, fp, kind='quadratic', fill_value='extrapolate')
            repaired[i:j] = f_interp(np.arange(i, j))
    return repaired

def lowpass_smooth(x, sr, cutoff=7800.0, order=3):
    b, a = signal.butter(order, cutoff/(sr/2), btype='low')
    return signal.filtfilt(b, a, x)

def rms(x): return np.sqrt(np.mean(x**2) + 1e-16)

def compute_snr(ref, est):
    L = min(len(ref), len(est))
    ref = ref[:L]
    est = est[:L]
    noise = ref - est
    return 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-16))

def main(args):
    sr, corrupted = read_wav(args.input)
    if sr != 16000:
        print(f"Warning: input sample rate is {sr}, expected 16000 Hz.", file=sys.stderr)

    clean = None
    if args.clean and os.path.exists(args.clean):
        sr2, clean = read_wav(args.clean)
        if sr2 != sr:
            clean = signal.resample_poly(clean, sr, sr2)

    # 1) High-pass
    x_hp = highpass(corrupted, sr, fc=80.0, order=4)

    # 2) STFT + spectral subtraction + tonal suppression + Wiener-like smoothing
    x_rec = stft_enhance(x_hp, sr,
                         nfft=args.nfft,
                         hop=args.hop,
                         hp_alpha=args.alpha,
                         beta_floor=args.beta,
                         tonal_thresh=args.tonal_thresh,
                         tonal_att=args.tonal_att)

    # adjust length
    if len(x_rec) < len(corrupted):
        x_rec = np.pad(x_rec, (0, len(corrupted)-len(x_rec)))
    else:
        x_rec = x_rec[:len(corrupted)]

    # 3) Inpaint gaps
    x_inpaint = inpaint_gaps(x_rec, corrupted, sr)

    # 4) Clipping repair
    x_repaired = repair_clipping(x_inpaint, corrupted)

    # 5) Lowpass smoothing
    x_smooth = lowpass_smooth(x_repaired, sr, cutoff=7800.0)

    # 6) RMS normalize to corrupted loudness
    r_cor = rms(corrupted)
    r_out = rms(x_smooth)
    if r_out > 0:
        x_out = x_smooth * (r_cor / (r_out + 1e-12))
    else:
        x_out = x_smooth

    x_out = np.clip(x_out, -0.99, 0.99)
    write_wav(args.output, sr, x_out)
    print(f"Wrote reconstructed file to: {args.output}")

    if clean is not None:
        snr_cor = compute_snr(clean, corrupted)
        snr_rec = compute_snr(clean, x_out)
        print(f"SNR (corrupted vs clean): {snr_cor:.2f} dB")
        print(f"SNR (reconstructed vs clean): {snr_rec:.2f} dB")
        print(f"SNR improvement: {snr_rec - snr_cor:.2f} dB")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Speech restoration (classical DSP)")
    p.add_argument('--input', type=str, default='./corrupted.wav', help='Path to corrupted input WAV (16 kHz mono)')
    p.add_argument('--output', type=str, default='./reconstructed.wav', help='Path for output WAV')
    p.add_argument('--clean', type=str, default=None, help='Optional clean reference WAV for metrics')
    p.add_argument('--nfft', type=int, default=1024, help='STFT FFT size')
    p.add_argument('--hop', type=int, default=1024//4, help='STFT hop size')
    p.add_argument('--alpha', type=float, default=1.2, help='Spectral subtraction over-subtraction factor')
    p.add_argument('--beta', type=float, default=0.02, help='Spectral subtraction flooring factor')
    p.add_argument('--tonal_thresh', type=float, default=5.0, help='Tonal peak threshold (× median)')
    p.add_argument('--tonal_att', type=float, default=0.25, help='Tonal peak attenuation factor')
    args = p.parse_args()
    main(args)
