"""Log-mel-style spectrogram patches for each 1 s hydrophone window.

Same windowing as the hand-crafted pipeline (1 s, 50% overlap) so the two
representations are evaluated on exactly the same windows and groups.
Output: X (n_windows, 32, 32) float32 + a metadata frame aligned row-for-row.
"""
import numpy as np, pandas as pd, soundfile as sf
from scipy import signal as sps

FS, WIN_SEC, HOP_SEC = 8000, 1.0, 0.5
N_MELS, N_FRAMES = 32, 32

def mel_filterbank(n_mels, n_fft, fs):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    edges = mel2hz(np.linspace(hz2mel(50), hz2mel(fs / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        fb[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return fb

def spectrogram(w, fb, n_fft=256):
    hop = max(1, (len(w) - n_fft) // (N_FRAMES - 1))
    f, t, Z = sps.stft(w, fs=FS, nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None)
    S = np.abs(Z) ** 2
    M = fb @ S
    M = np.log10(M + 1e-12)
    # fix frame count
    if M.shape[1] >= N_FRAMES:
        M = M[:, :N_FRAMES]
    else:
        M = np.pad(M, ((0, 0), (0, N_FRAMES - M.shape[1])), mode="edge")
    return M.astype(np.float32)

def main():
    inv = pd.read_csv("hydro_inventory.csv")
    inv = inv[inv.topo != "AMBIENT"].reset_index(drop=True)
    fb = mel_filterbank(N_MELS, 256, FS)
    wl, hop = int(WIN_SEC * FS), int(HOP_SEC * FS)
    X, meta = [], []
    for rid, r in enumerate(inv.itertuples()):
        sig = sf.read(r.file, channels=1, samplerate=FS, subtype="PCM_32", endian="LITTLE")[0]
        for k, s in enumerate(range(0, len(sig) - wl + 1, hop)):
            X.append(spectrogram(sig[s:s + wl], fb))
            meta.append(dict(recording_id=rid, window_idx=k, topo=r.topo, leak=r.leak,
                             flow=r.flow, noise=r.noise, sensor=r.sensor,
                             condition=f"{r.flow}|{r.noise}", is_leak=int(r.leak != "NL")))
        if rid % 30 == 0:
            print(f"  {rid}/{len(inv)}", flush=True)
    X = np.stack(X)
    np.save("spec_X.npy", X)
    pd.DataFrame(meta).to_csv("spec_meta.csv", index=False)
    print("X:", X.shape, "| meta:", len(meta))

if __name__ == "__main__":
    main()
