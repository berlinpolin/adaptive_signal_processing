import librosa
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import torch


def main():
    # input_path = 'D:\\Berlin\\mission\\algorithm\\beamforming\\sim\\input\\'
    # output_path = 'D:\\Berlin\\mission\\algorithm\\beamforming\\sim\\output\\'
    # filename = 'distractor_90deg_02_15_2022_11_52_02_27s.wav'

    # x, sr = librosa.load(input_path + filename, mono=False, sr=16000)

    # fc = 90
    # b, a = signal.butter(8, fc, btype='high', fs=sr)
    # zi = signal.lfilter_zi(b, a)
    # x0 = signal.lfilter(b, a, x[0,:])
    # x1 = signal.lfilter(b, a, x[1,:])

    n_fft = 512

    win_ana = np.sqrt(np.hanning(n_fft+1))[:n_fft]
    D = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2, window=win_ana)

    # sf.write(output_path + filename[:-4] + '_hpf_fc' + str(fc) + 'hz.wav', z.T, samplerate=sr)

    # fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # axs[0].plot(x[0,:])
    # axs[0].set_title('Ch0')
    # axs[0].set_ylim(-1, 1)
    # axs[0].grid(True)
    # axs[1].plot(x[1,:])
    # axs[1].set_title('Ch1')
    # axs[1].set_ylim(-1, 1)
    # axs[1].grid(True)
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()