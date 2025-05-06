import numpy as np
import matplotlib.pyplot as plt

def steering_vec(
    v,     # speed of sound
    f,     # frequency
    d,     # Mic. distance, m
    theta, # degree
    gain   # dB
):
    a = np.ones((2,1), dtype=np.complex_)
    a[1] = (10**(gain/20))*np.exp(-2j * np.pi * f * d * np.cos(theta * np.pi / 180) / v)

    return a

def main():
    num_sensor = 2
    voa = 180 # degree
    grid_size = 180
    v = 330         # m/s
    f = 4000        # hz
    d = 0.02        # m
    theta = 0       # degree
    phi = 90        # degree
    gain_s = 0      # dB
    gain_n = 0      # dB
    a_s = steering_vec(v, f, d, theta, gain_s)
    a_n = steering_vec(v, f, d, phi, gain_n)

    corr = a_s*np.conj(a_s.T) + a_n*np.conj(a_n.T)
    # corr_inv = np.linalg.inv(corr)
    corr_inv = np.array([[corr[1, 1], -corr[0, 1]], [-corr[1, 0], corr[0, 0]]])
    w_s = np.matmul(corr_inv, a_s) / np.matmul(np.conj(a_s.T), np.matmul(corr_inv, a_s))
    w_n = np.matmul(corr_inv, a_n) / np.matmul(np.conj(a_n.T), np.matmul(corr_inv, a_n))

    a_map = np.zeros((num_sensor, grid_size), dtype=np.complex_)
    grid = voa / grid_size
    for n in range(grid_size):
        a_map[:, n] = np.squeeze(steering_vec(v, f, d, grid * n, gain_n))

    spectrum_mag_s = np.squeeze(np.abs(np.matmul(np.conj(w_s.T), a_map))**2)
    # spectrum_mag_n = np.squeeze(np.abs(np.matmul(np.conj(w_n.T), a_map))**2)

    # print(np.max(spectrum_mag))
    # print(np.min(spectrum_mag))

    degree_sign = u'\xb0'
    plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(grid_size), 10*np.log10(spectrum_mag_s))
    plt.plot(np.arange(grid_size), spectrum_mag_s)
    plt.grid(True)
    plt.title('\u03B8={} {}, \u03C6={} {}, f={} Hz'.format(theta, degree_sign, phi, degree_sign, f))
    plt.xlabel('degree')
    plt.ylabel('dB')
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(grid_size), 10*np.log10(spectrum_mag_n))
    # plt.grid(True)
    # plt.xlabel('degree')
    # plt.ylabel('dB')
    plt.show()

if __name__ == "__main__":
    main()