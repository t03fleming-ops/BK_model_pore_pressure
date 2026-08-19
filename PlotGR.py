import numpy as np
import matplotlib.pyplot as plt


def plot_GR(alpha, N):
    mags = []
    for n in N:
        event = np.load(f'event_data_SRF/alpha_change/event_sample_SRF_a:{alpha}_{n}.npy')
        mag = np.log(np.abs(event[:, 2]))
        mags.append(mag)

    mags = np.concatenate(mags)

    fig, ax = plt.subplots(figsize=(15, 10))

    R_mu, mu = np.histogram(mags, bins=np.arange(np.min(mags), np.max(mags), 0.1))
    mu = mu[1:]
    mask = R_mu != 0
    mu = mu[mask]; R_mu = R_mu[mask]
    ax.plot(mu, np.log(R_mu))
    fname = f'Figures/GR_{alpha}.png'
    plt.savefig(fname)
    return fname