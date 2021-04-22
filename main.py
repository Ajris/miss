import csv

from pyabc import (ABCSMC,
                   RV, Distribution,
                   MedianEpsilon,
                   LocalTransition)
from pyabc.visualization import plot_kde_2d, plot_data_callback
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
import scipy as sp

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# beta, gamma, alpha, mi, theta, theta_0, sigma, eta, kappa_1, kappa_2

beta = 0.2  # contact rate of susceptible individuals with spreaders
GAMMA = 0.0714  # transmission rate from infected to recovery (1/14, because it takes two weeks)
alpha = 0.9  # infected to quarantine
MI = 0.01  # natural birth and death rate
theta = 0  # individuals home quarantine or stay at home rate
theta_0 = 0.0  # lift stay at home order due to the ineffectiveness of home quarantine
SIGMA = 0.1923  # people who completed incubation period becomes infected at a rate of sigma
eta = 0  # exposed to qurantine

kappa_1 = 0.7  # infectious recover
kappa_2 = 0.7  # isolated recover

psi = MI + theta + theta_0
phi = SIGMA + eta + MI
epsilon = alpha + GAMMA + MI

R_0 = beta * SIGMA * (MI + theta_0) / (phi * epsilon * psi)


def get_from_csv():
    with open('data/full_global_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        data = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            if row['Country_code'] == 'PL' and len(data) < 50:
                # print(row['Cumulative_cases'])
                data.append(float(row['New_cases']))
            line_count += 1
        print(f'Processed {line_count} lines.')
        print(len(data))
        return data


# beta, gamma, alpha, mi, theta, theta_0, sigma, eta, kappa_1, kappa_2
# def f(z, t0, beta, gamma, alpha, mi, theta, theta_0, sigma, eta, kappa_1, kappa_2):
#     u, w, h, v, q, r, d = z
#
#     dudt = mi - beta * u * w - (theta + mi) * u + theta_0 * h
#     dhdt = theta * u - (mi + theta_0) * h
#     dvdt = beta * u * w - (sigma + eta + mi) * v
#     dwdt = sigma * v - (alpha + gamma + mi) * w
#     dqdt = eta * v + alpha * w - (gamma + mi) * q
#     drdt = kappa_1 * gamma * w + kappa_2 * gamma * q - mi * r
#     dddt = (1 - kappa_1) * gamma * w + (1 - kappa_2) * gamma * q - mi * d
#
#     return dudt, dwdt, dhdt, dvdt, dqdt, drdt, dddt

def f(z, t0, eta, alpha):
    u, w, h, v, q, r, d = z

    dudt = MI - beta * u * w - (theta + MI) * u + theta_0 * h
    dhdt = theta * u - (MI + theta_0) * h
    dvdt = beta * u * w - (SIGMA + eta + MI) * v
    dwdt = SIGMA * v - (alpha + GAMMA + MI) * w
    dqdt = eta * v + alpha * w - (GAMMA + MI) * q
    drdt = kappa_1 * GAMMA * w + kappa_2 * GAMMA * q - MI * r
    dddt = (1 - kappa_1) * GAMMA * w + (1 - kappa_2) * GAMMA * q - MI * d

    return dudt, dwdt, dhdt, dvdt, dqdt, drdt, dddt


def distance(simulation, data):
    return np.absolute(data["X_2"] - simulation["X_2"]).sum()


def main1():
    measurement_data = np.array(get_from_csv()) / 40000000
    measurement_times = np.arange(len(measurement_data))
    u = 39999999 / 40000000
    w = 0.0
    h = 0
    v = 1 / 40000000
    q = 0
    r = 0
    d = 0
    init = np.array([u, w, h, v, q, r, d])

    # beta, gamma, alpha, mi, theta, theta_0, sigma, eta, kappa_1, kappa_2
    def model(pars):
        sol = sp.integrate.odeint(
            f, init, measurement_times,
            args=(
                pars["eta"],
                # pars["gamma"],
                pars["alpha"],
                # pars["mi"],
                # pars["theta"],
                # pars["theta_0"],
                # pars["sigma"],
                # pars["eta"],
                # pars["kappa_1"],
                # pars["kappa_2"]
            ))

        new_scale = sol[:, 4]
        return {"X_2": new_scale}

    # beta, gamma, alpha, mi, theta, theta_0, sigma, eta, kappa_1, kappa_2

    parameter_prior = Distribution(
        eta=RV("uniform", 0, 1),
        # gamma=RV("uniform", 0, 1),
        alpha=RV("uniform", 0, 1),
        # mi=RV("uniform", 0, 1),
        # theta=RV("uniform", 0, 1),
        # theta_0=RV("uniform", 0, 1),
        # sigma=RV("uniform", 0, 1),
        # eta=RV("uniform", 0, 1),
        # kappa_1=RV("uniform", 0, 1),
        # kappa_2=RV("uniform", 0, 1)
    )

    abc = ABCSMC(models=model,
                 parameter_priors=parameter_prior,
                 distance_function=distance,
                 population_size=5,
                 transitions=LocalTransition(k_fraction=.3),
                 eps=MedianEpsilon(500, median_multiplier=0.7),

                 )

    db_path = ("sqlite:///" +
               os.path.join("./", "test.db"))
    abc.new(db_path, {"X_2": measurement_data})
    h = abc.run(minimum_epsilon=0.1, max_nr_populations=3)
    print(*h.get_distribution(m=0, t=h.max_t))

    # fig = plt.figure(figsize=(10, 8))
    # for t in range(h.max_t + 1):
    #     ax = fig.add_subplot(3, np.ceil(h.max_t / 3), t + 1)
    #
    #     ax = plot_kde_2d(
    #         *h.get_distribution(m=0, t=t), "alpha", "eta",
    #         xmin=0, xmax=1, numx=200, ymin=0, ymax=1, numy=200, ax=ax)
    #     ax.set_title("Posterior t={}".format(t))
    #
    #     ax.legend()
    # fig.tight_layout()
    # plt.show()


if __name__ == '__main__':
    print(f"Calculating for R0 = {R_0}")
    main1()
