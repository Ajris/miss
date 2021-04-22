import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv

beta = 0.3
gamma = 0.0714
alpha = 0.0008
mi = 0.01
theta = 0
theta_0 = 0.0
sigma = 0.1923
eta = 0.005

kappa_1 = 0.7
kappa_2 = 0.7

psi = mi + theta + theta_0
phi = sigma + eta + mi
epsilon = alpha + gamma + mi

R_0 = beta * sigma * (mi + theta_0) / (phi * epsilon * psi)

def get_from_csv():
    with open('data/full_global_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        data = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            if row['Country_code'] == 'PL':
                data.append(float(row['New_cases']))
            line_count += 1
        print(f'Processed {line_count} lines.')
        print(len(data))
        return data



def main():
    def model(z, t, u, w, h, v, q, r, d):
        u, w, h, v, q, r, d = z

        dudt = mi - beta * u * w - (theta + mi) * u + theta_0 * h
        dhdt = theta * u - (mi + theta_0) * h
        dvdt = beta * u * w - (sigma + eta + mi) * v
        dwdt = sigma * v - (alpha + gamma + mi) * w
        dqdt = eta * v + alpha * w - (gamma + mi) * q
        drdt = kappa_1 * gamma * w + kappa_2 * gamma * q - mi * r
        dddt = (1 - kappa_1) * gamma * w + (1 - kappa_2) * gamma * q - mi * d

        dNdt = [dudt, dwdt, dhdt, dvdt, dqdt, drdt, dddt]
        return dNdt
    measurement_data = np.array(get_from_csv()) / 40000000
    measurement_times = np.arange(len(measurement_data))

    n = 1460
    t = np.linspace(0, 730, n)

    u = 39999999 / 40000000
    w = 0.0
    h = 0
    v = 1 / 40000000
    q = 0
    r = 0
    d = 0
    z0 = [u, w, h, v, q, r, d]

    u_t = np.empty_like(t)
    w_t = np.empty_like(t)
    h_t = np.empty_like(t)
    v_t = np.empty_like(t)
    q_t = np.empty_like(t)
    r_t = np.empty_like(t)
    d_t = np.empty_like(t)

    u_t[0], w_t[0], h_t[0], v_t[0], q_t[0], r_t[0], d_t[0] = u, w, h, v, q, r, d

    for i in range(1, n):
        tspan = [t[i - 1], t[i]]
        z = odeint(model, z0, tspan, args=(u, w, h, v, q, r, d))

        u_t[i], w_t[i], h_t[i], v_t[i], q_t[i], r_t[i], d_t[i] = z[1]
        z0 = z[1]

    n_t = np.empty_like(t)
    for i in range(1, n):
        n_t[i] = (u_t[i] + w_t[i] + h_t[i] + v_t[i] + q_t[i] + r_t[i] + d_t[i])

    # plt.plot(t, n_t, 'b:', label='n(t)')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.scatter(measurement_times,measurement_data )

    # plt.plot(t, u_t, '-', color=("blue"), label='Susceptible')
    # plt.plot(t, h_t, 'r-', label='Quarantined-StayAtHome')
    # plt.plot(t, v_t, '-', color=("orangered"), label='Exposed')
    # plt.plot(t, w_t, '-', color=("orange"), label='Infected')
    plt.plot(t, q_t, 'r--', label='Quarantined-Isolated')
    # plt.plot(t, r_t, '-', color=("purple"), label='Recovered')
    # plt.plot(t, d_t, '-', color=("green"), label='Dead')
    plt.ylabel('Part of population')
    plt.xlabel('Time(Days)')
    plt.xlim(right = 500)
    plt.legend(loc='best')
    plt.show()
    plt.ion()



if __name__ == '__main__':
    print(f"Calculating for R0 = {R_0}")
    main()