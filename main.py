import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

beta = 0.3
gamma = 0.0714
alpha = 0
mi = 0.01
theta = 0
theta_0 = 0.0
sigma = 0.1923
ni = 0.1

kappa_1 = 0.9
kappa_2 = 0.9

psi = mi + theta + theta_0
phi = sigma + mi + ni
epsilon = alpha + gamma + mi

R_0 = beta * sigma * (mi + theta_0) / (phi * epsilon * psi)


def main():
    def model(z, t, u, w, h, v, q, r, d):
        u = z[0]
        w = z[1]
        h = z[2]
        v = z[3]
        q = z[4]
        r = z[5]
        d = z[6]
        dudt = mi - beta * u * w - (theta + mi) * u + theta_0 * h
        dhdt = theta * u - (mi + theta_0) * h
        dvdt = beta * u * w - (sigma + ni + mi) * v
        dwdt = sigma * v - (alpha + gamma + mi) * w
        dqdt = ni * v + alpha * w - (gamma + mi) * q
        drdt = kappa_1 * gamma * w + kappa_2 * gamma * q - mi * r

        dddt = (1 - kappa_1) * gamma * w + (1 - kappa_2) * gamma * q - mi * d

        dNdt = [dudt, dwdt, dhdt, dvdt, dqdt, drdt, dddt]

        return dNdt


    # number of time points
    n = 1460

    # time points
    t = np.linspace(0, 730, n)

    # step input
    u = 0.9
    w = 0.04
    h = 0
    v = 0.06
    q = 0
    r = 0
    d = 0
    # initial condition
    z0 = [u, w, h, v, q, r, d]

    # store solution

    u_t = np.empty_like(t)
    w_t = np.empty_like(t)
    h_t = np.empty_like(t)
    v_t = np.empty_like(t)
    q_t = np.empty_like(t)
    r_t = np.empty_like(t)
    d_t = np.empty_like(t)

    u_t[0] = u
    w_t[0] = w
    h_t[0] = h
    v_t[0] = v
    q_t[0] = q
    r_t[0] = r
    d_t[0] = d

    # solve ODE
    for i in range(1, n):
        # span for next time step
        tspan = [t[i - 1], t[i]]
        # solve for next step
        z = odeint(model, z0, tspan, args=(u, w, h, v, q, r, d))
        # store solution for plotting

        u_t[i] = z[1][0]
        w_t[i] = z[1][1]
        h_t[i] = z[1][2]
        v_t[i] = z[1][3]
        q_t[i] = z[1][4]
        r_t[i] = z[1][5]
        d_t[i] = z[1][6]
        # next initial condition
        z0 = z[1]

    n_t = np.empty_like(t)
    for i in range(1, n):
        n_t[i] = (u_t[i] + w_t[i] + h_t[i] + v_t[i] + q_t[i] + r_t[i] + d_t[i])

    # plot results
    # plt.plot(t, n_t, 'b:', label='n(t)')


    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(t, u_t, 'b-', label='Susceptible')
    plt.plot(t, h_t, 'r-', label='Quarantined-StayAtHome')
    plt.plot(t, v_t, 'g-', label='Exposed')
    plt.plot(t, w_t, 'y', label='Infected')
    plt.plot(t, q_t, 'r--', label='Quarantined-Isolated')
    plt.plot(t, r_t, 'g--', label='Recovered')
    plt.plot(t, d_t, 'g:', label='Dead')
    plt.ylabel('Part of population')
    plt.xlabel('Time(Days)')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    print(f"Calculating for R0 = {R_0}")
    main()