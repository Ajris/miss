import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

rate_susceptible_from_spreaders = 0.3
rate_from_infected_to_recovery_or_dead = 0.0714
duration_of_infectiousness = 1.0 / rate_from_infected_to_recovery_or_dead
rate_spreaders_to_isolated = 0.0
rate_natural_birth_and_death = 0.01
rate_susceptible_to_quarantine_or_stay_at_home = 0.0
rate_people_stay_at_home_due_to_ineffectiveness_of_home_quarantine = 0.0
rate_people_completed_incubation_become_infected = 0.1923
duration_of_incubation = 1.0 / rate_people_completed_incubation_become_infected
rate_exposed_to_isolated = 0.1

rate_infectious_recover = 0.9
rate_infectious_die = 1 - rate_infectious_recover
rate_isolated_infected_recover = 0.9
rate_isolated_infected_die = 1 - rate_isolated_infected_recover

psi = rate_natural_birth_and_death + rate_susceptible_to_quarantine_or_stay_at_home + rate_people_stay_at_home_due_to_ineffectiveness_of_home_quarantine
phi = rate_people_completed_incubation_become_infected + rate_natural_birth_and_death + rate_exposed_to_isolated
epsilon = rate_spreaders_to_isolated + rate_from_infected_to_recovery_or_dead + rate_natural_birth_and_death

basic_reproduction_number = rate_susceptible_from_spreaders * rate_people_completed_incubation_become_infected * (rate_natural_birth_and_death + rate_people_stay_at_home_due_to_ineffectiveness_of_home_quarantine) / (phi * epsilon * psi)

number_of_days = 730
number_of_points_on_chart = number_of_days * 2
period_interval = np.linspace(0, number_of_days, number_of_points_on_chart)

start_susceptible_population = 0.9
start_spreader_population = 0.04
start_quarantined_stay_at_home_population = 0.0
start_exposed_population = 0.06
start_quarantined_isolated_population = 0.0
start_recovered_population = 0.0
start_dead_population = 0.0

def model(current_population, t):
    current_susceptible_population, current_spreader_population, current_quarantined_stay_at_home_population, current_exposed_population, current_quarantined_isolated_population, current_recovered_population, current_dead_population = current_population

    increase_susceptible_population = rate_natural_birth_and_death - rate_susceptible_from_spreaders * current_susceptible_population * current_spreader_population - (rate_susceptible_to_quarantine_or_stay_at_home + rate_natural_birth_and_death) * current_susceptible_population + rate_people_stay_at_home_due_to_ineffectiveness_of_home_quarantine * current_quarantined_stay_at_home_population
    increase_spread_population = rate_people_completed_incubation_become_infected * current_exposed_population - (rate_spreaders_to_isolated + rate_from_infected_to_recovery_or_dead + rate_natural_birth_and_death) * current_spreader_population
    increase_quarantined_home_population = rate_susceptible_to_quarantine_or_stay_at_home * current_susceptible_population - (rate_natural_birth_and_death + rate_people_stay_at_home_due_to_ineffectiveness_of_home_quarantine) * current_quarantined_stay_at_home_population
    increase_exposed_population = rate_susceptible_from_spreaders * current_susceptible_population * current_spreader_population - (rate_people_completed_incubation_become_infected + rate_exposed_to_isolated + rate_natural_birth_and_death) * current_exposed_population
    increase_quarantined_isolated_population = rate_exposed_to_isolated * current_exposed_population + rate_spreaders_to_isolated * current_spreader_population - (rate_from_infected_to_recovery_or_dead + rate_natural_birth_and_death) * current_quarantined_isolated_population
    increase_recovered_population = rate_infectious_recover * rate_from_infected_to_recovery_or_dead * current_spreader_population + rate_isolated_infected_recover * rate_from_infected_to_recovery_or_dead * current_quarantined_isolated_population - rate_natural_birth_and_death * current_recovered_population
    increase_dead_population = (1 - rate_infectious_recover) * rate_from_infected_to_recovery_or_dead * current_spreader_population + (1 - rate_isolated_infected_recover) * rate_from_infected_to_recovery_or_dead * current_quarantined_isolated_population - rate_natural_birth_and_death * current_dead_population

    all_increases = [increase_susceptible_population, increase_spread_population, increase_quarantined_home_population, increase_exposed_population, increase_quarantined_isolated_population, increase_recovered_population, increase_dead_population]
    return all_increases


def main():
    current_population = [start_susceptible_population, start_spreader_population, start_quarantined_stay_at_home_population, start_exposed_population, start_quarantined_isolated_population, start_recovered_population, start_dead_population]

    susceptible_population = np.empty_like(period_interval)
    spreader_population = np.empty_like(period_interval)
    quarantined_stay_at_home_population = np.empty_like(period_interval)
    exposed_population = np.empty_like(period_interval)
    quarantined_isolated_population = np.empty_like(period_interval)
    recovered_population = np.empty_like(period_interval)
    dead_population = np.empty_like(period_interval)

    susceptible_population[0], spreader_population[0], quarantined_stay_at_home_population[0], exposed_population[0], quarantined_isolated_population[0], recovered_population[0], dead_population[0] = start_susceptible_population, start_spreader_population, start_quarantined_stay_at_home_population, start_exposed_population, start_quarantined_isolated_population, start_recovered_population, start_dead_population

    for i in range(1, number_of_points_on_chart):
        current_interval = [period_interval[i - 1], period_interval[i]]
        calculated_population_for_interval = odeint(model, current_population, current_interval)

        susceptible_population[i], spreader_population[i], quarantined_stay_at_home_population[i], exposed_population[i], quarantined_isolated_population[i], recovered_population[i], dead_population[i] = calculated_population_for_interval[1]
        current_population = calculated_population_for_interval[1]

    sum_of_all_population = np.empty_like(period_interval)
    for i in range(1, number_of_points_on_chart):
        sum_of_all_population[i] = (susceptible_population[i] + spreader_population[i] + quarantined_stay_at_home_population[i] + exposed_population[i] + quarantined_isolated_population[i] + recovered_population[i] + dead_population[i])

    # plt.plot(period_interval, sum_of_all_population, 'b:', label='n(t)')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(period_interval, susceptible_population, 'b-', label='Susceptible')
    plt.plot(period_interval, spreader_population, 'y', label='Infectiously Infected')
    plt.plot(period_interval, quarantined_stay_at_home_population, 'r-', label='Quarantined-StayAtHome')
    plt.plot(period_interval, exposed_population, 'g-', label='Exposed')
    plt.plot(period_interval, quarantined_isolated_population, 'r--', label='Quarantined-Isolated')
    plt.plot(period_interval, recovered_population, 'g--', label='Recovered')
    plt.plot(period_interval, dead_population, 'g:', label='Dead')
    plt.ylabel('Part of population')
    plt.xlabel('Time(Days)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print(f"Calculating for basic_reproduction_number = {basic_reproduction_number}")
    main()
