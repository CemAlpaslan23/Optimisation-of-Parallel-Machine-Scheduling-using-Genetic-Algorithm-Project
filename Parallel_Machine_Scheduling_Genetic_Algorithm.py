import pandas as pd
import numpy as np
import random

# Read the dataset
df = pd.read_excel("Job Scheduling Volume Data Small.xlsx")

# Number of jobs which is equal to 20
job_count = df.shape[0]

# Number of machines
machine_count = 2

# Reading the necessary values from the dataset, due dates and quantites are necessary
materials = list(df['Material'])
product_types = list(df['Product Type'])
due_dates = list(df['Due Date'])
quantities = list(df['Quantity'])

# Processing times for jobs (1 minutes for each job on each machine)
processing_times = np.full((job_count, machine_count), 1)

# Setup times (2 minutes for each job on each machine)
setup_times = np.full((job_count, machine_count), 2)

# Weekly production capacities of the machines
machine_capacities = [150000, 10000]

# Weekly working times for the machines
machine_work_times = [240 * 7, 180 * 7]

# Create fitness function, our fitness function goal is to minimize the tardiness
def fitness_function(schedule):
    tardiness_total = 0
    for machine_index, machine_schedule in enumerate(schedule):
        completion_time = 0
        for job_index in machine_schedule:
            completion_time += quantities[job_index - 1] * processing_times[job_index - 1][machine_index] + setup_times[job_index - 1][machine_index]
            tardiness = max(completion_time - due_dates[job_index - 1], 0)  # Calculate tardiness
            tardiness_total += tardiness
    return tardiness_total

# Create initial population. Encoding part of the algorithm
# This function creates individuals (population size amount) with chromosome sequences covering jobs from 1 to 20.
def create_initial_population():
    population = []
    for _ in range(population_size):
        individual = np.arange(1, job_count + 1) # Genes are represented as randomly ordered integers between 1 and the number of jobs (i.e. 20).
        np.random.shuffle(individual)
        population.append(list(individual))
    return population

# Crossover operation
# A variable called crossover_point is created and randomly assigned a value between 0 and job_count - 1. This will be used to determine the crossover point. job_count represents the length of each parental chromosome.
# child1 is created for the first child. The expression parent1[:crossover_point] retrieves the genes from the beginning of parent1 to the crossover point. Then, genes present in parent2 but not in parent1 are added to the sequence after parent1[:crossover_point]. This forms the combined chromosome of the first child.
# child2 is created for the second child. Similarly, parent2[:crossover_point] retrieves the genes from the beginning of parent2 to the crossover point. Then, genes present in parent1 but not in parent2 are added to the sequence after parent2[:crossover_point]. This forms the combined chromosome of the second child.
def crossover(parent1, parent2):
    crossover_point = random.randint(0, job_count - 1)
    child1 = parent1[:crossover_point] + [job for job in parent2 if job not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [job for job in parent1 if job not in parent2[:crossover_point]]
    return child1, child2


# Mutation operation
# It makes a mutation by swapping the genes of two individuals in the chromosome sequence. It has a 10 % mutation rate
def mutate(individual):
    if random.random() < mutation_rate:
        mutation_point1, mutation_point2 = random.sample(range(job_count), 2)
        individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual

# Create next generation
# The function takes the current population as input. It generates the next generation of individuals based on the current population. It starts a loop that continues until the size of next_generation reaches the desired population size (population_size). In each iteration of the loop, two parents (parent1 and parent2) are randomly selected from the current population using random.sample(population, 2).
# The crossover operation is performed between the selected parents to create two children (child1 and child2). This is achieved by the crossover function. Each child undergoes mutation with a certain probability. This is done by the mutate function. After crossover and mutation, the fitness of each child is evaluated using a fitness function (fitness_function). If one child has a lower fitness than the other, and if it's valid according to certain criteria (checked by the is_valid function), it is added to the next generation.
# Once the next generation is filled with the desired number of individuals, it is returned as the output of the function
def create_next_generation(population):
    next_generation = []
    while len(next_generation) < population_size:
        parent1, parent2 = random.sample(population, 2)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        # Evaluate and check the limits of children
        if fitness_function(split_schedule(child1)) <= fitness_function(split_schedule(child2)):
            if is_valid(child1):
                next_generation.append(child1)
        else:
            if is_valid(child2):
                next_generation.append(child2)
    return next_generation


# Check validity function, we are checking the weekly production capacities of the machines and weekly working time limitations of the machines
def is_valid(individual):
    machine_workloads = [0] * machine_count
    for job_index in individual:
        job_index -= 1  # Set job index to start from 1
        assigned_machine = job_index % machine_count
        processing_time = processing_times[job_index][assigned_machine]
        job_quantity = quantities[job_index]
        # Check total production quantity of assigned job
        if machine_workloads[assigned_machine] + job_quantity > machine_capacities[assigned_machine]:
            return False  # Exceeds machine capacity
        machine_workloads[assigned_machine] += job_quantity
        # Check total time limit of assigned job
        if machine_workloads[assigned_machine] > machine_work_times[assigned_machine]:
            return False  # Exceeds machine work time
    return True


# Run genetic algorithm
# This function repeatedly evolves a population of candidate solutions over multiple generations, aiming to find the best solution (job schedule) that minimizes the fitness function.
# After creating each new generation, it evaluates the fitness of each individual in the population, sorts them based on their fitness values, and selects the best individual (with the lowest fitness value) as the best solution for that generation.
def genetic_algorithm():
    population = create_initial_population()
    for generation in range(generations):
        population = create_next_generation(population)
        population.sort(key=lambda x: fitness_function(split_schedule(x)))
        best_individual = population[0]
        best_fitness = fitness_function(split_schedule(best_individual))
        print()
        print(f"Best fitness value in generation {generation+1}: {best_fitness}")
        # Get and print job schedule of the best individual
        best_schedule = split_schedule(best_individual)
        print("Job schedule:")
        for machine_index, machine_schedule in enumerate(best_schedule):
            print(f"Machine {machine_index+1}:", machine_schedule)
        
    return best_schedule

# Assign jobs to machines
# This function deals with the assignment of a specific job to a specific machine and does not make a random assignment. In order to distribute the indexes and contents of the jobs evenly among the machines, it assigns the job index according to the number of machines. This allows jobs to be distributed more evenly between machines so that overall production time can be optimised.
def split_schedule(individual):
    schedule = [[] for _ in range(machine_count)]
    for i, job_index in enumerate(individual):
        schedule[i % machine_count].append(job_index)
    return schedule


# Create Gantt chart
# It shows jobs assigned to machines. It shows the starting time of the jobs, completion time of the jobs and the processing time of each job
def plot_gantt_chart(schedule):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab20(np.linspace(0, 1, job_count))  # Set color palette as tab20

    for machine_index, machine_schedule in enumerate(schedule):
        start_time = 0
        for job_index in machine_schedule:
            job_index -= 1  # Zero-based index
            processing_time = quantities[job_index] * processing_times[job_index][machine_index]
            setup_time = setup_times[job_index][machine_index]

            # Draw the processing time bar
            ax.barh(machine_index, processing_time, left=start_time, color=colors[job_index], alpha=0.8, edgecolor='black')
            ax.text(start_time + processing_time / 2, machine_index, f'Job {job_index + 1}', color='black', fontsize=8, ha='center', va='center')
            
            # Update start time to include processing time
            start_time += processing_time
            
            # Draw the setup time bar
            ax.barh(machine_index, setup_time, left=start_time, color='white', alpha=1.0, edgecolor='black')

            # Update start time to include setup time
            start_time += setup_time

    ax.set_yticks(range(machine_count))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(machine_count)])
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart')
    ax.invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Print the machine work times on each machine and the completion times of whole jobs on each machine
def print_machine_work_times(schedule):
    for machine_index, machine_schedule in enumerate(schedule):
        total_machine_time = 0
        job_times = []
        last_job_completion_time = 0  # To track the completion time of the last job on each machine
        for job_index in machine_schedule:
            job_index -= 1  # Zero-based index
            processing_time = quantities[job_index] * processing_times[job_index][machine_index]
            setup_time = setup_times[job_index][machine_index]
            total_time = processing_time + setup_time
            job_times.append(f"Job{job_index + 1}: {total_time} minutes")
            total_machine_time += total_time
            last_job_completion_time = total_machine_time  # Update the last job completion time
    
        print()
        print(f"Job times on Machine {machine_index + 1}: {', '.join(job_times)}")
        print()
        print(f"Machine {machine_index + 1} Last Job Completion Time: {last_job_completion_time} minutes")
        print()
        
# Function for printing start time, end time and tardiness values of each jobs      
def print_schedule_details(schedule):
    for machine_index, machine_schedule in enumerate(schedule):
        start_time = 0
        for job_index in machine_schedule:
            job_index -= 1  # Zero-based index
            processing_time = quantities[job_index] * processing_times[job_index][machine_index]
            setup_time = setup_times[job_index][machine_index]
            end_time = start_time + processing_time + setup_time
            tardiness = max(end_time - due_dates[job_index], 0)
            print(f"Machine {machine_index + 1}, Job {job_index + 1}: Start Time: {start_time}, End Time: {end_time}, Tardiness: {tardiness}")
            start_time = end_time  # Update start time for next job
            
# Optimize genetic algorithm parameters
def optimize_genetic_algorithm_parameters():
    best_parameters = None
    best_fitness = float('inf')
    
    population_sizes = [25, 50, 75, 100]
    mutation_rates = [0.05, 0.1, 0.15, 0.2]
    generations_list = [50, 75, 100, 125]
    
    for pop_size in population_sizes:
        for mut_rate in mutation_rates:
            for gens in generations_list:
                print(f"Testing with population size: {pop_size}, mutation rate: {mut_rate}, generations: {gens}")
                
                # Update global parameters
                global population_size, mutation_rate, generations
                population_size = pop_size
                mutation_rate = mut_rate
                generations = gens
                
                # Run genetic algorithm
                population = create_initial_population()
                for generation in range(generations):
                    population = create_next_generation(population)
                    population.sort(key=lambda x: fitness_function(split_schedule(x)))
                
                best_individual = population[0]
                current_fitness = fitness_function(split_schedule(best_individual))
                
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_parameters = (pop_size, mut_rate, gens)
    
    print(f"Optimal parameters: Population size: {best_parameters[0]}, Mutation rate: {best_parameters[1]}, Generations: {best_parameters[2]}")
    print(f"Best fitness value: {best_fitness}")
    return best_parameters

# Call the function to find the optimal parameters
best_parameters = optimize_genetic_algorithm_parameters()

# Now we can update your genetic algorithm with the optimal parameters, and run genetic algorithm and visualize the result
population_size, mutation_rate, generations = best_parameters
best_schedule = genetic_algorithm()
print_machine_work_times(best_schedule)
print_schedule_details(best_schedule)
plot_gantt_chart(best_schedule)
