import numpy as np
import copy
import math
import random


# downward is parameter setting
train = np.loadtxt("gp-training-set.csv", delimiter=',')
train_set_size = len(train)
para_num = len(train[0])
mutation_rate = 0.1
fitness_check_num = 50
pop_num = 100
iter_num = 10000

class indivdual:
    def __init__(self):
        self.genome_list = []   # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # genome
        self.fitness = 0  # fitness value

    def __eq__(self, other):
        self.genome_list = copy.deepcopy(other.genome_list)
        self.fitness = other.fitness


def fitness(genome_list):
    fitness = 0
    for i in range(fitness_check_num):
        index = random.randint(0, train_set_size - 1)
        prediction = 0
        for j in range(para_num - 1):
            prediction += train[index][j] * genome_list[j]
        prediction -= genome_list[para_num - 1]
        if prediction >= 0:
            prediction = 1
        else:
            prediction = 0
        if prediction == train[index][para_num - 1]:
            fitness += 1
    return fitness


def initPopulation(pop, pop_num):
    for i in range(pop_num):
        ind = indivdual()
        for j in range(para_num):
            ind.genome_list.append(random.uniform(-2.0, 2.0))
        ind.fitness = fitness(ind.genome_list)
        pop.append(ind)


def selection(pop_num):
    # randomly select two parents for crossover
    return np.random.choice(pop_num, 2)


def crossover(parent1, parent2):
    child1, child2 = indivdual(), indivdual()
    for j in range(para_num):
        if random.randint(1, 4) > 3:
            child1.genome_list.append(0.9 * parent1.genome_list[j] + 0.1 * parent2.genome_list[j])
            child2.genome_list.append(0.1 * parent1.genome_list[j] + 0.9 * parent2.genome_list[j])
        else:
            child1.genome_list.append(0.9 * parent2.genome_list[j] + 0.1 * parent1.genome_list[j])
            child2.genome_list.append(0.1 * parent2.genome_list[j] + 0.9 * parent1.genome_list[j])
    child1.fitness = fitness(child1.genome_list)
    child2.fitness = fitness(child2.genome_list)
    return child1, child2


def mutation(pop):
    # ind = np.random.choice(pop)
    # ind.x = np.random.uniform(-
    ind = pop[random.randint(0, pop_num - 1)]
    ind.genome_list[random.randint(0, para_num - 1)] = random.uniform(-3.0, 3.0)
    ind.fitness = fitness(ind.genome_list)


def evolve():
    ###################################################
    # You may change these numbers to see what happens#
    ###################################################
    POP = []  # the population
    initPopulation(POP, pop_num)

    for it in range(iter_num):
        a, b = selection(pop_num)
        # print(1)
        # if np.random.random() < 0.75:  # 以0.75的概率进行交叉结合
        child1, child2 = crossover(POP[a], POP[b])
        new = sorted([POP[a], POP[b], child1, child2],
                     key=lambda ind: ind.fitness,
                     reverse=True)
        POP[a], POP[b] = new[0], new[1]

        if np.random.random() < mutation_rate:
            mutation(POP)

        POP.sort(key=lambda ind: ind.fitness, reverse=True)

        if fitness(POP[0].genome_list) == fitness_check_num & POP[0].fitness == fitness_check_num:
            print("fitness check satisfied")
            print(POP[0].genome_list)
            print(POP[0].fitness)
            return POP[0]
    print("max iter reached")
    print(POP[0].genome_list)
    print(POP[0].fitness)
    return POP[0]


final = evolve()


def test(genome_list):
    test_score = 0
    for i in range(train_set_size):
        # index = random.randint(0, train_set_size - 1)
        prediction_value = 0
        for j in range(para_num - 1):
            prediction_value += train[i][j] * genome_list[j]
        prediction_value -= genome_list[para_num - 1]
        if prediction_value >= 0:
            prediction_value = 1
        else:
            prediction_value = 0
        if prediction_value == train[i][para_num - 1]:
            test_score += 1
    return test_score


print(test(final.genome_list))
