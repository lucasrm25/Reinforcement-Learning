from deap import base
#from deap import benchmarks
from deap import creator
from deap import tools
import numpy as np
import operator
import random
from matplotlib import pyplot as plt


class Particle_Swarm_Opt:
    
    def __init__(self, fun, population=100, size=32, phi1=2.0, phi2=2.0, pmin=-6, pmax=6, smin=-3, smax=3):
        self.reset(fun, population, size, phi1, phi2, pmin, pmax, smin, smax)
        
        
    def reset(self, fun, population=100, size=32, phi1=2.0, phi2=2.0, pmin=-6, pmax=6, smin=-3, smax=3):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.__generate, size=size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.__updateParticle, phi1=phi1, phi2=phi2)
        self.toolbox.register("evaluate", fun)
        
        self.pop = self.toolbox.population(n=population)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields + ["best_global"]
        self.best = None
        
        
    def optimize(self, GEN=100, disp=True):
        for g in range(GEN):
            for part in self.pop:
                part.fitness.values = (self.toolbox.evaluate(part), )
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not self.best or self.best.fitness < part.fitness:
                    self.best = creator.Particle(part)
                    self.best.fitness.values = part.fitness.values
            for part in self.pop:
                self.toolbox.update(part, self.best)
    
            # Gather all the fitnesses in one list and print the stats
            self.logbook.record(gen=len(self.logbook), evals=len(self.pop), **self.stats.compile(self.pop), best_global=self.best.fitness.values[0])
            if disp: 
                print(self.logbook.stream)  
        return self.pop, self.logbook, self.best
     
   
    def __generate(self, size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        return part


    def __updateParticle(self, part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))    # Local best
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))         # Global best
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2))) # update speeds
        for i, speed in enumerate(part.speed):  # clip velocities
            if speed < part.smin:
                part.speed[i] = part.smin
            elif speed > part.smax:
                part.speed[i] = part.smax
        part[:] = list(map(operator.add, part, part.speed))  # update positions
        
    def plot(self):
        plt.plot( [l['gen'] for l in self.logbook], [l['max'] for l in self.logbook] ,label='max')
        plt.plot( [l['gen'] for l in self.logbook], [l['min'] for l in self.logbook] ,label='min')
        plt.plot( [l['gen'] for l in self.logbook], [l['avg'] for l in self.logbook] ,label='avg')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)