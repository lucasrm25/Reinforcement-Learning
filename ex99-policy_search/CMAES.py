from deap import base
from deap import creator
from deap import tools
from deap import cma
from deap import algorithms
import numpy as np
from matplotlib import pyplot as plt
#import multiprocessing


class CMAES:
    
    def __init__(self, fun, lambda_=20, size=36):
        self.reset(fun, lambda_, size)
        
        
    def reset(self, fun, lambda_=20, size=36):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        #pool = multiprocessing.Pool(8)
        #self.toolbox.register("map", pool.map)
        self.toolbox.register("evaluate", fun)
        
        N=size
        strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=lambda_)
        self.toolbox.register("generate", strategy.generate, creator.Individual)
        self.toolbox.register("update", strategy.update)
        
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)    
    
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields        
        
        
    def optimize(self, GEN=100, disp=True):
        pop, logbook = algorithms.eaGenerateUpdate(self.toolbox, ngen=GEN, stats=self.stats, halloffame=self.hof, verbose=disp)   
        
        for i in range(len(logbook)):
            self.logbook.record(**logbook[i]) 
            
        return pop, self.logbook
     
        
    def plot(self):
        plt.plot( range(len(self.logbook)), [l['max'] for l in self.logbook] ,label='max')
        plt.plot( range(len(self.logbook)), [l['min'] for l in self.logbook] ,label='min')
        plt.plot( range(len(self.logbook)), [l['avg'] for l in self.logbook] ,label='avg')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        
        
        
        