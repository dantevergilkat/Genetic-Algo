import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from City import City
from Fitness import Fitness

def createRoute(cityList): # create one random route passing all cities; What is cityList
    route = random.sample(cityList, len(cityList)) # random sampling without replacement (distribution (array-like), the number of samples)
    return route

def initialPopulation(popSize, cityList): # create initial population; population consists routes 
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).calFitness() # Transform each city into Fitness and compute calFitness
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True) # sorted for choosing the top 1 route (highest fitness score)

def selection(popRanked, eliteSize): # what is eliteSize
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum() # cumulative sum
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum() # roulette wheel, proportion of cumulative sum |?? need to change ??|
    
    for i in range(0, eliteSize): 
        selectionResults.append(popRanked[i][0])  # popRanked is an list, each element has 2 part: "Index" and "Fitness"
    for i in range(0, len(popRanked) - eliteSize): # why minus here?
        pick = 100*random.random() # a random number between 0 and 1
        for i in range(0, len(popRanked)): # len() of a dataframe is the number of rows # ?? Why i not j # There may be a pick again
            if pick <= df.iat[i,3]: # df.iat[a,b] get a value at row i and column 3 (cum_perc column), then compare it with a random number
                # !! cum_perc is high (fitNess is low) => The element of popRanked has higher change to be chosen !!
                selectionResults.append(popRanked[i][0]) # add chosen route index
                break # in each first loop, only choose 1 result; why need to do this?
    return selectionResults # just save index, the size of selectionResults is eliteSize + (len(popRanked) - eliteSize)

def matingPool(population, selectionResults): # should change name to generateChild
    matingpool = []
    for i in range(0, len(selectionResults)): # map index in selectionResults to city in population
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def childBreed(parent1, parent2): # Breed but stil keep all the cities
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * (len(parent1) - 1)) # !! Add -1 here, or else there are no cell in parent1
    geneB = int(random.random() * (len(parent1) - 1)) # !! Add -1 here
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene + 1): # !! Add +1 here
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def popBreed(matingpool, eliteSize): # what is eliteSize?
    children = []
    length = len(matingpool) - eliteSize # why minus here?
    pool = random.sample(matingpool, len(matingpool)) # sampling without replacement

    for i in range(0,eliteSize): # old children?
        children.append(matingpool[i])
    
    for i in range(0, length): # new children
        child = childBreed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children # len(children) = len(matingpool)

def childMutate(individual, mutationRate): # individual: is a route
    for swapped in range(len(individual)): # swapped: position of cell that being swapped
        if(random.random() < mutationRate):
            swapWith = int(random.random() * (len(individual) - 1)) # !! Add -1 here, swapWith: index to swap with
            
            cell1 = individual[swapped]
            cell2 = individual[swapWith]
            
            individual[swapped] = cell2
            individual[swapWith] = cell1
    return individual

def popMutate(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)): # mutate each rout in a population
        mutatedInd = childMutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen) # return a sorted list, each of elements has 2 parts : Index, Fitness
    selectionResults = selection(popRanked, eliteSize) # return a list of indices of chosen highest score cities, indices can be the same
    matingpool = matingPool(currentGen, selectionResults) # map indices in selectionResults to routes in population
    children = popBreed(matingpool, eliteSize)  # population of old children and new children
    nextGeneration = popMutate(children, mutationRate) # create next generation by mutate in final

    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial best distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final best distance: " + str(1 / rankRoutes(pop)[0][1])) # first element, get Fitness
    bestRouteIndex = rankRoutes(pop)[0][0] # first element, get Index
    bestRoute = pop[bestRouteIndex]
    return bestRoute

cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
