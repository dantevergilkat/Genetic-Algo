import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
#from City import City
#from Fitness import Fitness

class Fitness:
    def __init__(self, route):
        self.route = route
        self.dist = 0
        self.fit = 0.0 # fitness = 1/distance
    
    def calDistance(self):
        if self.dist ==0:
            totalDist = 0
            #print(self.route)
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                if i+1 >= len(self.route):
                    toCity = self.route[0]
                else:
                    toCity = self.route[i+1]
                #print('From city ' + str(fromCity) + ', to city ' + str(toCity))
                totalDist += distMatrix[fromCity][toCity]
            self.dist = totalDist

        return self.dist
    
    def calFitness(self):
        if self.fit == 0: # why need to check == 0 here (0 means init?)
            self.fit = 1 / float(self.calDistance())

        return self.fit

'''def createRoute(cityList): # create one random route passing all cities; What is cityList # !! FIX THIS !!
    lst = []
    lst.append(0)
    last = 0
    for i in range(nCity-1):
        index = -1
        while index == -1 or last+1 == len(lst):
            print(lst, ' ', last)
            index = random.randint(0, len(edgeList[lst[last]]) - 1)
            if edgeList[lst[last]][index][0] not in lst:
                #print('Yeah')
                lst.append(edgeList[lst[last]][index][0])
        last += 1
    lst.append(0)
    print('Route: ', lst)
    return lst'''

def createRoute(cityList, route=[0]):
    top = route[len(route) - 1]
    flag = 0
    back = 0
    notVisitedNeighbor = []
    for (neighbor,_) in edgeList[top]:
        if neighbor == 0:
            back = 1
        if neighbor not in route:
            flag = 1
            notVisitedNeighbor.append(neighbor)

    if len(cityList) == len(route) and back == 1:
        ret =  route[:]
        ret.append(0)
        return ret

    if flag == 0:
        return -1

    while notVisitedNeighbor:
        index = random.randint(0, len(notVisitedNeighbor) - 1)
        sroute = route[:]
        sroute.append(notVisitedNeighbor[index])
        del notVisitedNeighbor[index]

        ret = createRoute(cityList, sroute)
        if ret != -1:
            break

    return ret



def initialPopulation(popSize, cityList): # create initial population; population consists routes 
    population = []

    for i in range(0, popSize):
        print('[+] ', i)
        route = createRoute(cityList)
        population.append(route)
        print('Route: ', route)
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
    
    #geneA = int(random.random() * (len(parent1) - 1)) # !! Add -1 here, or else there are no cell in parent1
    #geneB = int(random.random() * (len(parent1) - 1)) # !! Add -1 here
    geneA = random.randint(1, len(parent1)-2) # CHANGE
    geneB = random.randint(1, len(parent1)-2) # CHANGE
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    childP1.append(0)
    for i in range(startGene, endGene + 1): # !! Add +1 here
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    child.append(0)
    return child

def checkRouteExisted(route):
    for i in range(len(route) - 1):
        fromCity = route[i]
        toCity = route[i+1]
        if distMatrix[fromCity][toCity] == 0:
            return False

    return True

def popBreed(matingpool, eliteSize): # what is eliteSize?
    children = []
    length = len(matingpool) - eliteSize # why minus here?
    pool = random.sample(matingpool, len(matingpool)) # sampling without replacement

    for i in range(0,eliteSize): # old children?
        children.append(matingpool[i])
    
    for i in range(0, length): # new children
        child = childBreed(pool[i], pool[len(matingpool)-i-1])
        # !!check child existed!!
        if checkRouteExisted(child) == True:
            children.append(child)
        else:
            children.append(matingpool[i])
    return children # len(children) = len(matingpool)

def childMutate(individual, mutationRate): # individual: is a route
    for swapped in range(1, len(individual)-1): # swapped: position of cell that being swapped
        if(random.random() < mutationRate):
            #swapWith = int(random.random() * (len(individual) - 1)) # !! Add -1 here, swapWith: index to swap with
            swapWith = random.randint(1, len(individual)-2)
            
            cell1 = individual[swapped]
            cell2 = individual[swapWith]
            
            individual[swapped] = cell2
            individual[swapWith] = cell1
    '''if individual == [0,4,1,3,2,0]:
        print('Mutate prob')'''
    return individual

def popMutate(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)): # mutate each rout in a population
        c_pop = population[ind][:] 
        mutatedInd = childMutate(c_pop, mutationRate)
        if checkRouteExisted(mutatedInd) == True:
            if mutatedInd == [0,4,1,3,2,0]:
                print(mutatedInd)
            mutatedPop.append(mutatedInd)
        else:
            '''if population[ind] == [0,4,1,3,2,0]:
                print('Mutate prob')'''
            mutatedPop.append(population[ind])
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen) # return a sorted list, each of elements has 2 parts : Index, Fitness
    selectionResults = selection(popRanked, eliteSize) # return a list of indices of chosen highest score cities, indices can be the same
    matingpool = matingPool(currentGen, selectionResults) # map indices in selectionResults to routes in population
    children = popBreed(matingpool, eliteSize)  # population of old children and new children
    nextGeneration = popMutate(children, mutationRate) # create next generation by mutate in final

    '''if [0,4,1,3,2,0] in children:
        print('Children False')
    if [0,1,3,2,4,0] in children:
        print('Children True')'''
    if [0,4,1,3,2,0] in nextGeneration:
        print('Children False')

    #return nextGeneration
    return children

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial best distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        #print('[+] Generation: ', i)
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final best distance: " + str(1 / rankRoutes(pop)[0][1])) # first element, get Fitness
    bestRouteIndex = rankRoutes(pop)[0][0] # first element, get Index
    bestRoute = pop[bestRouteIndex]
    return bestRoute

'''nCity = 4
nEdge = 6
cityList = []
edgeList = []

for i in range(0,4):
    cityList.append(i)

# Init edges
edgeList = [[(1,10), (2,15), (3,20)],
            [(0,10), (2,35), (3,25)],
            [(0,15), (1,35), (3,30)],
            [(0,20), (1,25), (2,30)]]
distMatrix = [[0, 10, 15, 20],
              [10, 0, 35, 25],
              [15, 35, 0, 30],
              [20, 25, 30, 0]]'''

nCity = 5 
nEdge = 8
cityList = []
edgeList = []

for i in range(0,5):
    cityList.append(i)

# Init edges
edgeList = [[(1,2), (3,12), (4,5)],
            [(0,2), (2,4), (3,8)],
            [(1,4), (3,3), (4,3)],
            [(0,12), (1,8), (2,3), (4,10)],
            [(0,5), (2,3), (3,10)]]
distMatrix = [[0, 2, 0, 12, 5],
              [2, 0, 4, 8, 0],
              [0, 4, 0, 3, 3],
              [12, 8, 3, 0, 10],
              [5, 0, 3, 10, 0]]

bestRoute = geneticAlgorithm(population=cityList, popSize=10, eliteSize=2, mutationRate=0.01, generations=500)
print('Best route: ', bestRoute)
