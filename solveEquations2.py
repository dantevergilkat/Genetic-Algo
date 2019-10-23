import numpy as np
import pandas as pd
import random
import struct
from codecs import decode

a = input("Nhap a:")
b = input("Nhap b: ")
c = input("Nhap c: ")

def binary(num): # Convert float to binary (IEEE 754) 
    s = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))
    return np.asarray(list(s))

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 4)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('!f', bf)[0]

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

class Equation:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
        self.obj = 0

    def fObjective(self, x): # Tinh gia tri ham so (ax^2) + bx + c
        self.obj = self.a*(x**2) + self.b*x + self.c
        return self.obj

    def calFitness(self, x): # 1/Hamming distance, x is a string of binary
        float_x = bin_to_float(x)
        l_side = self.fObjective(float_x) # ve trai
        r_side = 0 # ve phai
        l_side = binary(l_side) # gray code: numpy array
        r_side = binary(r_side) # gray code: numpy array

        # Hamming distance
        dist = np.sum(np.abs(a-b))

        return 1/(1 + dist)

    

class Genetic:
    def __init__(self, n, cr, mr, lower=0, upper=30, length=32):
        self.n = n # so luong chromosome trong population
        self.crossover_rate = cr
        self.mutation_rate = mr
        self.population = []
        self.min_x_val = lower # gia tri nho nhat khi random gia tri ban dau cho x
        self.max_x_val = upper # gia tri lon nhat khi random gia tri ban dau cho x
        self.defaultLen = length # the length of a chromosome, 32 bit lenght IEEE 754

    def initPopulation(self):
        for i in range(len(self.n)):
            #x = random.randint(self.min_x_val, self.max_x_val)
            x = random.uniform(self.min_x_val, self.max_x_val)
            # Convert to IEEE 754
            
            bin_x = binary(x)
            #gray = str(random.randint(0,1)) + gray # them bit sign
            self.population.append(bin_x)

    def rankVals(self, equation):
        fitnessResults = {}
        for i in range(self.n):
            fitnessResults[i] = equation.calFitness(self.population[i]) # Transform each city into Fitness and compute calFitness
        return sorted(fitnessResults.items(), reverse = True) # sorted for choosing the top 1 route (highest fitness score)

    def selection(self, popRanked, eliteSize=0): # what is eliteSize
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum() # cumulative sum
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum() # roulette wheel, proportion of cumulative sum |?? need to change ??|
        
        '''for i in range(0, eliteSize): 
            selectionResults.append(popRanked[i][0])'''  # popRanked is an list, each element has 2 part: "Index" and "Fitness"
        for i in range(0, len(popRanked)): # why minus here?
            pick = 100*random.random() # a random number between 0 and 1
            for i in range(0, len(popRanked)): # len() of a dataframe is the number of rows # ?? Why i not j # There may be a pick again
                if pick <= df.iat[i,3]: # df.iat[a,b] get a value at row i and column 3 (cum_perc column), then compare it with a random number
                    # !! cum_perc is high (fitNess is low) => The element of popRanked has higher change to be chosen !!
                    selectionResults.append(popRanked[i][1]) # add chosen route index
                    break # in each first loop, only choose 1 result; why need to do this?
        return selectionResults # save list of bit string

    def selectParents(self, selectionResults):
        parents = []
        for i in range(0, len(selectionResults)):
            pick = random.random()
            if pick < self.crossover_rate:
                parents.append((i, selectionResults[i])) # luu index va gene

        return parents
    
    def crossover(self, parents):
        C = []
        l = len(parents)
        for i in range(0, l):
            pick = random.randint(1, self.defaultLen-1)
            C.append(pick)

        for i in range(0, l):
            children = np.copy(parents[i][1])
            tmp = np.copy(parents[l-i-1][1])
            children[C[i]:] = tmp[C[i]:]
            self.population[parents[i][0]] = children

        return self.population
        
    def mutation(self):
        for i in range(self.n):
            pick = random.randint(1, self.defaultLen)
            '''rand_val = random.uniform(self.min_x_val, self.max_x_val)
            bin_rand_val = binary(rand_val)'''
            r_val = random.randint(0,1)
            self.population[i][pick] = r_val 

        return self.population

    def geneticAlgo(self, a, b, c, n_generation=50):
        self.initPopulation()
        equ = Equation(a, b, c)
        for i in range(n_generation):
            rankedVals = self.rankVals(equ)
            selectionResults = self.selection(rankedVals)
            parents = self.selectParents(selectionResults)
            self.crossover(parents)
            self.mutation()

        best_val_index = self.rankVals(self.population)[0][0]
        best_val = self.population[best_val_index]
        return best_val 
