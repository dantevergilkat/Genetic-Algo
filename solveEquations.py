import numpy as np
import pandas as pd
import random


a = input("Nhap a:")
b = input("Nhap b: ")
c = input("Nhap c: ")

def bin_dec_to_gray(n): # chuyen thap phan, nhi phan sang gray code
    """Convert Binary to Gray codeword and return it."""
    if isinstance(n, str):
        n = int(n, 2) # convert to int
    n ^= (n >> 1)
 
    # bin(n) returns n's binary representation with a '0b' prefixed
    # the slice operation is to remove the prefix
    gray = bin(n)[2:]
    gray = [int(char) for char in gray]
    gray = np.asarray(gray)
    return gray

def gray2bin(bits):
    b = [bits[0]]
    for nextb in bits[1:]:
        b.append(b[-1] ^ nextb)
    b = np.asarray(b)
    return b

class Equation:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
        self.obj = 0

    def fObjective(self, x): # Tinh gia tri ham so (ax^2) + bx + c
        self.obj = self.a*(x**2) + self.b*x + self.c
        return self.obj

    def calFitness(self, x): # 1/Hamming distance
        l_side = self.fObjective(x) # ve trai
        r_side = 0 # ve phai
        l_side = bin_dec_to_gray(l_side) # gray code: numpy array
        r_side = bin_dec_to_gray(r_side) # gray code: numpy array

        # Hamming distance
        dist = np.sum(np.abs(a-b))

        return 1/(1 + dist)

    

class Genetic:
    def __init__(self, n, cr, mr):
        self.n = n # so luong chromosome trong population
        self.crossover_rate = cr
        self.mutation_rate = mr
        self.population = []
        self.min_x_val = 0 # gia tri nho nhat khi random gia tri ban dau cho x
        self.max_x_val = 30 # gia tri lon nhat khi random gia tri ban dau cho x
        self.defaultLen = 10 # the length of a chromosome

    def initPopulation(self):
        for i in range(len(self.n)):
            #x = random.randint(self.min_x_val, self.max_x_val)
            x = random.uniform(self.min_x_val, self.max_x_val)
            # Convert to IEEE 754
            
            gray = bin_dec_to_gray(x)
            #gray = str(random.randint(0,1)) + gray # them bit sign
            gray = np.insert(gray, 0, random.randint(0,1)) # prepend sign value 0: am, 1: duong
            self.population.append(gray)

    def selection(self, popRanked, eliteSize): # what is eliteSize
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Fitness"])
        df['cum_sum'] = df.Fitness.cumsum() # cumulative sum
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum() # roulette wheel, proportion of cumulative sum |?? need to change ??|
        
        '''for i in range(0, eliteSize): 
            selectionResults.append(popRanked[i][0])'''  # popRanked is an list, each element has 2 part: "Index" and "Fitness"
        for i in range(0, len(popRanked)): # why minus here?
            pick = 100*random.random() # a random number between 0 and 1
            for i in range(0, len(popRanked)): # len() of a dataframe is the number of rows # ?? Why i not j # There may be a pick again
                if pick <= df.iat[i,2]: # df.iat[a,b] get a value at row i and column 3 (cum_perc column), then compare it with a random number
                    # !! cum_perc is high (fitNess is low) => The element of popRanked has higher change to be chosen !!
                    selectionResults.append(popRanked[i][0]) # add chosen route index
                    break # in each first loop, only choose 1 result; why need to do this?
        return selectionResults # save list of bit

    def selectParents(self, selectionResults):
        parents = []
        for i in range(0, len(selectionResults)):
            pick = random.random()
            if pick < self.crossover_rate:
                parents.append(selectionResults[i])

        return parents
    
    def crossover(self, parents):
        C = []
        for i in range(0, len(parents)):
            pick = random.randint(1, self.defaultLen-1)
            C.append(pick)

        