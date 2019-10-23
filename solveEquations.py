import numpy as np
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
    return bin(n)[2:]

class Equation:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
        self.obj = 0

    def fObjective(self, x): # Tinh gia tri ham so (ax^2) + bx + c
        self.obj = self.a*(x**2) + self.b*x + self.c
        return self.obj

    def calFitness(self): # distance

    

class Genetic:
    def __init__(self, n, cr, mr):
        self.n = n # so luong chromosome trong population
        self.crossover_rate = cr
        self.mutation_rate = mr
        self.population = []
        self.min_x_val = 0 # gia tri nho nhat khi random gia tri ban dau cho x
        self.max_x_val = 30 # gia tri lon nhat khi random gia tri ban dau cho x

    def initPopulation(self):
        for i in range(len(self.n)):
            x = random.randint(self.min_x_val, self.max_x_val)
            gray = bin_dec_to_gray(x)
            gray = str(random.randint(0,1)) + gray # them bit sign
            self.population.append(gray)