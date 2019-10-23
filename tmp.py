import pandas as pd
import numpy as np
import random

df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
df2['cum_sum'] = df2.a.cumsum()
df2['cum_perc'] = 100*df2.cum_sum/df2.a.sum()

# print(df2)
# print(df2.dtypes)
# print(random.random())
# print(len(df2))

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
        #b.append(b[-1] ^ nextb)

    b = np.asarray(b)
    return b

for i in range(10):
    gray = bin_dec_to_gray(i)
    '''bin_num = input("Enter binary number: ")
    print('Gray code: ', bin_dec_to_gray(bin_num))'''
    #gray = list(gray)
    #print(list(gray))
    print('Gray code: ', gray, ", Dec code: ", gray2bin(gray))

a = np.array([1,0,1,0])
b = np.array([0,0,0,1])
b = np.insert(b, 0, random.randint(0,1))
print(b)
s = np.sum(np.abs(a-b))
print(s)