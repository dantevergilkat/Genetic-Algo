import pandas as pd
import numpy as np
import random
import sys

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
#s = np.sum(np.abs(a-b))
#print(s)

import struct
def binary(num):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))
#print(binary(float(1)))
'''def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    print(type(packed))

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    # 
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [c for c in packed]

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)'''

print(binary(263.3))
iee = binary(263.3)
print(type(iee))
gra1 = bin_dec_to_gray(iee)
print(gra1)
gra2 = gray2bin(gra1)
print(gra2)
#iee = np.asarray(list(iee))
print(iee)

def change_char(s, p, r):
    return s[:p]+r+s[p+1:]

from codecs import decode
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

res = bin_to_float(iee)
print(res)
print(type(res))

arr = np.array([1,2,3,4])
print(len(arr))
s = ''
for i in range(4):
    s += str(arr[i])

print(type(s))
print(int(7.4))