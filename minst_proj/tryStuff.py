# file to try things

import numpy as np 

s = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

if __name__ == "__main__":
    print(f"The array is {s}")

    # now trying to do some slices
    j = s[1:4]
    print(j)
    print(f"This is just using the indexes of the array \n {s[[1,2,3]]}")