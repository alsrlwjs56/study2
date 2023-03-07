import numpy as np
from lib.models import sum_mat

def main():
    mingi = sum_mat()
    print(mingi.c)
    print(mingi.sum(1, 2))
    print(mingi.c)
    dnchoi = sum_mat()
    print(dnchoi.c)
    print(dnchoi.sum(10, 20))
    print(dnchoi.c)
    
if __name__=="__main__":
    main()