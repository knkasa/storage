# Synthetic Difference in Difference.

import pdb  
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from synthdid.synthdid import Synthdid as sdid
from synthdid.get_data import quota, california_prop99
import multiprocessing
import os


class multi_class():
    def __init__(self, cols, *args, **kwargs):
        try:

            self.df = california_prop99()
            print(f"Number of cores:{os.environ['NUMBER_OF_PROCESSORS']}")

            with multiprocessing.Pool( 
                    processes=int(os.environ['NUMBER_OF_PROCESSORS'])//2,  # optional.  default=max number of cores.
                    initializer=self.initializer,  # optional.  Function to run when each processes is started.
                    maxtasksperchild=len(cols),  # optional
                    initargs=None,  # optional.  You can send a tuple to the initializer function.
                ) as pool:

                res = pool.map(self.run_synthetic, cols)

            print(res)
            #pdb.set_trace()

        except Exception as e:
            raise Exception(e)

    def run_synthetic(self, col):
        california_estimate = sdid( self.df, unit="State", time="Year", treatment="treated", outcome=col).fit().vcov(method='placebo')
        print(f"ID:{os.getpid()}  {california_estimate.summary().summary2} ")

    def initializer(self):
        print(f"Worker {os.getpid()} is initialized.")


def main():
    cols = ["PacksPerCapita", "PacksPerCapita", "PacksPerCapita", "PacksPerCapita"]
    multi_class(cols)

if __name__ == '__main__':
    main()




