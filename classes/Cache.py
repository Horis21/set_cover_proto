from classes.CacheEntry import CacheEntry
import time
import pandas as pd


class Cache:
    def __init__(self):
        self.entries = {}
        self.start_time = time.time()
        self.bounds = pd.DataFrame(columns=[    #     'time', 'lb', 'ub'
        ])

    def get_entry(self, df) -> CacheEntry:
        if self.entries.get(df) is None:
            self.entries[df] = CacheEntry()
        return self.entries[df]
    
    def save_bounds(self, lb, ub):
        self.bounds = pd.concat([self.bounds, pd.DataFrame([{
                                'time': time.time() - self.start_time,
                                'lb': lb,
                                'ub': ub
                            }])], ignore_index=True)
        
    def write_bounds(self, name):
        output_csv = 'anytime_bounds/' + name +  '_anytime_bounds.csv'
        with open(output_csv, 'w', newline='') as file:
                    self.bounds.to_csv(file, sep=' ', index=False, header=False)

    