import numpy as np

class StatisticTM:
    def __init__(self, prefix):
        self.prefix = prefix
        self.tms=[]
        self.comments=None

    def add_tm(self, tm):
        self.tms.append(tm)
    def add_comments(self, str):
        if self.comments is None:
            self.comments = str
        else:
            self.comments = self.comments + ":" + str
    
    def __str__(self):
        return (f"{self.prefix:22} time: min:{min(self.tms):6.3f}, max:{max(self.tms):6.3f}, mean:{np.average(self.tms):6.3f}, total: {len(self.tms)}, comments: {self.comments}")

# unit test.
# stm = StatisticTM("Model SD21")
# stm.add_tm(20.000001)
# stm.add_tm(10000.02222)
# stm.add_tm(30)
# print(stm)
# stm = StatisticTM("Model SD22xxxx")
# stm.add_tm(20)
# stm.add_tm(10.11111111)
# stm.add_tm(30)
# print(stm)