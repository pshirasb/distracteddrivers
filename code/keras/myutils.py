import time
import sys
import re
import numpy as np
import pandas as pd


def confusion_matrix(pred, target):
    """ Returns a Confusion Matrix as a numpy array
        using pandas.
    """
    return pd.crosstab(pred, 
                target, 
                rownames=['Predicted'], 
                colnames=['True'], 
                margins=True)

# Columns are target, rows are preds
def table(pred, target, num_classes = 10):
    """ Returns a Confusion Matrix as a numpy array
    """
    
    # pred and target must have the shape [num_examples, 1]
    if pred.shape != target.shape:
        print pred.shape
        print target.shape
        raise ValueError("pred and target must have the same shape")

    retn = np.zeros([num_classes, num_classes], dtype=np.int64)
    d = np.concatenate((pred, target), axis = 1)
    d_size = len(d)
    for i in xrange(num_classes):
        for j in xrange(num_classes):
            matches = [x for x in xrange(d_size) 
                        if d[x,0] == i and d[x,1] == j]
            retn[i,j] = len(matches)
    return retn


def benchmark(func):
    """ A decorator that prints function execution time 
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        retn = func(*args, **kwargs)
        elapsed = time.time() - start
        print "[" + func.__name__ + ": %.4f sec]" % elapsed
        return retn
        
    return wrapper

class ProgressBar():
    """ CLI ProgressBar object
    """
    def __init__(self):
        self.count = 0
        self.max_count = 0
        self.current_time = 0
        self.start_time = 0
        self.txt = ""     # Extra text at the end of the line
        self.pretxt = ""  # Extra text at the start of the line
        
    def start(self, max_count):
        self.count = 0
        self.max_count = max_count
        self.start_time = time.time()
        self.current_time = self.start_time
        self.txt = ""     # Extra text at the end of the line
        self.pretxt = ""  # Extra text at the start of the line

    def step(self, num_steps = 1, txt = "", pretxt = ""):
        self.txt = txt
        self.pretxt = pretxt
        self.count += num_steps
        self.__show_progess_bar()

    def stop(self):
        elapsed = time.time() - self.start_time
        #print "[elapsed: %.4f sec]" % elapsed
        self.count = 0
        self.max_count = 0
        self.current_time = 0
        self.start_time = 0
        self.txt = ""

    def __show_progess_bar(self):
        
        bar_width    = 30
        bar_filled   = int(round(bar_width * self.count / 
                            float(self.max_count)))
        bar_unfilled = bar_width - bar_filled
        bar_percent  = (100 * (self.count / float(self.max_count)))

        elapsed = time.time() - self.start_time
        sec_per_count = float(elapsed) / self.count
        eta = (self.max_count - self.count) * sec_per_count
        
        eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta))
        eta_str = re.sub("00[hm] ", "    ", eta_str)

        sec_per_step = time.time() - self.current_time
        self.current_time = time.time()
       
        s  = "\r"
        s += self.pretxt
        s += "["
        s += "=" * (bar_filled)
        s += ">" * (bar_unfilled != 0)
        s += "." * (bar_unfilled - 1)
        s += "] "
        s += "%5.2f%% | ETA: %s (%.4fs) %s" % (
              bar_percent, eta_str, sec_per_step, self.txt)
        print s,
        #sys.stdout.write("%s\r" % s)
        if(self.count == self.max_count):
            print("")
        sys.stdout.flush()
        

def test_pgbar():
    
    pgbar = ProgressBar()
    pgbar.start(15)
    for i in range(15):
        time.sleep(1)
        pgbar.step()
    pgbar.stop()

def progress_bar(count, max):
    """ simple CLI progress bar 
    """
    bar_width    = 50
    bar_filled   = int(round(bar_width * count / float(max)))
    bar_unfilled = bar_width - bar_filled
    bar_percent  = round(100 * (count / float(max)))

    s  = "\r"
    s += "["
    s += "=" * bar_filled
    s += " " * bar_unfilled
    s += "] "
    s += "%s%%" % (bar_percent)
    print s,
    if(count == max):
        print("\n")
    sys.stdout.flush()

