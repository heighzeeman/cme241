import itertools
import numpy as np
from io import StringIO
import sys
from typing import Tuple
import matplotlib.pyplot as plt

from rl.distribution import (
    Choose,
    Constant,
    Distribution,
)
from rl.markov_process import (
    FiniteMarkovProcess,
    FiniteMarkovRewardProcess,
    MarkovProcess,
    MarkovRewardProcess,
    NonTerminal,
    State,
)


class SnakesAndLadders(FiniteMarkovRewardProcess[int]):
    def __init__(self):
        ladders_snakes_dict = {1:38, 4:14, 8:30, 21:42, 28:76, 50:67, 71:92, 80:99, 97:78, 95:56, 88:24, 62:18, 48:26, 36:6, 32:10}
        F = { i : i if i not in ladders_snakes_dict else ladders_snakes_dict[i] for i in range(101) }
        for s in range(101, 106):
            F[s] = F[200 - s]
        
        # Constant reward of 1 for each step => total reward is the number of steps taken
        transition_reward_map = { s: Choose([ (F[s1], 1.0) for s1 in range(s+1, s+7) ]) for s in range(100)}
        super().__init__(transition_reward_map)

def main():
    SNL = SnakesAndLadders()
    ten_traces = itertools.islice(SNL.traces(Constant(NonTerminal(0))), 10)
    processed_traces = [ [ s.state for s in trace ] for trace in ten_traces ]
    for trace in processed_traces:
        plt.plot(trace)
    plt.xlabel('Number of steps elapsed')
    plt.ylabel('Square')
    plt.title('Graph of ten traces of Snakes and Ladders')
    plt.savefig('traces.png')
    plt.clf()
    
    tthous_traces = itertools.islice(SNL.traces(Constant(NonTerminal(0))), 10000)
    processed_lens = [ len(list(trace)) - 1 for trace in tthous_traces ]
    plt.hist(processed_lens, bins=200)
    plt.title('Histogram of number of steps needed for 10,000 SnL games')
    plt.xlabel('Number of steps')
    plt.ylabel('Number of games')
    plt.savefig('hist.png')
    print('Theoretical avg:',  SNL.get_value_function_vec(1.0)[0], 'and 10k sample avg:', np.mean(processed_lens))
        

if __name__ == '__main__':
    main()