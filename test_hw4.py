from typing import Iterator, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from rl.approximate_dynamic_programming import extended_vf, evaluate_mrp
from rl.approximate_dynamic_programming import value_iteration as approx_value_iteration
from rl.distribution import Distribution
from rl.function_approx import FunctionApprox, Dynamic
from rl.iterate import converged, iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import DeterministicPolicy, Policy, UniformPolicy

S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
NTStateDistribution = Distribution[NonTerminal[S]]

def approx_policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int,
    done
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    def update(vf_policy: Tuple[ValueFunctionApprox[S], Policy[S, A]]) -> \
      Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]:
        vf, pi = vf_policy
        mrp: MarkovRewardProcess[S] = mdp.apply_policy(pi)
        policy_vf = converged(evaluate_mrp(mrp, γ, vf, non_terminal_states_distribution, num_state_samples), done)
        
        
        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(policy_vf, s1)
            
        def deter_policy(state: S) -> A:
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]
        
        
        return policy_vf, DeterministicPolicy(deter_policy)

    pi_0: Policy[S, A] = UniformPolicy(lambda x : mdp.actions(NonTerminal(x)))

    return iterate(update, (approx_0, pi_0))



from dataclasses import dataclass
from typing import Dict, Mapping
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess
from rl.distribution import Categorical, Choose
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[int, Categorical[Tuple[InventoryState, float]]]
]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(alpha, beta)
                ip: int = state.inventory_position()
                base_reward: float = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                        {(InventoryState(ip - i, order), base_reward):
                         self.poisson_distr.pmf(i) for i in range(ip)}

                    probability: float = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: float = base_reward - self.stockout_cost *\
                        (probability * (self.poisson_lambda - ip) +
                         ip * self.poisson_distr.pmf(ip))
                    sr_probs_dict[(InventoryState(0, order), reward)] = \
                        probability
                    d1[order] = Categorical(sr_probs_dict)

                d[state] = d1
        return d


def approx_policy_iteration_result(
    mdp: MarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int,
    criterion
) -> Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]:
    return converged(approx_policy_iteration(mdp, gamma, approx_0, non_terminal_states_distribution,num_state_samples, criterion), done=lambda x, y: criterion(x[0], y[0]))


def approx_value_iteration_result(
    mdp: MarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int,
    criterion
) -> ValueFunctionApprox[S]:
    return converged(approx_value_iteration(mdp, gamma, approx_0, non_terminal_states_distribution,num_state_samples), done=criterion)

    
if __name__ == '__main__':
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FiniteDeterministicPolicy[InventoryState, int] = \
        FiniteDeterministicPolicy(
            {InventoryState(alpha, beta): user_capacity - (alpha + beta)
             for alpha in range(user_capacity + 1)
             for beta in range(user_capacity + 1 - alpha)}
    )
    
    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result
    
   
    approx = { s: 0.0 for s in si_mdp.non_terminal_states }
    dist: NTStateDistribution[InventoryState] = Choose(si_mdp.non_terminal_states)
    
    def almost_equal_si_mdp_vf_pis(
        x1: ValueFunctionApprox[S],
        x2: ValueFunctionApprox[S]) -> bool:
        temp = np.max([abs(x1(s) - x2(s)) for s in si_mdp.non_terminal_states])
        #print(temp)
        return temp < 1e-2
    
    print("MDP Approx Value Iteration Optimal Value Function")
    print("--------------")
    a_opt_vf_vi = approx_value_iteration_result(si_mdp, user_gamma, \
      Dynamic(approx), dist, user_capacity*2, almost_equal_si_mdp_vf_pis)
    print('Optimal value function:')
    pprint(a_opt_vf_vi)
    print()
    
    print("MDP Approx Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    a_opt_vf_pi, a_opt_policy_pi = approx_policy_iteration_result(si_mdp, user_gamma, \
      Dynamic(approx), dist, user_capacity*2, almost_equal_si_mdp_vf_pis)
    print('Optimal value function:')
    pprint(a_opt_vf_pi)
    print('Optimal policy:')
    print({s.state: a_opt_policy_pi.act(s).value for s in si_mdp.non_terminal_states})
    print()
    
    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()
    

