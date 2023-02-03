from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hands: Tuple[int, int]
    on_orders: Tuple[int, int]

    def inventory_position(self) -> Tuple[int, int]:
        return self.on_hands[0] + self.on_orders[0], self.on_hands[1] + self.on_orders[1]


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]
]


class SimpleTwoInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, Tuple[int, int, int]]):

    def __init__(
        self,
        capacities: Tuple[int, int],
        poisson_lambdas: Tuple[float, float],
        holding_costs: Tuple[float, float],
        stockout_costs: Tuple[float, float],
        supply_cost: float,
        transfer_cost: float
    ):
        self.capacities: Tuple[int, int] = capacities
        self.poisson_lambdas: Tuple[float, float] = poisson_lambdas
        self.holding_costs: Tuple[float, float] = holding_costs
        self.stockout_costs: Tuple[float, float] = stockout_costs
        self.supply_cost = supply_cost
        self.transfer_cost = transfer_cost

        self.poisson_distrs = poisson(poisson_lambdas[0]), poisson(poisson_lambdas[1])
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha0 in range(self.capacities[0] + 1):
            for alpha1 in range(self.capacities[1] + 1):
                alphas = alpha0, alpha1
                for beta0 in range(self.capacities[0] + 1 - alpha0):
                    for beta1 in range(self.capacities[1] + 1 - alpha1):
                        betas = beta0, beta1
                        state: InventoryState = InventoryState(alphas, betas)
                        ips: Tuple[int, int] = state.inventory_position()
                        d1: Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]] = {}
                        
                        for transfer in range(-alpha0, alpha1 + 1):
                            base_rewards = -self.transfer_cost*abs(transfer) -\
                                             self.holding_costs[0]*min(alpha0+transfer, alpha0) -\
                                             self.holding_costs[1]*min(alpha1-transfer, alpha1)
                            new_ips = ips[0]+transfer, ips[1]-transfer
                            for order0 in range(self.capacities[0] - new_ips[0] + 1):
                                for order1 in range(self.capacities[1] - new_ips[1] + 1):
                                    action = order0, order1, transfer
                                    new_base_rewards = base_rewards - self.supply_cost*(order0 + order1)
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = {}
                                    for i0 in range(new_ips[0]):
                                        for i1 in range(new_ips[1]):
                                            next_state_alphas = new_ips[0]-i0, new_ips[1]-i1
                                            next_state = InventoryState(next_state_alphas, action[:2])
                                            sr_probs_dict[(next_state, new_base_rewards)] =\
                                              self.poisson_distrs[0].pmf(i0) * self.poisson_distrs[1].pmf(i1)
                                              
                                    # 3 cases remaining       
                                    probs: Tuple[float, float] =\
                                        1 - self.poisson_distrs[0].cdf(new_ips[0] - 1),\
                                        1 - self.poisson_distrs[1].cdf(new_ips[1] - 1)
                                    
                                    # Case 1 : i0 >= new_ips[0], i1 < new_ips[1]
                                    for i1 in range(new_ips[1]):
                                        next_state_alphas = 0, new_ips[1]-i1
                                        next_state = InventoryState(next_state_alphas, action[:2])
                                        reward = new_base_rewards - self.stockout_costs[0] *\
                                            (probs[0] * (self.poisson_lambdas[0] - new_ips[0]) +\
                                            new_ips[0] * self.poisson_distrs[0].pmf(new_ips[0]))
                                        sr_probs_dict[(next_state, reward)] = probs[0] *\
                                          self.poisson_distrs[1].pmf(i1)
                                    
                                    # Case 2 : i1 >= new_ips[1], i0 < new_ips[0]
                                    for i0 in range(new_ips[0]):
                                        next_state_alphas = new_ips[0]-i0, 0
                                        next_state = InventoryState(next_state_alphas, action[:2])
                                        reward = new_base_rewards - self.stockout_costs[1] *\
                                            (probs[1] * (self.poisson_lambdas[1] - new_ips[1]) +\
                                            new_ips[1] * self.poisson_distrs[1].pmf(new_ips[1]))
                                        sr_probs_dict[(next_state, reward)] = probs[1] *\
                                          self.poisson_distrs[0].pmf(i0)
                                    
                                    # Case 3 : i0 == new_ips[0], i1 == new_ips[1]
                                    reward = new_base_rewards - self.stockout_costs[1] *\
                                            (probs[1] * (self.poisson_lambdas[1] - new_ips[1]) +\
                                            new_ips[1] * self.poisson_distrs[1].pmf(new_ips[1])) -\
                                            self.stockout_costs[0] * probs[0] *\
                                            ((self.poisson_lambdas[0] - new_ips[0]) + new_ips[0] *\
                                                self.poisson_distrs[0].pmf(new_ips[0]))
                                    sr_probs_dict[(InventoryState((0,0), action[:2]), reward)] =\
                                        probs[0] * probs[1]
                                        
                                    d1[action] = Categorical(sr_probs_dict)

                        d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacities = 3, 3
    user_poisson_lambdas = 1.0, 1.0
    user_holding_costs = 1.0, 1.0
    user_stockout_costs = 10.0, 10.0
    user_supply_cost = 0.5
    user_transfer_cost = 0.5

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, Tuple[int, int, int]] =\
        SimpleTwoInventoryMDPCap(
            capacities=user_capacities,
            poisson_lambdas=user_poisson_lambdas,
            holding_costs=user_holding_costs,
            stockout_costs=user_stockout_costs,
            supply_cost=user_supply_cost,
            transfer_cost=user_transfer_cost
        )

    #print("MDP Transition Map")
    #print("------------------")
    #print(si_mdp)
    
    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result
    
    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
    print("Action is structured as follows: (store 1's order, store 2's order, store 2->1 transfer amount)")
    