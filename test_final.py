from dataclasses import dataclass, replace
from typing import Any, Union, Tuple, Dict, Mapping, Callable, Iterable, Iterator, TypeVar, Set, Sequence
from operator import itemgetter
import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple

import numpy as np
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical, Constant, Choose, SampledDistribution
from rl.function_approx import AdamGradient, DNNApprox, DNNSpec, Weights
from rl.markov_decision_process import FiniteMarkovDecisionProcess, MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from rl.markov_process import MarkovRewardProcess
from rl.monte_carlo import greedy_policy_from_qvf, epsilon_greedy_policy
from rl.policy import Policy, DeterministicPolicy, FiniteDeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory
from scipy.stats import poisson, nbinom, rv_discrete, rv_continuous
import time


@dataclass(frozen=True)
class InvState:
    on_hands: Tuple[int, int]
    on_orders: Tuple[int, int]

    def inventory_position(self) -> Tuple[int, int]:
        return self.on_hands[0] + self.on_orders[0], self.on_hands[1] + self.on_orders[1]

InvAction = Tuple[int, int, int]

InvOrderMapping = Mapping[
    InvState,
    Mapping[InvAction, Categorical[Tuple[InvState, float]]]
]

A = TypeVar('A')
S = TypeVar('S')

### Deep Q-Learning with a target and main NN FuncApprox model ###
from copy import deepcopy
from rl.td import PolicyFromQType
def deep_q_learning_experience_replay(
    mdp: MarkovDecisionProcess[S, A],
    policy_from_q: PolicyFromQType,
    states: NTStateDistribution[S],
    approx_0: DNNApprox[Tuple[NonTerminal[S], A]],
    γ: float,
    max_episode_length: int,
    mini_batch_size: int,
    weights_decay_half_life: float,
    main_update_every: int = 4,
    targ_update_every: int = 100
) -> Iterator[QValueFunctionApprox[S, A]]:
    exp_replay: ExperienceReplayMemory[TransitionStep[S, A]] = \
        ExperienceReplayMemory(
            #time_weights_func=lambda t: 0.5 ** (t / weights_decay_half_life),
        )
    targ_q: DNNApprox[Tuple[NonTerminal[S], A]] = approx_0
    main_q: DNNApprox[Tuple[NonTerminal[S], A]] = deepcopy(targ_q)
    yield targ_q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(main_q, mdp)
            action: A = policy.act(state).sample() 
            next_state, reward = mdp.step(state, action).sample()
            exp_replay.add_data(TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            if steps % main_update_every == 0:
                trs: Sequence[TransitionStep[S, A]] = \
                    exp_replay.sample_mini_batch(mini_batch_size)
                main_q = main_q.update(
                    [(
                        (tr.state, tr.action),
                        tr.reward + γ * (
                            max(targ_q((tr.next_state, a))
                                for a in mdp.actions(tr.next_state))
                            if isinstance(tr.next_state, NonTerminal) else 0.)
                    ) for tr in trs],
                )
            yield targ_q
            steps += 1
            state = next_state

        if steps >= min(targ_update_every, max_episode_length):
            targ_q = replace(targ_q, weights=deepcopy(main_q.weights))




class SimpleTwoInventoryMDPCap(FiniteMarkovDecisionProcess[InvState, InvAction]):

    def __init__(
        self,
        capacities: Tuple[int, int],
        lambdas: Tuple[float, float],
        holding_costs: Tuple[float, float],
        stockout_costs: Tuple[float, float],
        supply_cost: float,
        transfer_cost: float
    ):
        self.capacities: Tuple[int, int] = capacities
        self.lambdas: Tuple[float, float] = lambdas
        self.holding_costs: Tuple[float, float] = holding_costs
        self.stockout_costs: Tuple[float, float] = stockout_costs
        self.supply_cost = supply_cost
        self.transfer_cost = transfer_cost

        self.distrs = poisson(lambdas[0]), poisson(lambdas[1])
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InvState, Dict[int, Categorical[Tuple[InvState,
                                                            float]]]] = {}

        for alpha0 in range(self.capacities[0] + 1):
            for alpha1 in range(self.capacities[1] + 1):
                alphas = alpha0, alpha1
                for beta0 in range(self.capacities[0] + 1 - alpha0):
                    for beta1 in range(self.capacities[1] + 1 - alpha1):
                        betas = beta0, beta1
                        state: InvState = InvState(alphas, betas)
                        ips: Tuple[int, int] = state.inventory_position()
                        d1: Dict[InvAction, Categorical[Tuple[InvState, float]]] = {}
                        
                        for transfer in range(-alpha0, alpha1 + 1):
                            base_rewards = -self.transfer_cost*abs(transfer) -\
                                             self.holding_costs[0]*min(alpha0+transfer, alpha0) -\
                                             self.holding_costs[1]*min(alpha1-transfer, alpha1)
                            new_ips = ips[0]+transfer, ips[1]-transfer
                            for order0 in range(self.capacities[0] - new_ips[0] + 1):
                                for order1 in range(self.capacities[1] - new_ips[1] + 1):
                                    action = order0, order1, transfer
                                    new_base_rewards = base_rewards - self.supply_cost*(order0 + order1)
                                    sr_probs_dict: Dict[Tuple[InvState, float], float] = {}
                                    for i0 in range(new_ips[0]):
                                        for i1 in range(new_ips[1]):
                                            next_state_alphas = new_ips[0]-i0, new_ips[1]-i1
                                            next_state = InvState(next_state_alphas, action[:2])
                                            sr_probs_dict[(next_state, new_base_rewards)] =\
                                              self.distrs[0].pmf(i0) * self.distrs[1].pmf(i1)
                                              
                                    # 3 cases remaining       
                                    probs: Tuple[float, float] =\
                                        1 - self.distrs[0].cdf(new_ips[0] - 1),\
                                        1 - self.distrs[1].cdf(new_ips[1] - 1)
                                    
                                    # Case 1 : i0 >= new_ips[0], i1 < new_ips[1]
                                    for i1 in range(new_ips[1]):
                                        next_state_alphas = 0, new_ips[1]-i1
                                        next_state = InvState(next_state_alphas, action[:2])
                                        reward = new_base_rewards - self.stockout_costs[0] *\
                                            (probs[0] * (self.lambdas[0] - new_ips[0]) +\
                                            new_ips[0] * self.distrs[0].pmf(new_ips[0]))
                                        sr_probs_dict[(next_state, reward)] = probs[0] *\
                                          self.distrs[1].pmf(i1)
                                    
                                    # Case 2 : i1 >= new_ips[1], i0 < new_ips[0]
                                    for i0 in range(new_ips[0]):
                                        next_state_alphas = new_ips[0]-i0, 0
                                        next_state = InvState(next_state_alphas, action[:2])
                                        reward = new_base_rewards - self.stockout_costs[1] *\
                                            (probs[1] * (self.lambdas[1] - new_ips[1]) +\
                                            new_ips[1] * self.distrs[1].pmf(new_ips[1]))
                                        sr_probs_dict[(next_state, reward)] = probs[1] *\
                                          self.distrs[0].pmf(i0)
                                    
                                    # Case 3 : i0 == new_ips[0], i1 == new_ips[1]
                                    reward = new_base_rewards - self.stockout_costs[1] *\
                                            (probs[1] * (self.lambdas[1] - new_ips[1]) +\
                                            new_ips[1] * self.distrs[1].pmf(new_ips[1])) -\
                                            self.stockout_costs[0] * probs[0] *\
                                            ((self.lambdas[0] - new_ips[0]) + new_ips[0] *\
                                                self.distrs[0].pmf(new_ips[0]))
                                    sr_probs_dict[(InvState((0,0), action[:2]), reward)] =\
                                        probs[0] * probs[1]
                                        
                                    d1[action] = Categorical(sr_probs_dict)

                        d[state] = d1
        return d


class TwoInventoryMDPCap(MarkovDecisionProcess[InvState, InvAction]):
    def __init__(
        self,
        capacities: Tuple[int, int],
        holding_costs: Tuple[float, float],
        stockout_costs: Tuple[float, float],
        supply_cost: float,
        transfer_cost: float,
        distribution: Union[rv_discrete, rv_continuous],
        distr_kwargs: Tuple[Dict, Dict]
    ):
        self.capacities: Tuple[int, int] = capacities
        self.holding_costs: Tuple[float, float] = holding_costs
        self.stockout_costs: Tuple[float, float] = stockout_costs
        self.supply_cost = supply_cost
        self.transfer_cost = transfer_cost
        self.distrs = distribution(**distr_kwargs[0]), distribution(**distr_kwargs[1])
        self.distr_name = distribution.name

    def actions(self, state: NonTerminal[InvState]) -> Iterable[InvAction]:
        ips: Tuple[int, int] = state.state.inventory_position()
        for transfer in range(-state.state.on_hands[0] , state.state.on_hands[1] + 1):
            new_ips = ips[0]+transfer, ips[1]-transfer
            for order0 in range(self.capacities[0] - new_ips[0] + 1):
                for order1 in range(self.capacities[1] - new_ips[1] + 1):
                    action = order0, order1, transfer
                    yield action

    def step(
        self,
        state: NonTerminal[InvState],
        action: InvAction
    ) -> SampledDistribution[Tuple[NonTerminal[InvState], float]]:
        order0, order1, transfer = action
        on_hands = np.array(state.state.inventory_position()) + (transfer, -transfer)
        new_on_orders = order0, order1
        base_rewards = -self.transfer_cost*abs(transfer) -\
                        self.holding_costs[0]*min(on_hands[0], state.state.on_hands[0]) -\
                        self.holding_costs[1]*min(on_hands[1], state.state.on_hands[1]) -\
                        self.supply_cost*(order0 + order1)

        def sampler():
            demands = self.distrs[0].rvs(), self.distrs[1].rvs()
            new_on_hands = on_hands - demands
            rewards = base_rewards + min(0, new_on_hands[0])*self.stockout_costs[0] + \
                        min(0, new_on_hands[1])*self.stockout_costs[1]
            new_on_hands = max(0, new_on_hands[0]), max(0, new_on_hands[1])
            return NonTerminal(InvState(on_hands=new_on_hands, on_orders=new_on_orders)), rewards

        return SampledDistribution(sampler=sampler)

    def newsvendor(self) -> DeterministicPolicy[InvState, InvAction]:
        frac0 = (self.stockout_costs[0] - self.supply_cost) / (self.stockout_costs[0] + self.holding_costs[0])
        frac1 = (self.stockout_costs[1] - self.supply_cost) / (self.stockout_costs[1] + self.holding_costs[1])
        optimal_holdings: Tuple[float, float] = self.distrs[0].ppf(frac0), self.distrs[1].ppf(frac1)
        def action_for(state: InvState) -> InvAction:
            diff = np.rint(np.array(state.inventory_position()) - \
                            (self.distrs[0].stats('m'), self.distrs[1].stats('m')) -\
                            optimal_holdings).astype(int)
            if diff[0] > 0 and diff[1] < 0:
                transfer = max(-diff[0], diff[1], -state.on_hands[0])
            elif diff[0] < 0 and diff[1] > 0:
                transfer = min(-diff[0], diff[1], state.on_hands[1])
            else:
                transfer = 0
            tent_orders = -min(0, diff[0] + transfer), -min(0, diff[1] - transfer)
            orders = tuple(np.minimum(self.capacities - np.array(state.inventory_position()), tent_orders))
            return *orders, transfer

        return DeterministicPolicy(action_for=action_for)


def evaluate_policy_on_mdp(mdp: MarkovDecisionProcess[InvState, InvAction],
                           pol: Policy[InvState, InvAction],
                           gamma: float,
                           start_distrib: NTStateDistribution[InvState],
                           num_traces: int = 100,
                           num_samples: int = 100):
    mrp : MarkovRewardProcess[InvState] = mdp.apply_policy(pol)
    discounts = np.power(gamma * np.ones(num_traces), np.arange(num_traces))
    reward_trace_iter = mrp.reward_traces(start_distrib)
    rewards = np.empty((num_samples, num_traces))
    for sample in range(num_samples):
        trace_iter = next(reward_trace_iter)
        for trace in range(num_traces):
            step = next(trace_iter)
            rewards[sample, trace] = step.reward
    return np.sum(rewards * discounts, axis=1)


if __name__ == '__main__':
    from pprint import pprint

    user_capacities = 8, 7
    user_ps = 0.3, 0.5
    user_lambdas = tuple(np.multiply(user_ps, user_capacities))
    user_holding_costs = 1.0, 4.0
    user_stockout_costs = 10.0, 20.0
    user_supply_cost = 2.0
    user_transfer_cost = 1.0

    user_gamma = 0.95
    
    pois_policies = {}
    
    '''
    print('Generating explicit finite MDP...')
    si_start = time.time()
    si_mdp: FiniteMarkovDecisionProcess[InvState, InvAction] =\
        SimpleTwoInventoryMDPCap(
            capacities=user_capacities,
            lambdas=user_lambdas,
            holding_costs=user_holding_costs,
            stockout_costs=user_stockout_costs,
            supply_cost=user_supply_cost,
            transfer_cost=user_transfer_cost
        )
    print('Generated finite MDP. Time elapsed (s): {}'.format(time.time() - si_start))

    from rl.dynamic_programming import value_iteration_result
    print("Poisson MDP iteration starting...")
    pois_start = time.time()
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    print('Completed. Time elapsed (sec) = {}'.format(time.time() - pois_start))
    pois_policies['Optimal'] = opt_policy_vi
    '''
    
    #########################

    # Same MDP as the simple inventory case, except specified without transition dict, only step function.
    print('Generating implicit (possibly non-finite) MDP with the same mechanics...')
    t_start = time.time()
    t_mdp: MarkovDecisionProcess[InvState, InvAction] =\
        TwoInventoryMDPCap(
            capacities=user_capacities,
            holding_costs=user_holding_costs,
            stockout_costs=user_stockout_costs,
            supply_cost=user_supply_cost,
            transfer_cost=user_transfer_cost,
            distribution=poisson,
            distr_kwargs=(
                { 'mu': user_lambdas[0] },
                { 'mu': user_lambdas[1]}
            )
        )
    print('Generated implicit (possibly non-finite) MDP. Time elapsed (s): {}'.format(time.time() - t_start))

    def t_mdp_sampler():
        a = np.random.randint(user_capacities[0] + 1)
        b = np.random.randint(user_capacities[1] + 1)
        x = np.random.randint(a + 1)
        y = np.random.randint(b + 1)
        return NonTerminal(InvState(on_hands=(x, y), on_orders=(a-x, b-y)))
    pois_policies['Newsvendor'] = t_mdp.newsvendor()
    

    ### Setting up the Deep Q-Learning components
    # Functions
    ffs : Sequence[Callable[[Tuple[NonTerminal[InvState], InvAction]], float]] = [
        lambda _: 1.,
        lambda x: x[0].state.on_hands[0],
        lambda x: x[0].state.on_hands[1],
        lambda x: x[0].state.on_orders[0],
        lambda x: x[0].state.on_orders[1],
        lambda x: x[1][0],
        lambda x: x[1][1],
        lambda x: x[1][2]
    ]

    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0., x)

    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def sigmoid(x: np.ndarray) -> np.ndarray:
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def sigmoid_grad(x: np.ndarray) -> np.ndarray:
        sigX = sigmoid(x)
        return sigX * (1. - sigX)

    def eps_greedy_policy(
        q: QValueFunctionApprox[InvState, InvAction],
        mdp: MarkovDecisionProcess[InvState, InvAction],
    ) -> DeterministicPolicy[InvState, InvAction]:
        try:
            eps = 2**(- eps_greedy_policy.times_called / 1000)
            if eps > 0.01:
                eps_greedy_policy.times_called += 1
        except AttributeError:
            eps = 1.
            eps_greedy_policy.times_called = 1
        return epsilon_greedy_policy(q, mdp, eps)

    # Objects
    ds : DNNSpec = DNNSpec(
        neurons=[24, 12],
        bias=True,
        hidden_activation=relu,
        hidden_activation_deriv=relu_grad,
        output_activation=lambda x: x,
        output_activation_deriv=lambda x: np.ones_like(x)
    )
    
    qvf2_dnn_approx: QValueFunctionApprox[InvState, InvAction] = DNNApprox.create(
        feature_functions=ffs,
        dnn_spec=ds,
        regularization_coeff=0.05
    )

    
    #### Starting the Deep Q-Learning proper
    qvf2_iter : Iterator[QValueFunctionApprox[InvState, InvAction]] = \
        deep_q_learning_experience_replay(
            mdp= t_mdp,
            policy_from_q= eps_greedy_policy,
            states= SampledDistribution(sampler=t_mdp_sampler),
            approx_0= qvf2_dnn_approx,
            γ= user_gamma,
            max_episode_length=100,
            mini_batch_size=128,
            weights_decay_half_life=700
        )

    print('Beginning deep Q-learning...')
    start_time = time.time()
    total_iters = 50 * 100   # should be multiple of max_episode_length above
    for it in range(1, 1 + total_iters):
        qvf2_approx = next(qvf2_iter)
        if it % (total_iters // 10) == 0:
            print('DQL iteration {} of {}'.format(it, total_iters))
    print('Total time for 5000 iterations: {} seconds'.format(time.time() - start_time))

    qvf2_policy : DeterministicPolicy[InvState, InvAction] = \
        greedy_policy_from_qvf(qvf2_approx, t_mdp.actions)
    
    pois_policies['Deep Q-Learning'] = qvf2_policy
    

    start_distrib = Constant(NonTerminal(InvState((0, 0), (0,0))))
    for name, pol in pois_policies.items():
        rewards = evaluate_policy_on_mdp(t_mdp, pol, user_gamma, start_distrib)
        print('Discounted rewards of 100 samples w/ 100 steps each of policy: {}\n{}\nMean: {}'.format(
            name, rewards, np.mean(rewards)
        ))

    
    t_mdp_states = [ NonTerminal(InvState(on_hands=(x, y), on_orders=(a-x, b-y)))
                    for a in range(user_capacities[0] + 1)
                    for b in range(user_capacities[1] + 1)
                    for x in range(a + 1)
                    for y in range(b + 1)
                ]
    for state in t_mdp_states:
        print('State: {}, action = {}'.format(state.state, qvf2_policy.act(state).value))
    

    
    