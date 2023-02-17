from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List

from rl.distribution import (
    Bernoulli,
    Categorical,
    Distribution,
    Gaussian,
    SampledDistribution,
    Constant,
)
from rl.markov_process import MarkovProcess, NonTerminal, State

import itertools
import numpy as np

@dataclass(frozen=True)
class DollarsAndShares:

    dollars: float
    shares: int


PriceSizePairs = Sequence[DollarsAndShares]


@dataclass(frozen=True)
class OrderBook:

    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        return self.descending_bids[0].dollars

    def ask_price(self) -> float:
        return self.ascending_asks[0].dollars

    def mid_price(self) -> float:
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        return self.ascending_asks[-1].dollars - \
            self.descending_bids[-1].dollars

    @staticmethod
    def eat_book(
        ps_pairs: PriceSizePairs,
        shares: int
    ) -> Tuple[DollarsAndShares, PriceSizePairs]:
        '''
        Returned DollarsAndShares represents the pair of
        dollars transacted and the number of shares transacted
        on ps_pairs (with number of shares transacted being less
        than or equal to the input shares).
        Returned PriceSizePairs represents the remainder of the
        ps_pairs after the transacted number of shares have eaten into
        the input ps_pairs.
        '''
        rem_shares: int = shares
        dollars: float = 0.
        for i, d_s in enumerate(ps_pairs):
            this_price: float = d_s.dollars
            this_shares: int = d_s.shares
            dollars += this_price * min(rem_shares, this_shares)
            if rem_shares < this_shares:
                return (
                    DollarsAndShares(dollars=dollars, shares=shares),
                    [DollarsAndShares(
                        dollars=this_price,
                        shares=this_shares - rem_shares
                    )] + list(ps_pairs[i+1:])
                )
            else:
                rem_shares -= this_shares

        return (
            DollarsAndShares(dollars=dollars, shares=shares - rem_shares),
            []
        )

    def sell_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.descending_bids)
                                     if d_s.dollars < price), None)
        eligible_bids: PriceSizePairs = self.descending_bids \
            if index is None else self.descending_bids[:index]
        ineligible_bids: PriceSizePairs = [] if index is None else \
            self.descending_bids[index:]

        d_s, rem_bids = OrderBook.eat_book(eligible_bids, shares)
        new_bids: PriceSizePairs = list(rem_bids) + list(ineligible_bids)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_asks: List[DollarsAndShares] = list(self.ascending_asks)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_asks)
                                          if d_s.dollars >= price), None)
            if index1 is None:
                new_asks.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_asks[index1].dollars != price:
                new_asks.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_asks[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_asks[index1].shares + rem_shares
                )
            return d_s, OrderBook(
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                descending_bids=new_bids
            )

    def sell_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_bids = OrderBook.eat_book(
            self.descending_bids,
            shares
        )
        return (d_s, replace(self, descending_bids=rem_bids))

    def buy_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.ascending_asks)
                                     if d_s.dollars > price), None)
        eligible_asks: PriceSizePairs = self.ascending_asks \
            if index is None else self.ascending_asks[:index]
        ineligible_asks: PriceSizePairs = [] if index is None else \
            self.ascending_asks[index:]

        d_s, rem_asks = OrderBook.eat_book(eligible_asks, shares)
        new_asks: PriceSizePairs = list(rem_asks) + list(ineligible_asks)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_bids: List[DollarsAndShares] = list(self.descending_bids)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_bids)
                                          if d_s.dollars <= price), None)
            if index1 is None:
                new_bids.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_bids[index1].dollars != price:
                new_bids.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_bids[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_bids[index1].shares + rem_shares
                )
            return d_s, replace(
                self,
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                ascending_asks=new_asks
            )

    def buy_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_asks = OrderBook.eat_book(
            self.ascending_asks,
            shares
        )
        return (d_s, replace(self, ascending_asks=rem_asks))

    def pretty_print_order_book(self) -> None:
        from pprint import pprint
        print()
        print("Bids")
        pprint(self.descending_bids)
        print()
        print("Asks")
        print()
        pprint(self.ascending_asks)
        print()

    def display_order_book(self) -> None:
        import matplotlib.pyplot as plt

        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        if self.descending_bids:
            plt.bar(bid_prices, bid_shares, color='blue')

        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        if self.ascending_asks:
            plt.bar(ask_prices, ask_shares, color='red')

        all_prices = sorted(bid_prices + ask_prices)
        all_ticks = ["%d" % x for x in all_prices]
        plt.xticks(all_prices, all_ticks)
        plt.grid(axis='y')
        plt.xlabel("Prices")
        plt.ylabel("Number of Shares")
        plt.title("Order Book")
        # plt.xticks(x_pos, x)
        plt.show()


class facialOBMP(MarkovProcess[OrderBook]):
    samples: int = 100
    
    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[State[OrderBook]]:
        def next_state(state=state):
            num_buy_LOs = np.random.randint(0, 4)
            num_sell_LOs = np.random.randint(0, 4)  
            buy_prices = np.random.normal((state.state.mid_price() + 9*state.state.bid_price())/10, \
                                    state.state.bid_ask_spread() / 1.1, num_buy_LOs)
            sell_prices = np.random.normal((state.state.mid_price() + 9*state.state.ask_price())/10, \
                                    state.state.bid_ask_spread() / 1.1, num_sell_LOs)
            buy_LOs = zip(buy_prices, np.random.poisson(state.state.ascending_asks[0].shares, num_buy_LOs))
            sell_LOs = zip(sell_prices, np.random.poisson(state.state.descending_bids[0].shares, num_sell_LOs))
            num_buy_MOs = np.random.poisson(5 - num_buy_LOs)
            num_sell_MOs = np.random.poisson(5 - num_sell_LOs)
            buy_MOs = np.random.binomial(state.state.ascending_asks[0].shares, 0.3, num_buy_MOs)
            sell_MOs = np.random.binomial(state.state.descending_bids[0].shares, 0.3, num_sell_MOs)
            
            new_state = deepcopy(state.state)
            for MO in buy_MOs:
                new_state = new_state.buy_market_order(MO)[1]
            for MO in sell_MOs:
                new_state = new_state.sell_market_order(MO)[1]
            for LO in buy_LOs:
                new_state = new_state.buy_limit_order(*LO)[1]
            for LO in sell_LOs:
                new_state = new_state.sell_limit_order(*LO)[1]
            
            return NonTerminal(new_state)
            
        return SampledDistribution(next_state, expectation_samples=self.samples)


if __name__ == '__main__':
    from numpy.random import poisson

    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    mp = facialOBMP()
    traces = itertools.islice(mp.simulate(Constant(NonTerminal(ob0))), 10)
    
    for ob in traces:
        ob.state.pretty_print_order_book()
        ob.state.display_order_book()