"""
Iron Condor Strategy
Neutral income strategy: sell an OTM call spread + sell an OTM put spread.

Pricing: Cox-Ross-Rubinstein (CRR) binomial tree — American-style exercise.
Max Profit: net credit received
Max Loss:   spread width - net credit

Author: Agentic Trading
Version: 1.0.0
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call', steps: int = 100,
              style: str = 'american') -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
    dt   = T / steps
    u    = math.exp(sigma * math.sqrt(dt))
    d    = 1.0 / u
    disc = math.exp(-r * dt)
    p    = (math.exp(r * dt) - d) / (u - d)
    values = [
        max(S * (u ** j) * (d ** (steps - j)) - K, 0.0) if option_type == 'call'
        else max(K - S * (u ** j) * (d ** (steps - j)), 0.0)
        for j in range(steps + 1)
    ]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1 - p) * values[j])
            if style == 'american':
                spot_ij   = S * (u ** j) * (d ** (i - j))
                intrinsic = max(spot_ij - K, 0.0) if option_type == 'call' else max(K - spot_ij, 0.0)
                values[j] = max(intrinsic, cont)
            else:
                values[j] = cont
    return round(values[0], 6)


@dataclass
class CondorResult:
    call_credit:        float
    put_credit:         float
    net_credit:         float
    max_profit:         float
    max_loss:           float
    upper_breakeven:    float
    lower_breakeven:    float
    profit_zone_width:  float
    risk_reward_ratio:  float


class IronCondorStrategy:
    """
    Iron Condor Strategy (CRR pricing)

    Sell OTM call spread + OTM put spread, same expiry.
    Profit when underlying stays inside the short strikes.
    Uses Cox-Ross-Rubinstein binomial tree for all four legs.

    Parameters
    ----------
    call_short_offset  : short call % above spot  (default +5%)
    call_long_offset   : long call % above spot   (default +10%)
    put_short_offset   : short put % below spot   (default -5%)
    put_long_offset    : long put % below spot    (default -10%)
    days_to_expiry     : days until expiry (default 45)
    risk_free_rate     : annual risk-free rate (default 0.05)
    implied_volatility : IV override
    contracts          : number of condors (default 1)
    crr_steps          : binomial tree steps (default 100)
    """

    def __init__(
        self,
        call_short_offset:  float = 0.05,
        call_long_offset:   float = 0.10,
        put_short_offset:   float = -0.05,
        put_long_offset:    float = -0.10,
        days_to_expiry:     int   = 45,
        risk_free_rate:     float = 0.05,
        implied_volatility: Optional[float] = None,
        contracts:          int   = 1,
        crr_steps:          int   = 100,
    ):
        self.call_short_offset  = call_short_offset
        self.call_long_offset   = call_long_offset
        self.put_short_offset   = put_short_offset
        self.put_long_offset    = put_long_offset
        self.days_to_expiry     = days_to_expiry
        self.risk_free_rate     = risk_free_rate
        self.implied_volatility = implied_volatility
        self.contracts          = contracts
        self.crr_steps          = crr_steps

    def evaluate(self, spot_price: float, historical_vol: float = 0.20) -> CondorResult:
        sigma = self.implied_volatility or historical_vol
        T     = self.days_to_expiry / 365.0
        r     = self.risk_free_rate
        n     = self.crr_steps

        cs = round(spot_price * (1 + self.call_short_offset), 2)
        cl = round(spot_price * (1 + self.call_long_offset),  2)
        ps = round(spot_price * (1 + self.put_short_offset),  2)
        pl = round(spot_price * (1 + self.put_long_offset),   2)

        # Bear call spread: sell cs (short), buy cl (long) → credit
        call_credit = (crr_price(spot_price, cs, T, r, sigma, 'call', n) -
                       crr_price(spot_price, cl, T, r, sigma, 'call', n)) * self.contracts * 100

        # Bull put spread: sell ps (short), buy pl (long) → credit
        put_credit  = (crr_price(spot_price, ps, T, r, sigma, 'put', n) -
                       crr_price(spot_price, pl, T, r, sigma, 'put', n)) * self.contracts * 100

        net_credit   = call_credit + put_credit
        spread_width = (cl - cs) * self.contracts * 100
        max_profit   = net_credit
        max_loss     = spread_width - net_credit

        upper_be = cs + net_credit / (self.contracts * 100)
        lower_be = ps - net_credit / (self.contracts * 100)

        return CondorResult(
            call_credit=call_credit, put_credit=put_credit,
            net_credit=net_credit, max_profit=max_profit, max_loss=max_loss,
            upper_breakeven=upper_be, lower_breakeven=lower_be,
            profit_zone_width=upper_be - lower_be,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
        )

    def greeks(self, spot_price: float, historical_vol: float = 0.20) -> Dict:
        sigma = self.implied_volatility or historical_vol
        T     = self.days_to_expiry / 365.0
        r     = self.risk_free_rate
        n     = self.crr_steps
        dS    = spot_price * 0.01

        def net_price(s):
            cs = spot_price * (1 + self.call_short_offset)
            cl = spot_price * (1 + self.call_long_offset)
            ps = spot_price * (1 + self.put_short_offset)
            pl = spot_price * (1 + self.put_long_offset)
            return (
                - crr_price(s, cs, T, r, sigma, 'call', n)
                + crr_price(s, cl, T, r, sigma, 'call', n)
                - crr_price(s, ps, T, r, sigma, 'put',  n)
                + crr_price(s, pl, T, r, sigma, 'put',  n)
            )

        pu, pd = net_price(spot_price + dS), net_price(spot_price - dS)
        base   = net_price(spot_price)
        return {
            'net_delta': (pu - pd) / (2 * dS),
            'net_gamma': (pu - 2 * base + pd) / (dS ** 2),
        }


if __name__ == "__main__":
    strategy = IronCondorStrategy(days_to_expiry=45)
    result = strategy.evaluate(spot_price=150.0, historical_vol=0.20)
    print("Iron Condor Analysis (CRR Pricing)")
    print("=" * 45)
    print(f"Net Credit:        ${result.net_credit:.2f}")
    print(f"Max Profit:        ${result.max_profit:.2f}")
    print(f"Max Loss:          ${result.max_loss:.2f}")
    print(f"Upper Breakeven:   ${result.upper_breakeven:.2f}")
    print(f"Lower Breakeven:   ${result.lower_breakeven:.2f}")
    print(f"Profit Zone Width: ${result.profit_zone_width:.2f}")
    print(f"Risk/Reward:       {result.risk_reward_ratio:.2f}x")
