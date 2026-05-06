export interface IronCondorConfig {
  callShortOffset: number;   // short call % above spot  (default +0.05 = 5% OTM)
  callLongOffset: number;    // long call % above spot   (default +0.10 = 10% OTM)
  putShortOffset: number;    // short put % below spot   (default -0.05 = 5% OTM)
  putLongOffset: number;     // long put % below spot    (default -0.10 = 10% OTM)
  daysToExpiry: number;      // days until expiry (default 45)
  riskFreeRate: number;      // annual risk-free rate (default 0.05)
  impliedVolatility?: number; // IV override
  contracts: number;         // number of condors (default 1)
  crrSteps: number;          // binomial tree steps (default 100)
}

export const defaultConfig: IronCondorConfig = {
  callShortOffset: 0.05,
  callLongOffset: 0.10,
  putShortOffset: -0.05,
  putLongOffset: -0.10,
  daysToExpiry: 45,
  riskFreeRate: 0.05,
  contracts: 1,
  crrSteps: 100,
};

/**
 * Cox-Ross-Rubinstein binomial tree for American-style option pricing
 */
export function crrPrice(
  S: number, 
  K: number, 
  T: number, 
  r: number, 
  sigma: number,
  optionType: 'put' | 'call' = 'call',
  steps: number = 100
): number {
  if (T <= 0 || sigma <= 0) {
    return Math.max(optionType === 'call' ? S - K : K - S, 0.0);
  }
  
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1.0 / u;
  const disc = Math.exp(-r * dt);
  const p = (Math.exp(r * dt) - d) / (u - d);
  
  // Initialize terminal values
  const values: number[] = [];
  for (let j = 0; j <= steps; j++) {
    const spot = S * Math.pow(u, j) * Math.pow(d, steps - j);
    values.push(optionType === 'call' ? Math.max(spot - K, 0.0) : Math.max(K - spot, 0.0));
  }
  
  // Backward induction
  for (let i = steps - 1; i >= 0; i--) {
    for (let j = 0; j <= i; j++) {
      const cont = disc * (p * values[j + 1] + (1 - p) * values[j]);
      // American-style exercise
      const spot = S * Math.pow(u, j) * Math.pow(d, i - j);
      const intrinsic = optionType === 'call' ? Math.max(spot - K, 0.0) : Math.max(K - spot, 0.0);
      values[j] = Math.max(intrinsic, cont);
    }
  }
  
  return Math.round(values[0] * 1000000) / 1000000; // Round to 6 decimal places
}

export interface CondorResult {
  callCredit: number;
  putCredit: number;
  netCredit: number;
  maxProfit: number;
  maxLoss: number;
  upperBreakeven: number;
  lowerBreakeven: number;
  profitZoneWidth: number;
  riskRewardRatio: number;
}

export class IronCondorStrategy {
  private callShortOffset: number;
  private callLongOffset: number;
  private putShortOffset: number;
  private putLongOffset: number;
  private daysToExpiry: number;
  private riskFreeRate: number;
  private impliedVolatility: number | undefined;
  private contracts: number;
  private crrSteps: number;

  constructor(config: IronCondorConfig = defaultConfig) {
    this.callShortOffset = config.callShortOffset;
    this.callLongOffset = config.callLongOffset;
    this.putShortOffset = config.putShortOffset;
    this.putLongOffset = config.putLongOffset;
    this.daysToExpiry = config.daysToExpiry;
    this.riskFreeRate = config.riskFreeRate;
    this.impliedVolatility = config.impliedVolatility;
    this.contracts = config.contracts;
    this.crrSteps = config.crrSteps;
  }

  evaluate(spotPrice: number, historicalVol: number = 0.20): CondorResult {
    const sigma = this.impliedVolatility ?? historicalVol;
    const T = this.daysToExpiry / 365.0;
    const r = this.riskFreeRate;

    const callShort = Math.round(spotPrice * (1 + this.callShortOffset) * 100) / 100;
    const callLong = Math.round(spotPrice * (1 + this.callLongOffset) * 100) / 100;
    const putShort = Math.round(spotPrice * (1 + this.putShortOffset) * 100) / 100;
    const putLong = Math.round(spotPrice * (1 + this.putLongOffset) * 100) / 100;

    // Bear call spread: sell callShort (short), buy callLong (long) → credit
    const callCredit = (
      crrPrice(spotPrice, callShort, T, r, sigma, 'call', this.crrSteps) -
      crrPrice(spotPrice, callLong, T, r, sigma, 'call', this.crrSteps)
    ) * this.contracts * 100;

    // Bull put spread: sell putShort (short), buy putLong (long) → credit
    const putCredit = (
      crrPrice(spotPrice, putShort, T, r, sigma, 'put', this.crrSteps) -
      crrPrice(spotPrice, putLong, T, r, sigma, 'put', this.crrSteps)
    ) * this.contracts * 100;

    const netCredit = callCredit + putCredit;
    const spreadWidth = (callLong - callShort) * this.contracts * 100;
    const maxProfit = netCredit;
    const maxLoss = spreadWidth - netCredit;

    const upperBreakeven = callShort + netCredit / (this.contracts * 100);
    const lowerBreakeven = putShort - netCredit / (this.contracts * 100);
    const profitZoneWidth = upperBreakeven - lowerBreakeven;
    const riskRewardRatio = maxLoss > 0 ? maxProfit / maxLoss : 0;

    return {
      callCredit,
      putCredit,
      netCredit,
      maxProfit,
      maxLoss,
      upperBreakeven,
      lowerBreakeven,
      profitZoneWidth,
      riskRewardRatio,
    };
  }
}
