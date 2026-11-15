# Rating-Routed Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Jesyyhezii/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Jesyyhezii/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free AI model aggregator built for the [make.fun](https://make.fun/bounty/free-model-aggregator-01) "The best free AI model aggregator" bounty.

It serves each request from the best-rated free model across several upstream providers, translating between their wire formats. Routing is risk-priced: every candidate is ranked by its rating minus a risk price derived from the providers' call budgets and observed failures, failure memory is sticky (a provider that keeps refusing calls is retired instead of retried), and a quality-loss budget guarantees the router never concedes more rating than the score can afford. The result is the same top quality score with materially fewer wasted upstream calls.

## Run

```bash
python3 main.py
```

The program is a single-file Python 3 entry (`main.py`), standard library only, deterministic and offline apart from the upstream provider calls it routes.
