# POMDP Reward Equation

Date: 2026-08-12

## Definition

For one hierarchy level, the hidden state is the true geographic hypothesis
`l`, the belief state is the current posterior `b_t(l)`, and each action `a_t`
is a candidate verification task.

The observation `o_t` is the predicted support rating from the verification
model, where `o_t in {1, 2, 3, 4, 5}`.

## Observation Likelihood

For each candidate hypothesis and action, the planner estimates:

```text
P(o | l, a)
```

using repeated MLLM samples.

The marginal observation probability under the current belief is:

```text
P(o | b_t, a_t) = sum_l b_t(l) P(o | l, a_t)
```

## Belief Update

After observing `o`, the hypothetical next belief is:

```text
b_{t+1}^o(l) = eta b_t(l) P(o | l, a_t)
```

where `eta` is the normalization constant.

## Reward

The thesis reward combines uncertainty reduction, top-hypothesis confidence gain,
and ranking-margin gain:

```text
R(b_t, a_t, o, b_{t+1}) =
    lambda_H [H(b_t) - H(b_{t+1})]
  + lambda_M [max_l b_{t+1}(l) - max_l b_t(l)]
  + lambda_G [margin(b_{t+1}) - margin(b_t)]
  - cost(a_t)
```

where:

```text
H(b) = -sum_l b(l) log b(l)
margin(b) = top_1(b) - top_2(b)
```

## Action Value

The selected action maximizes the expected one-step POMDP return:

```text
Q(b_t, a_t) = sum_o P(o | b_t, a_t) R(b_t, a_t, o, b_{t+1}^o)
a_t* = argmax_a Q(b_t, a)
```

## Code Switch

Use the full reward equation with:

```bash
POMDP_POLICY=eig \
POMDP_REWARD_MODE=combined \
POMDP_ENTROPY_WEIGHT=1.0 \
POMDP_MAP_GAIN_WEIGHT=1.0 \
POMDP_MARGIN_GAIN_WEIGHT=0.5 \
python evaluate.py --limit 1500 --batch_size 2 --out results/pomdp_combined_limit1500.json
```

Legacy ablations remain available:

```text
POMDP_REWARD_MODE=entropy
POMDP_REWARD_MODE=map_gain
POMDP_REWARD_MODE=margin_gain
```

