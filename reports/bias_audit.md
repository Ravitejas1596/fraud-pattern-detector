## Bias audit (limited slices)

This audit checks basic performance skews across a couple of available categorical slices. It is not a substitute for a full fairness review.

### Slice: ProductCD

| group   |     n |   fraud_rate |   mean_score |   roc_auc |
|:--------|------:|-------------:|-------------:|----------:|
| W       | 31699 |    0.0188334 |    0.0178998 |  0.907018 |
| C       |  5909 |    0.100186  |    0.0925368 |  0.937132 |
| H       |  5777 |    0.0282153 |    0.0291698 |  0.935227 |
| R       |  5633 |    0.0220131 |    0.0220831 |  0.950415 |
| S       |   982 |    0.0305499 |    0.0314242 |  0.979237 |


### Slice: DeviceType

| group   |     n |   fraud_rate |   mean_score |   roc_auc |
|:--------|------:|-------------:|-------------:|----------:|
| missing | 32896 |    0.019972  |    0.0187756 |  0.907488 |
| desktop | 10797 |    0.0391775 |    0.0385399 |  0.952158 |
| mobile  |  6307 |    0.067544  |    0.0640898 |  0.958122 |

