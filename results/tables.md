# Results Tables

## Table 1 - Data

| Item | Value |
|---|---|
| Train / validation / test windows | 5891 / 1461 / 2947 |
| Train / validation / test volunteers | 17 / 4 / 9 |

Mean and standard deviation of each channel (training windows only):

| Channel | Mean | Std |
|---|---|---|
| body_acc_x | -0.0007 | 0.1990 |
| body_acc_y | -0.0003 | 0.1226 |
| body_acc_z | -0.0003 | 0.1091 |
| body_gyro_x | -0.0008 | 0.4016 |
| body_gyro_y | -0.0008 | 0.3894 |
| body_gyro_z | -0.0003 | 0.2652 |
| total_acc_x | 0.8000 | 0.4195 |
| total_acc_y | 0.0207 | 0.3832 |
| total_acc_z | 0.0765 | 0.3768 |

## Table 2 - Network and training

| Item | Value |
|---|---|
| Trainable parameters | 46982 |
| Initial loss (before training) | 1.7770 |
| Best epoch (lowest validation loss) | 10 |
| Training / validation loss at best epoch | 0.1081 / 0.0787 |
| Validation accuracy at best epoch | 0.9726 |
| Training time and hardware | 12.3 min, CPU |

## Table 3 - Test results

| Metric | Walk | Up | Down | Sit | Stand | Lay |
|---|---|---|---|---|---|---|
| Per-class accuracy | 0.9052 | 0.9851 | 0.9429 | 0.8330 | 0.8402 | 0.9497 |
| Mean of max_t alpha_t | 0.2792 | 0.1550 | 0.2707 | 0.0152 | 0.0164 | 0.0110 |

**Overall Test Accuracy: 0.9077** (90.77%)

### Full 6 x 6 Test Confusion Matrix (rows = true activity, columns = predicted)

| True \ Predicted | Walk | Up | Down | Sit | Stand | Lay |
|---|---|---|---|---|---|---|
| WALKING | 449 | 13 | 34 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 4 | 464 | 2 | 1 | 0 | 0 |
| WALKING_DOWNSTAIRS | 5 | 18 | 396 | 0 | 1 | 0 |
| SITTING | 0 | 3 | 0 | 409 | 77 | 2 |
| STANDING | 0 | 0 | 0 | 85 | 447 | 0 |
| LAYING | 0 | 27 | 0 | 0 | 0 | 510 |
