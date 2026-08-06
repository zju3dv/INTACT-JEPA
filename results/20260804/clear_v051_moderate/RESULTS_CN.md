# CLEAR-LeWM v0.5.1 Moderate Direct Results

- Completion: `216/216`
- Status: `FINAL`
- Each task cell: 3 train seeds x 3 eval seeds x 100 episodes.
- Values are mean SR; +/- is sample std across the three train-seed means.

| Layout | Protocol | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|
| five_slot | p75 | 87.67+/-1.53 | 99.89+/-0.19 | 52.22+/-2.83 | 98.67+/-0.33 | 84.61+/-0.99 |
| five_slot | p77 | 87.44+/-1.84 | 100.00+/-0.00 | 48.44+/-2.01 | 98.89+/-0.77 | 83.69+/-0.17 |
| five_slot | p66 | 89.33+/-2.08 | 99.78+/-0.19 | 47.78+/-3.29 | 98.78+/-0.19 | 83.92+/-1.26 |
| four_slot | p75 | 88.78+/-2.83 | 100.00+/-0.00 | 48.22+/-3.34 | 96.78+/-0.19 | 83.44+/-0.47 |
| four_slot | p77 | 86.00+/-1.20 | 100.00+/-0.00 | 49.56+/-0.51 | 97.00+/-3.28 | 83.14+/-1.23 |
| four_slot | p66 | 87.78+/-2.83 | 99.89+/-0.19 | 48.89+/-3.02 | 96.11+/-5.59 | 83.17+/-1.88 |
