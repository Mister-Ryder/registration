# Learn2Reg MR–CT protocol_300 headline benchmark

| ID | Method | Mean Dice ↑ | ASSD (mm) ↓ | HD95 (mm) ↓ | Fold fraction ↓ |
|---|---|---:|---:|---:|---:|
| B00 | Identity | 0.3709 | 15.63 | 39.43 | 0.000000 |
| B01 | ANTs-SyN + MI | 0.5742 | 10.32 | 28.64 | 0.000398 |
| B02 | ConvexAdam + MIND-SSC | 0.7350 | 6.30 | 20.38 | 0.015578 |
| B03 | FireANTs | 0.4712 | 13.51 | 36.01 | 0.000000 |
| B04 | DINO-Reg | 0.7821 | 4.96 | 23.35 | 0.041090 |
| B05 | MASR/DNS + IO | 0.3808 | 15.04 | 38.75 | 0.003903 |
| B06 | SynMSE | 0.3949 | 14.96 | 38.53 | 0.014497 |
| B07 | Locor | 0.6273 | 11.81 | 30.93 | 0.000000 |
| B08 | DGMIR-U | 0.3618 | 16.34 | 42.21 | 0.005154 |
| B09 | M2M-Reg | 0.3944 | 13.99 | 38.45 | 0.000039 |
| B10 | PRA-CM v3 | 0.4124 | 15.30 | 39.36 | 0.000647 |
| B11 | TransMorph + MIND-SSC | 0.3977 | 13.91 | 36.36 | 0.040049 |
| B12 | CorrMLP + MIND-SSC | 0.4997 | 13.03 | 36.71 | 0.024432 |

All rows are reproduced public-8 CT→MR results. Mean Dice/ASSD/HD95 are equal-weight means of the four per-organ means; unavailable structures are excluded. Cases 0002 and 0004 lack an evaluable left-kidney annotation, leaving 30/32 organ-cases. Validation requires eight pair metrics, eight successful statuses, and eight archived flow fields per method.
