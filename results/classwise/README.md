# Class-level routing analysis

These files were derived from the complete saved decision logs for the matched
five-epoch CLEAR-100 runs. No model was retrained for this analysis.

Each class has 50 examples in the 2014 test bucket, so all 100 classes exceed
the predeclared minimum support of 20. F1 and IoU are calculated one class
against the remaining 99 classes. The classes are then ranked by historical F1
within each condition and split into four equal groups:

- Q1 contains the 25 weakest historical classes.
- Q4 contains the 25 strongest historical classes.

For the 2014 realistic mixture, the saliency-supervised router changes mean F1
relative to the historical model by +5.91, +2.00, +1.06 and -0.81 percentage
points from Q1 to Q4. The matched router without auxiliary saliency losses
changes the same strata by +2.91, +1.14, +1.16 and -0.44 points. The
saliency-supervised router therefore helps the weakest classes more under this
stress condition, but it does not protect the strongest classes completely.

On clean 2014 data, the corresponding saliency-supervised changes are +1.87,
-0.46, -0.46 and -0.47 points. The no-auxiliary-loss changes are +2.07, +0.02,
-0.32 and -0.54 points. This is evidence of a class-dependent trade-off rather
than a uniform gain.

The JSON file retains the aggregate comparisons and strata. The class metrics
CSV contains the complete class rows for inspection and plotting. The strata
CSV is the compact source used in the paper table. Helpful and harmful route
counts are grouped by the true class. Per-class precision and F1 also depend on false-positive
predictions arriving from other classes, so those counts do not fully explain a
class F1 change on their own.
