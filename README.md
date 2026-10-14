# Fall Detection for Older Adults with a Waist-Worn Accelerometer

**Model comparison, sampling-rate reduction, age-group false alarms and a real-phone pilot test**

*[Bản tiếng Việt](README.vi.md)*

> **Abstract.** Wearable fall detectors are only useful if they catch falls *and* rarely raise false alarms, otherwise users stop wearing them. Using the public SisFall dataset (38 participants, 4,505 recordings, 23 young adults and 15 older adults), I compare a fixed-threshold baseline, a Random Forest on orientation-independent features and two 1D-CNNs, evaluated with subject-wise 5-fold cross-validation. The Random Forest detects 97.6% of falls with a 1.1% false-alarm rate, about 20× fewer false alarms than the threshold baseline (21.1%), and keeps the same accuracy when the sampling rate is reduced 16-fold (200 Hz → 12.5 Hz), which matters for battery life. Contrary to my hypothesis, false alarms were not higher for older adults, even when comparing only activities performed by both groups. Finally, a pilot test on 19 recordings made with a hand-held phone (9 simulated falls onto a mattress, 10 daily activities) exposed a large gap: false alarms rose from 0.4–2.6% on SisFall to 3–5 out of 10 everyday activities, mostly when sitting down hard. Because the phone was held in the hand rather than worn at the waist, this gap mixes a change of device with a change of sensor position.

---

## 1. Problem

Falls are a leading cause of injury in older adults, and they are most dangerous when the person lives alone: they can lie on the floor for hours before anyone notices. Smartwatches and phones already contain accelerometers, so in principle they can detect a fall automatically and call for help.

The hard part is not detecting falls but avoiding **false alarms**. Sitting down hard on a chair or jumping to reach a shelf also produces a jolt that looks like a fall. A device that raises too many false alarms gets taken off and left in a drawer.

## 2. Research questions

| | Question |
|---|---|
| **RQ1** | How much do machine-learning models reduce false alarms compared with the "acceleration above a threshold means a fall" rule used by many simple devices? |
| **RQ2** | How far can the sampling rate be reduced (to save battery) without losing accuracy? |
| **RQ3** | Do the models raise more false alarms for **older adults**? The concern comes from fall data being recorded almost entirely by young volunteers. |
| **RQ4** | Do models trained on lab data still work on a real phone? |

## 3. Data

[SisFall](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/) (Sucerquia et al., 2017): an ADXL345 accelerometer worn at the waist, sampled at 200 Hz.

| | Participants | Daily activities | Falls |
|---|---|---|---|
| Young adults (19–30) | 23 | 1,809 | 1,723 |
| Older adults (60–75) | 15 | 898 | 75 |
| **Total** | **38** | **2,707** | **1,798** |

The dataset has 19 types of daily activity (walking, jogging, stairs, sitting, lying down…) and 15 types of fall (slips, trips, fainting…). Only one older participant (SE06) performed falls, and only this participant performed 4 vigorous activities: fast stairs (D06), lying down quickly (D13), stumbling while walking (D18) and a gentle jump (D19).

![Example signals](figures/4_example_signals.png)

*Figure 1. A fall produces an impact peak of about 14g followed by oscillations; sitting down hard on a chair peaks at about 5g.*

## 4. Method

**Preprocessing.** Raw values are converted to g. For each recording, a 3.2-second window is cut around the moment of maximum total acceleration (1.2 s before, 2 s after). For daily activities this is the most fall-like moment, so the models are tested on the hardest case.

**Models.**
1. **Fixed threshold** (baseline): a fall if total acceleration exceeds a threshold T, with T chosen on the training folds to balance sensitivity and specificity.
2. **Random Forest** (300 trees) on 10 **orientation-independent features**: statistics of the acceleration magnitude, the "free-fall" minimum before impact, variability after impact, the posture-change angle (direction of gravity before vs after) and the fraction of time above 2g.
3. **1D-CNN** (4 convolutional blocks) on the raw signal, in two variants: **all 3 axes** (`xyz`) and **magnitude only** (`mag`, independent of how the device is worn).

**Evaluation.** Subject-wise 5-fold cross-validation (GroupKFold): every participant is scored by a model that never saw their data. Splitting by recording instead would put the same person in both training and test sets and inflate the results.

Metrics: **sensitivity** (% of falls detected), **false-alarm rate** (% of daily activities flagged as falls = 1 − specificity) and F1.

## 5. Results

### RQ1: Machine learning vs a fixed threshold

| Model (200 Hz) | Falls detected | False alarms | F1 |
|---|---|---|---|
| Fixed threshold | 90.8% | 21.1% | 0.816 |
| Random Forest | 97.6% | 1.1% | 0.979 |
| CNN (magnitude) | 97.1% | 2.6% | 0.966 |
| CNN (3-axis) | **99.6%** | **0.4%** | **0.995** |

![Model comparison](figures/1_model_comparison.png)

The Random Forest cuts false alarms about **20-fold** compared with the fixed threshold while detecting more falls. The 3-axis CNN is the best model on SisFall, but because it relies on axis directions it may be sensitive to how the device is worn (see RQ4).

### RQ2: Reducing the sampling rate

| Sampling rate | Random Forest F1 | CNN (magnitude) F1 |
|---|---|---|
| 200 Hz | 0.979 | 0.966 |
| 100 Hz | 0.979 | 0.971 |
| 50 Hz | 0.977 | 0.980 |
| 25 Hz | 0.977 | 0.976 |
| 12.5 Hz | **0.985** | **0.981** |

![Sampling rate](figures/2_sampling_rate.png)

Reducing the sampling rate **16-fold** (200 Hz → 12.5 Hz) does not reduce accuracy. On a wearable, sampling less often lowers the power drawn by both the sensor and the processor.

### RQ3: Young vs older adults

| Model | False alarms: young | False alarms: older | Shared activities only: young | Shared activities only: older |
|---|---|---|---|---|
| Fixed threshold | 27.4% | 8.5% | 13.6% | 7.0% |
| Random Forest | 1.2% | 0.9% | 1.5% | 0.9% |

![Young vs older adults](figures/3_young_vs_older.png)

**The hypothesis was not supported:** older adults did not trigger more false alarms, even after removing the 4 vigorous activities that only one older participant performed. A plausible explanation is that older adults move more gently and produce fewer fall-like jolts. The large age gap for the threshold method comes mainly from young adults' vigorous activities.

**Activities the Random Forest most often mistook for falls:** "trying to stand up and collapsing back into a chair" (D11, 8.4%) and "turning over while lying down" (D14, 6.3%). Both combine a strong impact with a change of posture, just like a fall.

### RQ4: Testing on a real phone

I recorded my own data with the phyphox app on an iPhone **held in the hand** ("Acceleration with g" experiment, 100 Hz, interpolated to 200 Hz to match SisFall): **9 simulated falls onto a mattress** and **10 daily activities** (3 lying down, 3 sitting down, 4 walking). One fall recording was excluded because it captured the magnetometer instead of the accelerometer. The models were used exactly as trained on SisFall, with no further tuning.

| Model | Falls detected | False alarms on phone | False alarms on SisFall |
|---|---|---|---|
| Fixed threshold | 9/9 | 4/10 | 21.1% |
| Random Forest 200 Hz | 7/9 | 4/10 | 1.1% |
| Random Forest 12.5 Hz | 8/9 | 4/10 | 0.7% |
| CNN (magnitude) 200 Hz | 6/9 | 3/10 | 2.6% |
| CNN (magnitude) 12.5 Hz | 7/9 | 5/10 | 0.9% |
| CNN (3-axis) 200 Hz | 9/9 | 4/10 | 0.4% |

![Phone test](figures/5_phone_test.png)

**Lab performance did not transfer to the real world.** False alarms rose from about 1% to 30–50%. The first "sitting down" recording was flagged as a fall by **all 4 main models**: sitting down hard produced a 5–6g peak, and it is also the hardest activity on SisFall (D11). Walking was confused less often: each model flagged at most 1 of the 4 walking recordings. The 3-axis CNN still detected 9/9 falls, but with a hand-held phone nothing can be concluded about the effect of device orientation.

The sample is very small (19 recordings, one person), so these rates are rough estimates: 4/10 is consistent with anything from about 15% to 70%. Even so, the gap from about 1% on SisFall is large enough to conclude that **a model must be evaluated on real devices before its benchmark accuracy can be trusted**. The most likely main cause is **sensor position**: SisFall was recorded at the waist, while I held the phone in my hand, so arm movements (swinging while walking, bracing when sitting) are added to the signal. The phone's sensor also differs from the ADXL345, and I may have sat down harder than the SisFall participants. The next experiment is to repeat the recordings with the phone worn at the waist, to isolate the effect of position.

## 6. Limitations

- **Simulated falls, not real ones**, performed almost entirely by young adults (1 of 15 older participants). This is a common limitation of fall datasets.
- Each recording contributes **a single window around the impact peak**. A real device must process a continuous signal, so the real-world false-alarm rate (per hour of wear) may differ.
- Only **one dataset and one sensor position** (waist).
- One random seed per configuration; no confidence intervals yet.
- The phone test has only 19 recordings from one person, with simulated falls onto a mattress, and the phone was **held in the hand** rather than worn at the waist as in SisFall.

## 7. Future work

- **Close the lab-to-real gap:** fine-tune the model on a few minutes of the user's own daily activity (personalisation), or train on phone data as well.
- Evaluate on continuous signals and measure **false alarms per hour of wear**.
- Test on other datasets (wrist position, different sensors) to measure generalisation.
- Use a longer post-impact window ("does the person stay still after the impact?") to reduce confusion with D11 and D14.
- Run a small model directly on a microcontroller or smartwatch.

## 8. Reproducing the experiments

```bash
pip install torch
pip install -r requirements.txt
python prepare_sisfall.py --root <path to SisFall_dataset>
bash run_all.sh                 # 12 experiments; results table in results/results_table.md
python predict_phone.py --dir phone_data --model results/rf_200hz.pkl   # repeat for each model
python make_figures.py          # adds figure 5 once phone results exist
```

Step-by-step guide (Vietnamese): [HUONG_DAN.md](HUONG_DAN.md). KNIME analysis (Vietnamese): [KNIME.md](KNIME.md).

## References

Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A Fall and Movement Dataset. *Sensors*, 17(1), 198.
