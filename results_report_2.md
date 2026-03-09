# 📑 NYC Taxi Prediction: Data & Model Versioning (PRACTICE 2)

## 1. Executive Summary

This report details the implementation of **Data and Model Versioning** using DVC (Data Version Control) and Git for the NYC Green Taxi Tip **Regression** pipeline.

The objective of this practice was to simulate a production environment where new data arrives over time. 

- Baseline model (Version 1), simulated the arrival of new data to observe model degradation (concept/data drift)
- Retrained the model in the full data (old + new). 
- Finally, a production monitoring and rollback strategy was defined.

---

## 2. Versioning Strategy (DVC & Git)

* **Git:** Tracks the lightweight `.dvc` pointer files, configuration scripts (`config.py`), and the training/evaluation logic (`run.py`, `tests.py`).
* **DVC:** Computes hashes for the heavy `.parquet` data files and `.pkl` model artifacts, storing the actual files in a configured Google Drive remote storage.

---

## 3. Experimental Results: Concept Drift & Retraining

To evaluate the impact of new data over time, three specific states were evaluated on the 5-feature XGBoost Regression model.

### Performance Results

*Note: For Practice 2, Tip Amount regression, only rows where payment_type == 1 were considered, since all the others have tip = 0. This led to model_v3, with better results in comparison with Practice 1.*
*Note: `Data01` refers to January data. `Data02` refers to the combined January + February data.*


| Model Tag | State Description | Test MAE ($) | Test RMSE | CV MAE (Avg) |
| --- | --- | --- | --- | --- |
| **`xgb_5feats_v3_data01`** | **Baseline:** V3 Model evaluated on V1 Data | **$1.090** | **1.750** | 1.294 |
| `xgb_5feats_v3_data02` | **Drifted:** V3 Model evaluated on V2 Data | $1.178 | 2.01 | NaN |
| `xgb_5feats_v4_data02` | **Retrained:** V4 Model trained/evaluated on V2 Data | $1.185 | 2.086 | 1.165 |

### Analysis & Interpretation

1. **Baseline vs. Drift (`v3_data01` vs `v3_data02`):** When the exact same Version 3 model was evaluated on the newly arriving February data, the Mean Absolute Error (MAE) degraded from $1.090 to $1.178. This demonstrates clear **data drift**—the underlying patterns of taxi tips shifted between January and February.
2. **Drift vs. Retraining (`v3_data02` vs `v4_data02`):** Trying counteract the drift, a new model version was trained from scratch on the full `Data02` dataset. The retrained model achieved an even _worse_ MAE ($1.85) than the old model operating on the new data ($1.178). This probably indicates a radical change between ``Data01`` and ``Data02``, making the model struggle to generalize with both data. This indicates the need to further investigate the data from February to catch possible missing features, strong distribution deviations, etc.

**Conclusion:** The baseline V3 model tested on V1 data holds the best metrics overall, although it clearly suffered some concept drift with the data update. The fact that the model didn't improve training on the whole set indicates the need for further investigations on the data analysis and possibly training the model with only the new data, looking for a global pattern changing and the old data became useless.

---

## 4. Production Monitoring Strategy (Practice 2 questions)

Assuming the newly retrained model is deployed into production, the following monitoring architecture would be implemented to ensure system health and predictive quality.

A. Metrics to Monitor

1. **Data-Related Metric: Missing or Unrecognized `DOLocationID` Percentage**
* *Reason:* As described in Practice 1, the locations IDs are impactful but possibly fragile features. If the ID zones change, the model will recive unexpected `NaNs`. It would be relevant to track the percentage of missing IDs or new zones not present before, alerting pipeline failures before they ruin predictions. Similar actions should be considered for every features.


2. **Model-Performance Metric: 7-Day Rolling MAE**
* *Reason:* MEA could be the primary business metric. By joining the model's tip predictions with the actual tip amounts recorded in the database after the trips conclude (automatic real labeling), we can calculate the real-world MAE over a rolling 7-day window to catch gradual degradation and trigger the concept drift.


3. **System Metric: Error Rate (HTTP 500s / Exception Rate)**

**PS: I am not experienced in this part, so I googled possible outcomes**

* *Reason:* This tracks the percentage of prediction requests that result in a system crash. While the XGBoost model itself is fast, the surrounding API and preprocessing pipeline can easily fail if fed malformed data types (e.g., receiving a string instead of an expected integer).



B. Lifecycle Actions 

**When to Retrain:**

* **Threshold Trigger:** If the 7-Day Rolling MAE degrades consistently over a defined baseline threshold (e.g., an increase of >10% over two consecutive weeks), an automated retraining pipeline should be triggered using the most recent data.
* **Schedule Trigger:** Since taxi behavior is highly seasonal, retraining on a set schedule (e.g., the 1st of every month) is recommended to naturally capture slow-moving concept drift.
* **Temporal modelling:** A more complex but possibly more performatic action could be the temporal data analysis to catch possible long-term dependecies (e.g., trends in each month of the year) and use a time series-dependent model to make predictions considering the date as a feature.

**When to Roll Back:**

* **System Failure:** If the Error Rate spikes above 1% immediately after deploying a new model version, it indicates an integration bug or schema mismatch. The system should automatically roll back to the previous stable version.
* **Data Outage:** If a severe data pipeline failure occurs (e.g., the service providing location features goes entirely offline), we must temporarily roll back to a simpler, fallback model (e.g., a 3-feature model) that does not require those specific inputs to function.