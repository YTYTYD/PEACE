# PEACE
PEACE: A Dataset of Pharmaceutical Care for Cancer Pain Analgesia Evaluation and Medication Decision

# Abstract
Over half of cancer patients experience long-term pain management challenges. Recently, interest has grown in systems for cancer pain treatment effectiveness assessment (TEA) and medication recommendation (MR) to optimize pharmacological care. These systems aim to improve treatment effectiveness by recommending personalized medication plans based on comprehensive patient information. Despite progress, current systems lack multidisciplinary treatment (MDT) team assessments of treatment and the patient's perception of medication, crucial for effective cancer pain management. Moreover, managing cancer pain medication requires multiple adjustments to the treatment plan based on the patient's evolving condition, a detail often missing in existing datasets. To tackle these issues, we designed the PEACE dataset specifically for cancer pain medication research. It includes detailed pharmacological care records for over 38,000 patients, covering demographics, clinical examination, treatment outcomes, medication plans, and patient self-perceptions. Unlike existing datasets, PEACE records not only long-term and multiple follow-ups both inside and outside hospitals but also includes patients' self-assessments of medication effects and the impact on their lives. We conducted a proof-of-concept study with 11 machine learning algorithms on the PEACE dataset for the TEA (classification task) and MR (regression task). These experiments provide valuable insights into the potential of the PEACE dataset for advancing personalized cancer pain management. The dataset is accessible at: [https://github.com/YTYTYD/PEACE].

# For Reviewers
**See supplementary material for data**

# Usage
## Format
The PEACE is provided as a collection of compressed comma separated value (CSV) files.


## Application access
1. Complete some training such as the CITI (Collaborative Institutional Training Initiative at the University of Miami) “Data or Specimens Only Research” course as an MIT affiliate, as described in the instructions for completing required CITI training. Or you could provide a GCP certification (药物临床试验质量管理规范培训证书) .
2. Carefully read the terms of the [Data Use Agreement](https://github.com/YTYTYD/PEACE/blob/main/Data%20Use%20Agreement.docx), if you agree and wish to proceed, please send your application to the manager (admanoas@163.com, Jian Xiao). Please use an official e-mail address such as .edu 
3. Final approval of data access is required by Xiangya Hospital's Big Data Management Center.

Once an application has been approved, the researcher will receive emails containing instructions for downloading the dataset.


# Code availability
We present the results for 11 algorithms, which cover machine learning and deep learning algorithms, on the PEACE dataset for both tasks. These algorithms include 5 basic machine learning and neural network models: Decision Trees, Logistic Regression, Random Forests, SVM and MLP; 3 popular gradient boosting decision tree methods: LightGBM, XGBoost , and AdaBoost ; and 3 advanced neural network models designed for time-series data: iTransformer, TransTab, and Mamba.
The code is available at [ML code](https://github.com/YTYTYD/PEACE/tree/main/Code)



# Methods
The data used in this study was collected from two main sources. The first part originated from the Xiangya hospital, encompassing a broad range of patient information, including patient demographics, clinical signs, medication details, physiological parameters, and treatment outcomes. The second part of the data source is our cancer pain online follow-up platform. This platform allows continuous follow-up of cancer pain patients after hospital discharge through patient-initiated reports. It includes patient basic information, pain levels, adverse reactions from medication, dynamic adjustments to medication, treatment of adverse reactions, and other related data. 

The inclusion criteria for this research required subjects to have a definitive diagnosis of cancer with associated pain, confirmed via histopathological or cytological methods, with cancer being the primary diagnosis in their medical records. Exclusion criteria included cases with severely incomplete key medical records or significant medical complications. Our work is approved by the Institutional Review Board of the Xiangya Hospital. This work does not interfere with clinical care and treatment procedures. Informed consent is obtained from the patients, and all protected health information is de-identified.

# De-identification
In the collected data, patient identifiers were removed, and each patient was assigned a unique randomized code ID. Date and time values were shifted 30 to 80 years into the future using a personalized random offset measured in years. Each patient received an independent date transformation, ensuring that the temporal sequence within their data remained consistent. For instance, if the interval between two measurements in the original data was 15 days, the same interval was maintained in the PEACE dataset. However, temporal data for different patients are not comparable. This means that two patients treated in the year 2100 in the dataset are not necessarily treated in the same year in reality. Patients older than 89 years were uniformly labeled as 89 years old to protect their privacy, and patients younger than 18 years were excluded from the dataset. Finally, patient-specific diagnostic reports were reorganized, classified into different categories, and clearly labeled to facilitate data analysis and model training while ensuring privacy protection.

# Data Description
Our data construction process resulted in a comprehensive dataset encompassing 103 features, broadly categorized into six groups. The Patient Baseline Information group (50 features) captures demographic and clinical characteristics of the patients, potentially including age, gender, co-morbidities, and disease stage. The Comprehensive Pain Assessment group (15 features) details the extent and characteristics of the patients' pain experience, potentially including pain intensity scores, pain quality descriptors (e.g., visceral pain, somatic pain), and functional limitations. The Previous Analgesic Treatment group (23 features) details the medications and interventions previously used to manage the patients' pain, potentially including medication names, dosages, durations, and routes of administration. The Evaluation of Previous Analgesic Treatment group (5 features) captures the effectiveness and tolerability of prior pain management strategies, potentially including patient-reported outcomes or physician assessments. The Cancer Pain Medication Decision group (9 features) details the rationale behind the selection of specific pain medications for the study participants, potentially including factors like pain type, treatment history, and co-morbidities. The Follow-Up group (1 feature) captures information on patient outcomes after the intervention of interest, potentially including pain response or adverse events.

# Dataset Documentation
**Main Data:**
All_Data.csv: a .CSV file containing all patients in the dataset, with patient ID.


**Dictionaries:**
D\_ Numerical.csv: A .csv file containing the units of the numerical features
D\_ Multiclass.csv: A .csv file containing the meaning of multiclass features

**Model Training:**
All\_data.json: a .JSON file describing all the data in the dataset.
Train data: a .CSV file containing the training set of patients.
Test data: a .CSV file containing the test set of patients.




