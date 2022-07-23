# Efficient and Accurate Integration of Diagnostic Machine Learning Models into Radiology Reporting Workflows

This repository contains code used in the ideation, development, and analysis of a novel AI-assistance framework for chest x-ray interpretation. We include two demo notebooks, one used to explore production data from an existing AI-assisted chest x-ray interpretation tool and one used for estimating changes in clinician burden between the novel and baseline pipelines. To learn more about the details of the pipelines and analysis performed, please refer to the manuscript <a href="https://docs.google.com/document/d/19tmVz-mifbdNgVJqiWfnbzRC8oBEZYlMtYMCU3Pr084/edit#"><i>Efficient and Accurate Integration of Diagnostic Machine Learning Models into Radiology Reporting Workflows</i></a>.


### Table of Contents
- Overview
	- Workflow
- Setup / Usage
- Dataset
- Estimate Reduced Burden
- License
- Citation

## Overview
In recent years, significant strides in the machine learning (ML) for healthcare community have led to the integration of ML models into clinical workflows in a number of areas. From screening tools for diabetic retinopathy (Grzybowski et al.; Beede et al.) and colon cancer (Mori et al.; Liu et al.) to diagnostic aids for pulmonary nodules (Nam et al.) and tuberculosis (Lee et al.), artificial intelligence (AI) for clinical applications is becoming increasingly popular.

AI-based decision support tools typically aim to accomplish two major goals: improve patient care and reduce provider cost and burden. Numerous studies present evidence that AI-support can increase diagnostic performance or reduce diagnostic variability over clinicians working without AI support (Han et al.; Liu et al.; Nam et al.). These potential benefits do not come without risks, however, as the AI can fail in any deployment setting for a number of reasons, including distribution shift between training and deployment and underperformance in particular subpopulations (Chen et al.; Finlayson et al.) To mitigate the consequences of poor AI performance, these models are often used in an assistive setting, where a final decision is still made by a clinician. This allows for clinicians to correct mistakes made by the AI, but it still puts a heavy burden on the user, even in easy cases where the AI and the clinician might agree.

## Workflow
To better understand where AI-clinician collaboration could improve and determine how to best protect against these risks, we analyze real-world production data from a chest x-ray interpretation aid deployed in Vietnam hospitals and study cases of AI-clinician disagreement. Inspired by this real-world data, we hypothesize that we can incorporate protections against common AI-safety issues while simultaneously reducing or maintaining clinician burden. In particular, we can use production data to predict disagreement, and motivate decisions as to when and how to present model output using characteristics such as the expected trustworthiness of a given prediction and whether the clinician is expected to agree with it along with clinical significance of each pathology.

The proposed framework attempts to simultaneously accomplish three overarching goals. First, it assigns priority to findings based on expected disagreement and clinical significance and communicates that priority to the radiologist. Second, it attempts to automate inclusion or exclusion of findings from the report while minimizing risk of adverse effects, and finally, it attempts to improve report accuracy by automatically requesting a second opinion in rare cases of predicted uncertainty.
Concretely, in order to make use of disagreement, clinical significance, and AI quality modeling to inform when and how to present AI findings, the framework requires four primary components:
	Diagnostic Model: A model which predicts pathologies (e.g., based on a patient’s chest x-ray)
	Disagreement Model: A model that predicts whether the radiologist will agree or disagree with the AI prediction for a particular pathology
	Clinical Significance Categorization: A categorization that reflects clinical significance for each potential pathology. It is used to assign elevated priority to more urgent indications.
	Prediction Quality Model: For a particular Diagnostic Model prediction, a model that assigns a class regarding the trustworthiness of that prediction (e.g., either high or low)

Pathologies (i.e., findings) are assigned to one of four categories: Review Required, Quick Access, Automatic Inclusion, and Searchable. These categories are presented to a radiologist in order, as they are organized by decreasing demand for radiologist attention. Note, however, that findings in the Searchable category are not explicitly listed for review.

The “Review Required” category is for high clinical significance findings with predicted disagreement. Before submitting a report, the user is required to select “yes” or “no” for each of these findings. The “Quick Access” category, on the other hand, is for low clinical significance findings also with predicted disagreement. The user is expected to, at least, view these findings and add any at their discretion, but interaction is not necessary for report submission. For the “Automatic Inclusion” category, the description is quite self explanatory. These are findings predicted by the Diagnostic Model to be present in the image which have predicted agreement, and the user is only presented with the option to “remove” each finding. Finally, the “Searchable” category includes findings that the Diagnostic Model predicts are not present in the image for which there is also expected agreement. The user may search for the finding to add it to the report.

![overview](/img/overview.png)

## Setup / Usage
#### Using conda
```
$ conda env create --file disagree.yml
$ conda activate disagree`
(disagree) $
```

All functions used in the demo notebooks are documented and included in the utils.py file.


## Dataset
The data used in this work was collected in a post-deployment setting of DrAidTM,  an AI-supported chest x-ray interpretation tool. DrAidTM employs an AI model that takes in a chest x-ray image and outputs the presence or absence of 21 pathologies and findings. In the interface, the radiologist is presented with (1) the patient’s medical and Rx history, (2) a chest x-ray in DICOM format,  (3) relevant demographics such as age and gender, (4) predicted pathologies and other findings, and (5) regions of interest (ROIs) associated with each AI-predicted finding.  The user can view ROIs, add AI findings to a report, request a second opinion, search for additional pathologies to add, and generate both internal and patient-accessible reports. 

The data consisted of chest x-ray images, predicted pathologies, final radiologist report inclusions, and demographics for 10,569 patients. Pathologies and findings detected include cardiomegaly, fracture, lung lesion, pleural effusion, pneumothorax, atelectasis, consolidation, pneumonia, edema, cavitation, fibrosis, enlarged cardiomediastinum, widening mediastinum, pleural other, medical device, COVID-19, mass, nodule, mass or nodule (unknown), lung opacity, and tuberculosis. COVID-19, however, was excluded from the analysis due to prediction quality.  One additional category, “other findings”, was also excluded from analysis due to lack of specificity. 

The average rate of disagreement between the radiologist label and the diagnostic model label for the Nam Dinh Hospital sample was 5.4%. The average true positive rate (TPR) and false positive rate (FPR) of DrAidTM over all pathologies using the final report as ground truth were 0.73 and 0.041 respectively. These estimates, especially the low disagreement rate, inspired a novel pipeline which uses predicted agreement to organize predictions and protect against biases introduced by using an AI-based interpretation aid.

## Estimate Reduced Burden
