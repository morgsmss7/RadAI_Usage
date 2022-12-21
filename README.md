# AI-Clinician Collaboration via Disagreement Prediction: A Framework and Analysis Using Real-World Data from a Deployed Radiology AI System

This repository contains code used in the ideation, development, and analysis of a novel AI-assistance framework for chest x-ray interpretation. We include two demo notebooks, one used to explore production data from an existing AI-assisted chest x-ray interpretation tool and one used for estimating changes in clinician burden between the novel and baseline pipelines. To learn more about the details of the pipelines and analysis performed, please refer to this document:  <a href="https://docs.google.com/document/d/19tmVz-mifbdNgVJqiWfnbzRC8oBEZYlMtYMCU3Pr084/edit?usp=sharing"><i>AI-Clinician Collaboration via Disagreement Prediction: A Framework and Analysis Using Real-World Data from a Deployed Radiology AI System
</i></a>.


### Table of Contents
- [Background](#background)
- [A Novel Framework for AI-Clinician Collaboration](#a-novel-framework-for-ai-clinician-collaboration)
- [Dataset](#dataset)
- [Burden Comparison](#burden-comparison)
- [Setup](#setup)

## Background
In recent years, significant strides in the machine learning (ML) for healthcare community have led to the integration of ML models into clinical workflows in a number of areas. From screening tools for diabetic retinopathy and colon cancer to diagnostic aids for pulmonary nodules and tuberculosis, artificial intelligence (AI) for clinical applications is becoming increasingly popular.

AI-based decision support tools typically aim to accomplish two major goals: improve patient care and reduce provider cost and burden. Numerous studies present evidence that AI-support can increase diagnostic performance or reduce diagnostic variability over clinicians working without AI support, but these potential benefits do not come without risks. To mitigate the consequences of poor AI performance, these models are often used in an assistive setting, where a final decision is still made by a clinician. This allows for clinicians to correct mistakes made by the AI, but it still puts a heavy burden on the user, even in easy cases where the AI and the clinician might agree.

Additionally, simply providing the clinician with all predicted diagnoses equivalently, regardless of accuracy or urgency, may limit the effectiveness of the tool and introduce dangerous biases. For instance, over time, clinicians may begin to rely too heavily on the tool, and default decisions to the model even in circumstances where there may otherwise be doubt as to the accuracy of the result. This is known as automation bias. On the other hand, clinicians may begin to ignore recommendations from the tool if they find themselves sorting through excessive false positives. Otherwise, they risk missing an important diagnosis, distracted by an overwhelming number of predictions to assess. This phenomenon is similar to what is known as alert fatigue, and is another example of how AI assistance integration can fail even if the model itself is performing relatively well.

## A Novel Framework for AI-Clinician Collaboration

To better understand where AI-clinician collaboration could improve and determine how to best protect against these risks, we analyze real-world production data from a chest x-ray interpretation aid deployed in Vietnam hospitals and study cases of AI-clinician disagreement (see Data_Exploration.ipynb). Inspired by this real-world data, we hypothesize that we can incorporate protections against common AI-safety issues while simultaneously reducing or maintaining clinician burden. In particular, we can use production data to predict disagreement, and motivate decisions as to when and how to present model output using characteristics such as the expected trustworthiness of a given prediction and whether the clinician is expected to agree with it along with clinical significance of each pathology. 

In this work, we develop a novel framework for AI-clinician collaboration and compare it to a baseline framework in terms of clinician burden (see Burden_Comparison_Demo.ipynb). The novel framework attempts to simultaneously accomplish three overarching goals. First, it assigns priority to findings based on expected disagreement and clinical significance and communicates that priority to the radiologist. Second, it attempts to automate inclusion or exclusion of findings from the report while minimizing risk of adverse effects, and finally, it attempts to improve report accuracy by automatically requesting a second opinion in rare cases of predicted uncertainty.

Concretely, in order to make use of disagreement, clinical significance, and AI quality modeling to inform when and how to present AI findings, the framework requires four primary components:
- Diagnostic Model: A model which predicts pathologies (e.g., based on a patient’s chest x-ray)
- Disagreement Model: A model that predicts whether the radiologist will agree or disagree with the AI prediction for a particular pathology
- Clinical Significance Categorization: A categorization that reflects clinical significance for each potential pathology. It is used to assign elevated priority to more urgent indications.
- Prediction Quality Model: For a particular Diagnostic Model prediction, a model that assigns a class regarding the trustworthiness of that prediction (e.g., either high or low)

Pathologies (i.e., findings) are assigned to one of four categories: Review Required, Quick Access, Automatic Inclusion, and Searchable. These categories are presented to a radiologist in order, as they are organized by decreasing demand for radiologist attention. Note, however, that findings in the Searchable category are not explicitly listed for review.

The “Review Required” category is for high clinical significance findings with predicted disagreement. Before submitting a report, the user is required to select “yes” or “no” for each of these findings. The “Quick Access” category, on the other hand, is for low clinical significance findings also with predicted disagreement. The user is expected to, at least, view these findings and add any at their discretion, but interaction is not necessary for report submission. For the “Automatic Inclusion” category, the description is quite self explanatory. These are findings predicted by the Diagnostic Model to be present in the image which have predicted agreement, and the user is only presented with the option to “remove” each finding. Finally, the “Searchable” category includes findings that the Diagnostic Model predicts are not present in the image for which there is also expected agreement. The user may search for the finding to add it to the report.

![baseline](/img/baseline.png)
![overview](/img/novel.png)


## Dataset
The data used in this work was collected in a post-deployment setting of <a href="https://vinbrain.net/teleradiology"><i>DrAid</i></a>, an AI-supported chest x-ray interpretation tool. DrAid employs an AI model that takes in a chest x-ray image and outputs the presence or absence of 21 pathologies and findings. In the interface, the radiologist is presented with (1) the patient’s medical and Rx history, (2) a chest x-ray in DICOM format,  (3) relevant demographics such as age and gender, (4) predicted pathologies and other findings, and (5) regions of interest (ROIs) associated with each AI-predicted finding.  The user can view ROIs, add AI findings to a report, request a second opinion, search for additional pathologies to add, and generate both internal and patient-accessible reports. 

Data consisted of chest x-ray images, predicted pathologies, final radiologist report inclusions, and demographics for 10,569 patients. Pathologies and findings detected include cardiomegaly, fracture, lung lesion, pleural effusion, pneumothorax, atelectasis, consolidation, pneumonia, edema, cavitation, fibrosis, enlarged cardiomediastinum, widening mediastinum, pleural other, medical device, COVID-19, mass, nodule, mass or nodule (unknown), lung opacity, and tuberculosis. COVID-19, however, was excluded from the analysis due to prediction quality.  One additional category, “other findings”, was also excluded from analysis due to lack of specificity. 

Data is explored in `Data_Exploration.ipynb`. 

## Burden Comparison

To estimate the theoretical overall effect of the novel pipeline on clinician workload, we will compare the proposed framework to a baseline framework, where a Diagnostic Model similar to the DrAid tool sorts findings into the “Quick Access” and “Searchable” categories, displaying all findings that the AI predicts are present to the user. A diagram of both pipelines is included above and a demo of burden comparison is included in `notebooks/Burden_Comparison_Demo.ipynb`.

In particular, we assign a cost to each potential user interaction in each of the pipelines and sort each finding from each patient in the Nam Dinh data into the appropriate leaves in both pipelines. Next, we determine the proportions of the data in each leaf and weight each leaf by total expected interactions. Finally, we sum these weighted proportions to obtain overall interaction burden for each pipeline and calculate the ratio of novel pipeline burden to baseline burden. 

In assigning interaction cost, we assign selection weight to selection interactions such as “yes”, “no”, “add”, or “remove”, search weight to findings added from the Searchable section, and image review weight to account for viewing the x-ray to consider that particular finding. Image review weight is assigned such that higher weight is applied to higher priority findings (i.e. Review Required > Quick Access > Auto-Include > Searchable), and such that the average expected weight in the baseline framework is about 30, which we expect to equate to approximately 30 seconds, but further data collection is required to obtain the most accurate weights. Weights used in this analysis (shown as constants in the demo notebook) are merely reasonable estimates and are not data driven. They are currently modeled as suggestions to the user (e.g. we would like the user to spend 4 times as much time on each review required finding as compared to each searchable one) and may not reflect actual time spent on each interaction.

For the novel pipeline, however, we are still missing three important pieces of information in the Nam Dinh data: clinical significance labels, prediction quality labels, and disagreement model predictions. For clinical significance, we assign a single label to each pathology represented in the Nam Dinh data with the help of experts, but we recognize that clinical significance is likely context-dependent. Though we use one static categorization for our analysis (see `sig_findings` and `non_sig_findings` in the demo), in a deployment setting, categorizations could be fluid, with the clinical indication informing finding placement.  

Additionally, since we do not have ground truth prediction quality labels and because accuracy of such labels will not affect burden in our analysis (only prevalence of positive predictions), we simulate second opinion requests at a variety of probabilities by drawing from Bernoulli distributions. Finally, because we aim to study the effect of different disagreement models on burden, we simulate disagreement model predictions for a variety of true positive rates and false positive rates. For samples with and without AI-Clinician disagreement, we again draw from Bernoulli distributions with probabilities of predicting disagreement equal to the TPR and FPR respectively.

A demo of burden estimation is included in the `notebooks` directory. Additionally, all functions used in the demo notebooks are documented and included in the utils.py file.

## Setup
### Create a Virtual Environment
```
$ python -m venv radAI
$ source radAI/bin/activate
(radAI) $ pip install -r envs/requirements.txt
```
### Add Your Virtual Environment to Jupyter
```
(radAI) $ pip install ipykernel
(radAI) $ python -m ipykernel install --user --name=radAI
```

Select the radAI kernel whenever attempting to run notebooks in the `notebooks` folder.
