# DeepLineDP: Towards a Deep Learning Approach for Line-Level Defect Prediction

> Reproduction of _C. Pornprasit; C. Kla Tantithamthavorn, DeepLineDP: Towards a Deep Learning Approach for Line-Level Defect Prediction (2023)_.

## [📑 Article](./DeepLineDP_Towards_a_Deep_Learning_Approach_for_Line-Level_Defect_Prediction.pdf)

## [🌿 Overleaf](https://www.overleaf.com/project/6401cad83a0de65cdab78021)

## Team organisation:

[🛠️ Board](https://github.com/orgs/pwr-pbr23/projects/2/views/1)

[📦 Drive](https://drive.google.com/drive/folders/1_MXc-f5kqJA_23dCv05t6dJl375_0nnU?usp=sharing)

[📄 Team policy](./TeamPolicy.md)

[📄 Leader schedule](./TeamSchedule.md)

[📄 Team expectations agreement](./TeamExpectationsAgreement.md)

### Similar articles:

[LineDP: Predicting Defective Lines Using a Model-Agnostic Technique](https://www.computer.org/csdl/journal/ts/2022/05/09193975/1n0EsxgwzDy)

### Authors

- Jakub Tkaczyk
- Karol Waliszewski
- Mikołaj Macioszczyk

### Reproduction

Steps to replicate the study:
1. Access the shared file [🛠️ Main - with our research - DeepLineDp](https://colab.research.google.com/drive/1RRcg-vouL0gPwLS6mjSvePa06ikFZXzN?usp=sharing).
2. Create a copy of that file: `File -> Save a copy in Drive`.
3. On your copy of `Main - with our research - DeepLineDP`, execute all the tasks. They include data preprocessing, training DeepLineDP, RandomForest, XGBoost, LighGBM models, predicting results, top-k tokens investigation and comparison of the classifiers based on MCC, BA metrics. Predictions are stored as .csv files (one file per release of the library) located in `./comparisonModelsExperiment/output`, split by model folders: `prediction/DeepLineDP`, `RF-line-level-result`, `XGB-line-level-result`, `LGBM-line-level-result`. The comparison results of the classifiers with metrics are located in `./comparisonModelsExperiment/output/figure`.
4. The workflow asks you to connect a google drive. This is optional but will provide better user experience - it's possible to download all of the results and figures.
5. The next step is the evaluation results: Recall@Top20LOC, Effort@Top20Recall and IFA. After that the top-k investigation will be performed and at the end MCC, BA metrics will be calculated.
6.  The `./comparisonModelsExperiment/output` folder will be copied to your google drive as `comparisonModelsExperimentFigure.zip`. You can then download the file from google drive and unzip on your local machine.
7.  The evaluation results are located in `./figure` folder.
