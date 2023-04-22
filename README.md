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
3. On your copy of `Main - with our research - DeepLineDP`, execute all the tasks: they include data preprocessing, training DeepLineDP, RandomForest, XGBoost, LighGBM models and predicting results. Predictions are stored as .csv files (one file per release of the library) located in `./comparisonModelsExperiment/output`, split by model folders: `prediction/DeepLineDP`, `RF-line-level-result`, `XGB-line-level-result`, `LGBM-line-level-result`.
4. The next step is to get the evaluation results: Recall@Top20LOC, Effort@Top20Recall and IFA. The workflow requires you to connect a google drive. The whole `./comparisonModelsExperiment` folder will be copied to your google drive as `comparisonModelsExperiment.zip`. You should then download the file from google drive and unzip on your local machine. Make sure that extracted folder contains `output` folder with prediction data - it should be there after running google colab workflow.
5. Furthermore, install R language for your system environment (https://cran.r-project.org/).
6. Open `./comparisonModelsExperiment` folder.
7. Make sure your current pwd is `./comparisonModelsExperiment`.
8. Navigate to `script`: `cd ./script`.
9.  Install required packages by running: `Rscript ./install_packages.R`.
10. Run to evaluate the results: `Rscript ./get_evaluation_result.R`.
11. The evaluation figures are located in `./comparisonModelsExperiment/output/figure` folder.
