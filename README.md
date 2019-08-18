### To install
To install global dependencies if they are needed:
```
sudo apt-get install python3.6-tk
```
In the project's directory:
```
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```
To deactivate:
```$xslt
deactivate
```


### Model performance
|  Model | CV  | CV with best params  |
|---|---|---|
| Decision tree |  82 |   |
| Extra trees  |  82 |  84.6 |
| K-nearest neighbors | 73  |   |
| Linear discriminant analysis  | 82  | 82  |
| Linear regression  |  83 |   |
| Logistic regression  | 81  |   |
| Multi-layer perceptron  | 82.6  | 83.4  |
| Naive bayes  |   |   |
| Neural network Keras  |   |   |
| Neural network Keras with age prediction  |   |   |
| Quadratic discriminant analysis  |   |   |
| Random forest  |   |   |
| Support vector machines  |   |   |
| Xgboost classifier  |   |   |