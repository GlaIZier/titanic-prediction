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

### The best models with metrics are located in notebooks
#### Best test set score = 0.80382, top 13%
|  Model | The best achieved test set accuracy  |
|---|---|
| Extra trees | 80.382% | 
| Random forest | 79.904% |
| Multi-layer perceptron | 79.904% |
| Bag of models (Extra trees + Multi-layer perceptron) | 80.382% |

### Run notebooks
```jupyter notebook --port 8890```

### The performance of models in legacy python folder
|  Model | CV  | CV with best params  |
|---|---|---|
| Decision tree |  82 |   |
| Extra trees  |  82 |  84.6 |
| K-nearest neighbors | 73  |   |
| Linear discriminant analysis  | 82  | 82  |
| Linear regression  |  83 |   |
| Logistic regression  | 81  |   |
| Multi-layer perceptron  | 82.6  | 83.4  |
| Naive bayes  |  80 |   |
| Neural network Keras  | 80  |   |
| Neural network Keras with age prediction  | 80  |   |
| Quadratic discriminant analysis  |  47 | 47  |
| Random forest  | 83  | 84  |
| Support vector machines  | 81  | 83  |
| Xgboost classifier  |  81 | 83.7  |
