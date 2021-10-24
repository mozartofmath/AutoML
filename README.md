# Auto Deep Learning

## Installation Instructions

Install Anaconda 3 and then open a terminal from this directory and execute the following command. This will install numpy and tensorflow.
```
pip install -r requirements.txt
```

## Configurations
The program operates in two modes. ```"train"``` mode and ```"predict"``` mode. You can change the modes by editing the ```config.json``` file. 

```JSON
{
    "mode" : "train",
    "train" : {
        "input_path" : "TrainingFiles/input_7.txt",
        "output_path" : "TrainingFiles/output_7.txt",
        "model_path" : "Models/model_7.h5",
        "epochs" : 40,
        "max_hill_climbing_iterations" : 5
    },
    "predict" : {
        "model_path" : "Models/model_7.h5",
        "input_path":"PredictionFiles/input_7.txt",
        "output_path" : "PredictionOutputs/predicted_7.txt"
    }
}
```
If the mode is set to ```train```
<ul>
<li>"input_path" - path to the input file</li>
<li>"output_path" - path to the output file that contains the target labels</li>
<li>"model_path" - where to save the trained model. must always be a **.h5** file</li>
<li>"epochs" - number of epochs to train the model</li>
<li>"max_hill_climbing_iterations" - maximum number of iterations of hill climbing</li>
</ul>

If the mode is set to ```predict```
<ul>
<li>"input_path" - path to the input file</li>
<li>"output_path" - path to store the predictions</li>
<li>"model_path" - where to load the saved model. must always be a **.h5** file</li>
</ul>

## Execution
After editing the ```config.json``` file, execute the following command to run the program to train a model or perform predictions.
```
python auto_deep_learning.py
```
