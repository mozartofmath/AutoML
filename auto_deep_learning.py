import json
import numpy as np
import tensorflow as tf
import random

def output_shape(line):
    counts = []
    current_count = 0
    state = 0
    for c in line:
        if state == 0:
            if c == "{":
                state = 1
        else:
            if c == ',':
                current_count+=1
            elif c == "}":
                state = 0
                counts.append(current_count+1)
                current_count = 0
    return counts

def parse_inputs(input_path, output_path):
    input_, output_ = open(input_path, 'r'), open(output_path, 'r')
    X, y = [],[]
    for line in input_.readlines():
        l = [float(c) for c in line.strip().replace('{','').replace('}','').split(',')]
        X.append(l)
    #print(line)
    for line in output_.readlines():
        l = [float(c) for c in line.strip().replace('{','').replace('}','').split(',')]
        y.append(l)
    #print(line)
    print("<<<<< Parsed Input File >>>>>")
    return X, y, output_shape(line)

def format_output(vals, out_shape):
    strings = []
    for i in range(len(out_shape)):
        strings.append('{' + ",".join(vals[sum(out_shape[:i]):sum(out_shape[:i+1])]) + '}')
    return ','.join(strings)

def dense_block(block_input, n_layers, n_units, activation = 'relu'):
    block_output = tf.keras.layers.Dense(n_units, activation=activation)(block_input)
    for i in range(n_layers-1):
        block_output = tf.keras.layers.Dense(n_units, activation=activation)(block_output)
    return block_output

def conv_block(block_input, n_layers, n_filters, filter_size, stride, activation = 'relu'):
    block_output = tf.keras.layers.Conv2D(n_filters, 
                                          filter_size, 
                                          strides=(stride, stride), 
                                          activation=activation)(block_input)
    for i in range(n_layers-1):
        block_output = tf.keras.layers.Conv2D(n_filters, 
                                              filter_size, 
                                              strides=(stride, stride), 
                                              activation=activation)(block_output)
    block_output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(block_output)
    return block_output

def generate_model(X, y, problem_type):
    loss, optimizer, metrics = 'mse', 'adam', 'accuracy'
    inputs = tf.keras.layers.Input(shape=X.shape[1:])
    
    dense_out = dense_block(inputs, n_layers=1, n_units=X.shape[-1]*2, activation = 'relu')
    dense_out = dense_block(inputs, n_layers=2, n_units=X.shape[-1]*4, activation = 'relu')
    dense_out = dense_block(inputs, n_layers=1, n_units=X.shape[-1], activation = 'relu')
    
    if problem_type[0] == 'classification':
        outputs = tf.keras.layers.Dense(y.shape[-1], activation='sigmoid')(dense_out)
        loss = 'binary_crossentropy'
        metrics = 'mse'
    
    elif problem_type[0] == 'regression':
        outputs = tf.keras.layers.Dense(y.shape[-1])(dense_out)
        loss = 'mse'
        metrics = 'mse'
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    print('<<<<< Generated Suitable Model Architecture >>>>>')
    return model

def train_model(model, X, y, epochs):
    train_mask = [random.randint(0,9)<1 for i in range(len(X))]
    test_mask = [not b for b in train_mask]
    X_train, y_train = X[train_mask,:], y[train_mask,:]
    X_test, y_test = X[test_mask,:], y[test_mask,:]
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=10)
    print("<<<<< Model Trained >>>>>")
    return history

def automl():
    file = open('config.json', 'r')
    config = json.load(file)
    file.close()
    
    mode = config['mode']
    
    if mode == 'train':
        input_path = config['train']['input_path']
        output_path = config['train']['output_path']
        model_path = config['train']['model_path']
        
        X, y, out_shape = parse_inputs(input_path, output_path)
        X, y, out_shape = np.array(X), np.array(y), np.array(out_shape).astype(str)
        model = generate_model(X, y, ['regression', 0])
        history = train_model(model, X, y, epochs = 1)
        
        model.save(model_path)
        print(f"<<<<< Model Saved At: {model_path} >>>>>")
        
        model_metadata_path = model_path[:-3]+'_metadata.json'
        file = open(model_metadata_path, 'w')
        file.write('{"out_shape" : [' + ",".join(out_shape)+ ']}')
        file.close()
        
    if mode == 'predict':
        input_path = config['predict']['input_path']
        output_path = config['predict']['output_path']
        model_path = config['predict']['model_path']
        
        model_metadata_path = model_path[:-3]+'_metadata.json'
        
        X, _, _ = parse_inputs(input_path, input_path)
        X = np.array(X)
        
        model = tf.keras.models.load_model(model_path)
        print('<<<<< Loaded Model >>>>>')
        pred = (model.predict(X) > 0.5).astype(int).astype(str)
        
        metadata_file = open(model_metadata_path, 'r')
        metadata = json.load(metadata_file)
        metadata_file.close()
        out_shape = metadata['out_shape']
        
        out_file = open(output_path, 'w')
        for row in pred:
            out_file.write(format_output(row, out_shape) +'\n')
        out_file.close()
        print('<<<<< Completed Prediction >>>>>')
        print(f"<<<<< Prediction Saved At: {output_path} >>>>>")
if __name__ == '__main__':
    automl()