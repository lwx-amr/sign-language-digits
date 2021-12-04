from data_init import normalize_for_nn
from tensorflow import keras
import random
from sklearn.metrics import classification_report

accuracy_list = list()
normalized_dataSet = normalize_for_nn()

def nn_parser():
    random.shuffle(normalized_dataSet)    
    train_data = normalized_dataSet[:int(len(normalized_dataSet)*0.6)]
    test_data = normalized_dataSet[int(len(train_data)):int(len(normalized_dataSet)*0.8)]
    validate_data = normalized_dataSet[int((len(test_data)+len(train_data))):]    
    xTrain = list()
    yTrain = list()
    xTest = list()
    yTest = list()
    xVali = list()
    yVali = list()    
    for i in range(len(train_data)):
        xTrain.append(train_data[i][0])
        yTrain.append(train_data[i][1])
    
    for i in range(len(test_data)):
        xTest.append(test_data[i][0])
        yTest.append(test_data[i][1])
    
    for i in range(len(validate_data)):
        xVali.append(validate_data[i][0])
        yVali.append(validate_data[i][1])
    
    return xTrain, yTrain, xTest, yTest, xVali, yVali;

def model_arch_1():
    model = keras.Sequential()

    # Input Layers
    model.add(keras.layers.Dense(500, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(260, activation='relu'))
    model.add(keras.layers.Dense(400, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    
    return model

def model_arch_2():
    model = keras.Sequential()

    # Input Layers
    model.add(keras.layers.Dense(350, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(80, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(60, activation='relu'))
    model.add(keras.layers.Dense(400, activation='relu'))

    return model

def model_arch_3():
    model = keras.Sequential()

    # Input Layers
    model.add(keras.layers.Dense(500, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(80, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(480, activation='relu'))

    return model

def model_arch_4():
    model = keras.Sequential()

    # Input Layers
    model.add(keras.layers.Dense(500, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(400, activation='relu'))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(keras.layers.Dense(240, activation='relu'))
    model.add(keras.layers.Dense(350, activation='relu'))

    return model

def run_archituctures():
    
    models = []
    models.append(model_arch_1())
    models.append(model_arch_2())
    models.append(model_arch_3())
    models.append(model_arch_4())
    
    for j in range(len(models)):
        local_acc = list()
        for i in range(5):
            # Shuffle Data  
            xTrain, yTrain, xTest, yTest, xVali, yVali = nn_parser()
            # Get Architucture
            model = models[j]
            # Output Layers
            model.add(keras.layers.Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(xTrain, yTrain, epochs = 12, validation_data=(xVali, yVali), batch_size=32)
            #Test Part
            loss, acc = model.evaluate(xTest, yTest)
            local_acc.append(acc)
            Y_pred = model.predict(xTest)
            y_labels_predicted = []
            y_labels_actual = []
            for i in range(len(Y_pred)):
                y_labels_predicted.append(np.argmax(Y_pred[i]))
                y_labels_actual.append(np.argmax(yTest[i]))
            print(classification_report(y_labels_actual , y_labels_predicted))
        accuracy_list.append(local_acc)
        print(local_acc)        
    print(accuracy_list)
    for i in range(len(models)):
        with open('reports/nn.txt','a') as out:
        # Pass the file handle in as a lambda function to make it callable
            models[i].summary(print_fn=lambda x: out.write(x + '\n'))
            out.write("Train Accuracy : ")
            out.write( str(accuracy_list[i])+ "%")
            print(models[i].summary())

run_archituctures()