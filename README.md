[![Language](https://img.shields.io/badge/Swift-4.0-orange.svg?style=flat)](https://swift.org)
[![Licence](https://img.shields.io/dub/l/vibe-d.svg?maxAge=2592000)](https://opensource.org/licenses/MIT)

# Swift_NN

Swift_NN is a proof of concept app that utilizes a convolutional neural network programmed completely in Swift with no libraries to recignize single digit numbers within images. All of the calculations, including matrix multiplication, and splicing of multidimensional data is done using basic swift code. 

![image1](https://github.com/crweaver225/Swift_NN/blob/master/screenshots/ss1.png?raw=true)
![image3](https://github.com/crweaver225/Swift_NN/blob/master/screenshots/ss3.png?raw=true)

This project was done in conjunction with another project (https://github.com/crweaver225/CAINN) which uses a program coded in Python to train the neural network with no libraries other than Numpy. After training, the neural net is outputted in JSON and moved to this xcode project where Swift_NN parses the JSON and constructs a completley Swifty neural network. 

Theoretically, my Swift_Neural_Network class could work like a black box, parsing the provided neural network json and applying the values to an instance of the class like so:
```
 let swiftNN = Swift_Neural_Network()
 if let activation_functions = jsonResult["activity_functions"] as? [String] {
    swiftNN.activation_functions = swiftNN.convertToActivationFunctions(activationFunctions: activation_functions)
 }
 if let weights = jsonResult["weights"] as? [Matrix] {
    swiftNN.weights = weights
 }
 if let type = jsonResult["type"] as? [String] {
    swiftNN.layers = swiftNN.convertToLayerTypes(layers: type)
 }
 ```
 Then running the network would be as simple as:
 ```
 let results = self.swiftNN.executeNeuralNetwork(input: input_data] 
 ```
 
 Swift_NN Demo
 https://youtu.be/kopetp4nIQ8
