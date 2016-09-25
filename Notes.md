# Basic Neural Network Tutorial Notes

## Theory

### Introduction to Neural Networks

There are many different types of neural networks and techniques for training them but I'm just going to focus on the most basic one of them all -- the classic back propagation neural network (BPN). The back propagation refers to the fact that any mistakes made by the network training get sent backwards through it in an attempt to correct it and so teach the network what's right and wrong.

This BPN uses the gradient descent learning method. Trying to describe this simply at this point is going to be difficult so I'll leave it for a bit later, all you need to know is that it's so called because it follows the steepest gradient down a surface which represents the error funciton as it tries to find the minimum of the error function and by doing so decrease the error.

I wanted to skip over the basics of neural networks, and all that this is your brain and this is a neuron but I guess it's unavoidable. I'm not going to go into great detail as there is plenty of information already available online. Here are the wiki entries on [Feed-forward_neural networks (FFNNs)](http://en.wikipedia.org/wiki/Feedforward_neural_network) and [back propagation](http://en.wikipedia.org/wiki/Back-propagation).

This is the second version of the tutorial since half way through the first one I realized that I needed to actually go over some of the theory properly before I could go over the implementation and so I've decided to split the tutorial into two parts: part 1(this) will go over the basic theory needed and part 2 will discuss some more advanced topics and the implementation.

Now I haven't even told you what a Neural Network is and what it is used for. Silly me! NNs have a variety of uses especially in classification or function-fitting problems, they can also be used to create emerging behaviour in agents reacting to environment sensors. They are one of the most important artificial intelligence tools available today. Just for the record, I am by no means an expert on neural networks, I just have a bit of experience implementing and successfully using BPN's in various image classification problems before.

### The Neuron

Okay; enough blabbering from me; let's get into the thick of it. THe basic building block of a NN is the neuron. The basic neuron is consisted of a black bo with weighted inputs and an output.

**Note:** *perceptron - neuron that classifies its inputs into one of two categories, basically the output of a neuron is clamped to 1 or 0.*

![Perceptron](https://takinginitiative.files.wordpress.com/2008/03/perceptron.jpg?w=680)

The black box section of the neuron consists of an activation function F(X), in our case its F(wSum - T) where the wSum is the weighted sum of the inputs and T is a threshold or bias value. We'll come back to the threshould value just now. The weights are initialized to some small random values and during training get updated. The weighted sum (wSum) is given below.

```cpp
for(i = 0; i < n; i++) {
	wSum += weight[i] * input[i];
}
```

Simple, huh? Now for the function F, there are various functions that can be used for F; the most common ones include the step function and the sigmoid function.  We will be using the sigmoid function in our BPN as its again the classical activation function. The sigmoid function and its derivative are defined as:

```cpp
float sigmoid(float x) {
	return 1 / (1 + pow(e, -1 * x));
}
```

```cpp
float sigmoidDerivative(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}
```

The sigmoid function has the following graph:

![sigmoid activation function](https://takinginitiative.files.wordpress.com/2008/04/sigmoidfunction.png?w=680)

**Note:** Now it's very important to realize that the sigmoid function can never return a 0 or a 1 due to its asymptopic nature. So often it's a good idea to treat values over 0.9 as 1 and under 0.1 as 0.

Now we need to cover an important point regarding the input data and the desired output. Let's use the binary OR as an example to explain the function of the weights and threshold. With OR we want a binary output telling us whether it's true or not, so a single perceptron with two inputs is created. Now the search space for the neural network can be drawn as follows:

![The OR operator modeled by a single perceptron](https://takinginitiative.files.wordpress.com/2008/03/lsds.jpg?w=680)

The dark blue dots represent values of true and the light blue dot represents a value of false; you can clearly see how the two classes are separable.  We can draw a line separating them as in the above example. This separating line is called a hyperplane. A single neuron can create a single hyperplane and teh above function can be solved by a single neuron.

Another important point is that the hyperplane above is a straight line, this means we used a linear activation function (i.e. a step function) for our neuron. If we used a sigmoid function or similar the hyperplane would resemble a sigmoid shape as seen below. They hyperplane generated by the image depends on the activation function used.

![sigmoid hyperplane](https://takinginitiative.files.wordpress.com/2008/04/sigmoidhp.jpg?w=680)

Remember that Threshold (Bias) value we had earlier? What does that do? Simply put, it shifts the hyperplane left and right while the weights orientate the hyperplane. In graphical terms, the Threshold translates the hyperplane while the weights rotate it. This threshold also needs to be updated during hte learning process.

I'm not going to go into the details of how a neuron learns in detail or provide examples; there are many excellent books and online guides to do this. The basic procedure is as follows:
* run an input pattern through the function
* calculate the error (desired value - actual value)
* update the weights according to learning rate and error
* move onto next pattern

The learning rate term is a term that hasn't been mentioned before and is very important, it greatly affects the performance and accuracy of your network. I'll go over this in more detail once we get to the weight updates.

### The Multilayer Neural Network

As I mentioned before for linearly separable problems a single neuron is sufficient but what about problems that have more than one class or ones where data isn't so well separated like in the example below:

![non-linearly separable dataset](https://takinginitiative.files.wordpress.com/2008/03/nlsds.jpg?w=680)

Here we need at least two hyper-planes to solve this problem so we need 2 neurons. This requires us to link up these neurons together, to link them up we'll need shared inputs and outputs -- in other words: a multilayer neural network.  The standard architecture of a NN consists of 3 layers: an input layer, a hidden layer and an output layer.  THere are several proffs available that you will almost never need more than 3 layers (I'll try to get links to the papers soon) and allso more importantly we want to keep things simple.

**NOTE**: you almost never know what your search space looks like, that's why you're using a neural network, often you'll have to experiment with the neural network architecture in regards to how many hidden neurals you need to get a good result.

### A Basic Multilayer Neural Network:

![Standard Architecture for a Back Propagation Neural Network](https://takinginitiative.files.wordpress.com/2008/03/bpn.jpg?w=490&h=277)

Above is a basic multilayer neural network; the inputs are shared and so are the outputs; note that each of these links have separate weights.  Now what are those square blocks in the neural network? They are our thresholds (bias) values;  instead of having to store and update separate thresholds for each neuron (remember each neuron's activation function took a weighted sum minus a threshold as input), we simply create 2 extra neurons with a constant value of -1.  These neurons are then hooked up to the rest of the network and have their own weights  (these are technically the threshold values).

This results in the weighted sum + the weight of the threshold multiplied by -1. Obviously, you can see itt's the same as we had earlier. Now when we update the weights for the network during back propagation we automatically update the thresholds as well, saving us a few calculations and headaches.

Okay, so far, everything has (hopefully) been pretty simple, especially if you have a bit of a background in NNs or have read through an introductory chapter in an AI textbook. There are only 3 things left to discuss -- calculating the errors at the output, updating the weights (the back propagation) and the stopping conditions.

The only control over this architecture you have is over the number of hidden neurons since your inputs and desired outputs are already known, so deciding on how many hidden neurons you need is often a tricky matter. Too many is never good, and neither is too little. Some careful experimentation will often be required to find out an optimal amount of hidden neurons.

I'm not going to go over feeding the input forward, as it is really simple: all you do is calculate the output (the value of the activation function for the weighted sum of inputs) at a neuron and use it as the input for the next layer.

### The Neuron Error Gradients

Okay, so obviously, we need to update the weights in our neural network to give the correct output at the output layer. This forms the basis of training the neural network. We will make use of back-propagation for these weight updates. This just means input is fed in, the errors are calculated and filtered back through the network making changes to the weights to try to reduce the error.

The weight changes are calculated using the gradient descent method. This means we follow the steepest path on the error function to try and minimize it. I'm not going to go into the math behind gradient descent, the error function, and so on, since it's not really needed. Simply put, al we're doing is just taking the error at the output neurons (Desired value - actual value) and multiplying it by the gradient of the sigmoid function. If the difference is positive, we need to move up the gradient of the activation funciton and if it's negative, we need to move down the gradient of the activation function.

![Error: gradient - explanation](https://takinginitiative.files.wordpress.com/2008/04/errorgradientsexplanation.png?w=680)

This is the formula to calculate the basic error gradient for each output neuron k:

```cpp
float errorGradientForOutput(&OutputNode node) {
	return node->value * (1 - node->value) * (node->desiredValue - node->value);
}
```

There is a difference between the error gradients at the output and hidden layers. The hidden layer's error gradient is based on the output layer's error gradient (back propagation) so for the hidden layer, the error gradient for each hidden neuron is the gradient of the activation function multiplied by the weighted sum of the errors at the output layer, originating from that neuron (wow, gettign a bit crazy here, eh?):

```cpp
float errorGradientForHidden(&HiddenNode node) {
	float weightedSumOfErrors = 0f;
	int i;

	for(i = 0; i < node->outputsCount; i++) {
		weightedSumOfErrors += node->outputWeights[i] * errorGradientForOutput(node->outputs[i]);
	}

	return node->value * (1 - node->value) * weightedSumOfErrors;
}
```

### The Weight Update

The final step in the algorithm is to update the weights. This occurs as follows:

```cpp
void updateWeights(float learningRate, HiddenNode *hiddenNodes, OutputNode *outputNodes, int inputCount, int hiddenCount, int outputCount) {
	updateWeightsForHidden(learningRate, hiddenNodes, inputCount, hiddenCount);
	updateWeightsForOutput(learningRate, outputNodes, hiddenCount, outputCount);
}

void updateWeightsForHidden(float learningRate, HiddenNode *hiddenNodes, int inputCount, int hiddenCount) {
	float weightChange;
	int i, j;

	for(i = 0; i < inputCount; i++) {
		for(j = 0; j < hiddenCount; j++) {
			weightChange = learningRate * errorGradientForHidden(hiddenNodes[j]);
			hiddenNodes[j]->inputWeights[i] = hiddenNodes[j]->inputWeights[i] + weightChange;
		}
	}
}

void updateWeightsForOutput(float learningRate, OutputNode *outputNodes, int hiddenCount, int outputCount) {
	float weightChange;
	int i, j;

	for(i = 0; i < hiddenCount; i++) {
		for(j = 0; j < outputCount; j++) {
			weightChange = learningRate * errorGradientForOutput(outputNodes[j]);
			outputNodes[j]->hiddenWeights[i] = outputNodes[j]->hiddenWeights[i] + weightChange;
		}
	}
}
```

The alpha value you see above is the learning rate, this is usually a value between 0 and 1. It affects how large the weight adjustments are and so also affects the learning speed of the network. This value needs to be carfully selected to provide the best results. Too low, and it will take ages to learn; too high, and the adjustments might be too large and the accuracy will suffer as the network will constantly jump over a better solution and generally get stuck at some sub-optimal accuracy.

### The Learning algorithm

The BPN learns during a training epoch. You will probably go through several epochs before the network has sufficiently learned to handle all the data you've provided it and the end result is satisfactory. A training epoch is described below:

*For each input entry in the training data set:*
* feed input data in (feed forward)
* check output against desired value and feed back error (back-propagate)

*Where back-propagation consists of:*
* calculate error gradients
* update weights

### Stopping Conditions

These are some commonly used stopping conditions used for neural networks: desired accuracy, desired mean square error and elapsed epochs. I won't go over these in too much detail now, as I will be covering them in the next tutorial with some training examples. The main reasobnb I'm not going into detail here is that I haven't described the training of the network in detail. I need to go ovewr the creating of training data sets, what generalization and validation errors are and so on. All this will be covered in greater detail in the next tutorial.

## Implementation