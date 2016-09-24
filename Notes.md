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

![Perceptron](http://g.gravizo.com/g?
	digraph G {
		w1 [shape=box];
		w2 [shape=box];
		w3 [shape=box];
		w1 -> "F(wSum - T)" [label="input 1"]
		w2 -> "F(wSum - T)" [label="input 2"]
		w3 -> "F(wSum - T)" [label="input 3"]
		"F(wSum - T)" -> "" [style=filled,color="1.0 1.0 1.0",label="result"]
	}
)

#### Basic neuron as a black box

The black box section of the neuron consists of an activation function F(X), in our case its F(wSum - T) where the wSum is the weighted sum of the inputs and T is a threshold or bias value. We'll come back to the threshould value just now. The weights are initialized to some small random values and during training get updated. The weighted sum (wSum) is given below.

```
for(i=1; i<n; i++) {
	wSum += weight[i] * input[i];
}
```

Simple, huh? Now for the function F, there are various functions that can be used for F; the most common ones include the step function and the sigmoid function.  We will be using the sigmoid function in our BPN as its again the classical activation function. The sigmoid function and its derivative are defined as:

```
float sigmoid(float x) {
	return 1 / (1 + pow(e, -1 * x));
}
```

```
float sigmoidDerivative(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}
```

