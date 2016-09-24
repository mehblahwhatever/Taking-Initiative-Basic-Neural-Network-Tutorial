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
for(i=1; i < n; i++) {
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

