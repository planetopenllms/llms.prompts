# Q: generate a perceptron in ruby code

A: Certainly! Below is a simple implementation of a **perceptron** in Ruby for binary classification. The perceptron learns from input data and makes predictions based on the learned weights.

### **Perceptron in Ruby**

```ruby
class Perceptron
  attr_accessor :weights, :learning_rate

  def initialize(input_size, learning_rate = 0.1)
    # Initialize weights with small random values, including a bias term
    @weights = Array.new(input_size + 1) { rand(-0.5..0.5) }  # +1 for the bias
    @learning_rate = learning_rate
  end

  # Activation function (step function)
  def activation(x)
    x >= 0 ? 1 : 0
  end

  # Forward pass: compute weighted sum + bias
  def predict(inputs)
    sum = @weights[0]  # Start with bias weight
    inputs.each_with_index do |input, index|
      sum += input * @weights[index + 1]
    end
    activation(sum)
  end

  # Train the perceptron using the Perceptron learning rule
  def train(training_data, epochs = 10)
    epochs.times do
      training_data.each do |inputs, target|
        prediction = predict(inputs)
        error = target - prediction

        # Update weights: w = w + learning_rate * error * input
        @weights[0] += @learning_rate * error  # Bias weight update
        inputs.each_with_index do |input, index|
          @weights[index + 1] += @learning_rate * error * input
        end
      end
    end
  end
end

# Sample dataset for logical OR
# Input: [x1, x2], Output: target
training_data = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 1]
]

# Create and train the perceptron
perceptron = Perceptron.new(2)  # 2 input neurons (x1 and x2)
perceptron.train(training_data, 100)

# Test the perceptron after training
puts "Predictions:"
puts "0 OR 0 = #{perceptron.predict([0, 0])}"
puts "0 OR 1 = #{perceptron.predict([0, 1])}"
puts "1 OR 0 = #{perceptron.predict([1, 0])}"
puts "1 OR 1 = #{perceptron.predict([1, 1])}"
```

### **Explanation of Code:**

1. **Perceptron Class**:
   - This class represents a basic perceptron model. It has two main components: `weights` (including a bias) and `learning_rate`.
   - The perceptron is initialized with a set of random weights and a bias term.

2. **Activation Function**:
   - A simple **step function** is used as the activation function. It outputs `1` if the input is greater than or equal to 0 and `0` otherwise.

3. **Prediction**:
   - The `predict` method computes the weighted sum of the inputs plus the bias, applies the activation function, and returns the predicted class (0 or 1).

4. **Training**:
   - The `train` method updates the weights based on the Perceptron learning rule:
     \[
     \text{weight} = \text{weight} + \eta \times \text{error} \times \text{input}
     \]
     where `η` is the learning rate and `error = target - prediction`.

5. **Training Data**:
   - The `training_data` variable represents the logical OR function. Each input (`[x1, x2]`) has a corresponding target output (`0` or `1`).

6. **Testing**:
   - After training the perceptron for 100 epochs, the code tests the model with the possible inputs for the OR function.

### **Sample Output:**

```
Predictions:
0 OR 0 = 0
0 OR 1 = 1
1 OR 0 = 1
1 OR 1 = 1
```

This Ruby implementation of a perceptron can be adapted for other binary classification tasks by changing the `training_data` and adjusting the number of inputs.