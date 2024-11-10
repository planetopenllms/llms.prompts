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
