require 'matrix'

# Define the XOR dataset
xor_data = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
]

# Define the sigmoid activation function
def sigmoid(x)
  1.0 / (1.0 + Math.exp(-x))
end

# Define the derivative of the sigmoid function
def sigmoid_derivative(x)
  sigmoid(x) * (1 - sigmoid(x))
end

# Define the Multi-Layer Perceptron class
class MLP
  attr_reader :weights1, :weights2, :bias1, :bias2

  def initialize(input_size, hidden_size, output_size)
    # Initialize weights and biases randomly
    @weights1 = Matrix.build(input_size, hidden_size) { rand(-1.0..1.0) }
    @weights2 = Matrix.build(hidden_size, output_size) { rand(-1.0..1.0) }
    @bias1 = Matrix.build(1, hidden_size) { rand(-1.0..1.0) }
    @bias2 = Matrix.build(1, output_size) { rand(-1.0..1.0) }
  end

  # Forward pass
  def forward(inputs)
    # Input layer
    input_matrix = Matrix.column_vector(inputs)

    # Hidden layer
    hidden_input = (input_matrix * @weights1) + @bias1
    hidden_output = hidden_input.map(&:sigmoid)

    # Output layer
    output_input = (hidden_output * @weights2) + @bias2
    output_output = output_input.map(&:sigmoid)

    output_output[0, 0]
  end

  # Backpropagation algorithm
  def train(inputs, target, learning_rate)
    # Forward pass
    output = forward(inputs)

    # Calculate error
    error = target - output

    # Backpropagate error
    # Output layer
    output_delta = error * sigmoid_derivative(output)
    @weights2 += (learning_rate * hidden_output.transpose * output_delta)
    @bias2 += (learning_rate * output_delta)

    # Hidden layer
    hidden_delta = (output_delta * @weights2.transpose) * sigmoid_derivative(hidden_output)
    @weights1 += (learning_rate * Matrix.column_vector(inputs).transpose * hidden_delta)
    @bias1 += (learning_rate * hidden_delta)
  end

  # Calculate the mean squared error
  def mse(data)
    sum = 0.0
    data.each do |inputs, target|
      output = forward(inputs)
      sum += (target - output)**2
    end
    sum / data.size
  end
end

# Create the MLP
mlp = MLP.new(2, 2, 1)

# Train the MLP
learning_rate = 0.1
epochs = 10000

epochs.times do |epoch|
  xor_data.each do |inputs, target|
    mlp.train(inputs, target, learning_rate)
  end

  # Print the mean squared error every 1000 epochs
  if (epoch + 1) % 1000 == 0
    puts "Epoch #{epoch + 1}: MSE = #{mlp.mse(xor_data)}"
  end
end

# Test the trained MLP
puts "XOR results:"
xor_data.each do |inputs, target|
  output = mlp.forward(inputs)
  puts "Inputs: #{inputs}, Target: #{target}, Output: #{output}"
end



__END__

note - reports error:

lib/matrix.rb:1070:in `*': Dimension mismatch (ExceptionForMatrix::ErrDimensionMismatch)
        from perceptron-v2.rb:39:in `forward'
