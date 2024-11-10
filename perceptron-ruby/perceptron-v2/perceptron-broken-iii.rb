class MLP
  attr_accessor :weights_input_hidden, :weights_hidden_output, :learning_rate

  def initialize(input_size, hidden_size, output_size, learning_rate = 0.1)
    # Initialize weights (random values between -0.5 and 0.5)
    @weights_input_hidden = Array.new(input_size) { Array.new(hidden_size) { rand(-0.5..0.5) } }
    @weights_hidden_output = Array.new(hidden_size) { Array.new(output_size) { rand(-0.5..0.5) } }
    @learning_rate = learning_rate
  end

  # Sigmoid activation function
  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  # Derivative of the sigmoid function
  def sigmoid_derivative(x)
    sigmoid(x) * (1 - sigmoid(x))
  end

  # Forward pass: input -> hidden layer -> output layer
  def forward_pass(inputs)
    # Hidden layer activation
    @hidden_layer_inputs = Array.new(@weights_input_hidden[0].size) { 0 }
    @hidden_layer_outputs = Array.new(@weights_input_hidden[0].size) { 0 }

    @weights_input_hidden.each_with_index do |weights, i|
      @hidden_layer_inputs[i] = weights.zip(inputs).map { |w, x| w * x }.sum
      @hidden_layer_outputs[i] = sigmoid(@hidden_layer_inputs[i])
    end

    # Output layer activation
    @output_layer_inputs = Array.new(@weights_hidden_output[0].size) { 0 }
    @output_layer_outputs = Array.new(@weights_hidden_output[0].size) { 0 }

    @weights_hidden_output.each_with_index do |weights, i|
      @output_layer_inputs[i] = weights.zip(@hidden_layer_outputs).map { |w, h| w * h }.sum
      @output_layer_outputs[i] = sigmoid(@output_layer_inputs[i])
    end

    # Debugging: print the outputs of the layers
    puts "Hidden Layer Outputs: #{@hidden_layer_outputs}"
    puts "Output Layer Outputs: #{@output_layer_outputs}"

    @output_layer_outputs
  end

  # Backpropagation: Update weights based on error
  def backpropagate(inputs, target_outputs)
    # Check for nil in the output and target
    raise "Output layer is nil" if @output_layer_outputs.nil?
    raise "Target outputs are nil" if target_outputs.nil?

    # Calculate output layer error
    output_errors = @output_layer_outputs.zip(target_outputs).map { |o, t| o - t }

    # Debugging: print output errors
    puts "Output Errors: #{output_errors}"

    output_deltas = output_errors.zip(@output_layer_inputs).map { |err, input| err * sigmoid_derivative(input) }

    # Calculate hidden layer error
    hidden_errors = @weights_hidden_output.transpose.map { |weights| weights.zip(output_deltas).map { |w, delta| w * delta }.sum }
    hidden_deltas = hidden_errors.zip(@hidden_layer_inputs).map { |err, input| err * sigmoid_derivative(input) }

    # Update weights between hidden and output layer
    @weights_hidden_output.each_with_index do |weights, i|
      weights.each_with_index do |weight, j|
        @weights_hidden_output[i][j] -= @learning_rate * output_deltas[i] * @hidden_layer_outputs[j]
      end
    end

    # Update weights between input and hidden layer
    @weights_input_hidden.each_with_index do |weights, i|
      weights.each_with_index do |weight, j|
        @weights_input_hidden[i][j] -= @learning_rate * hidden_deltas[j] * inputs[i]
      end
    end
  end

  # Train the network for a specified number of epochs
  def train(training_data, epochs)
    epochs.times do |epoch|
      total_error = 0
      training_data.each do |inputs, target_outputs|
        outputs = forward_pass(inputs)
        backpropagate(inputs, target_outputs)

        # Calculate the error (sum of squared errors)
        total_error += outputs.zip(target_outputs).map { |o, t| (o - t) ** 2 }.sum
      end
      puts "Epoch #{epoch + 1}, Error: #{total_error}"
    end
  end
end

# Training data for XOR
# Input: [x1, x2], Output: target
training_data = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]]
]

# Initialize MLP with 2 input nodes, 2 hidden nodes, and 1 output node
mlp = MLP.new(2, 2, 1)

# Train the perceptron for 10000 epochs
mlp.train(training_data, 10000)

# Test the perceptron after training
puts "Predictions:"
puts "0 XOR 0 = #{mlp.forward_pass([0, 0]).map { |o| (o > 0.5 ? 1 : 0) }}"
puts "0 XOR 1 = #{mlp.forward_pass([0, 1]).map { |o| (o > 0.5 ? 1 : 0) }}"
puts "1 XOR 0 = #{mlp.forward_pass([1, 0]).map { |o| (o > 0.5 ? 1 : 0) }}"
puts "1 XOR 1 = #{mlp.forward_pass([1, 1]).map { |o| (o > 0.5 ? 1 : 0) }}"
