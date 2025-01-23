

class ADALINE
  attr_accessor :weights, :bias, :learning_rate

  def initialize(input_size, learning_rate = 0.01)
    # Initialize weights and bias
    @weights = Array.new(input_size, 0.0)  # Initial weights (random or zero)
    @bias = 0.0  # Initial bias
    @learning_rate = learning_rate  # Learning rate for gradient descent
  end

  # Sigmoid activation (for converting output to probability)
  def predict(inputs)
    sum = inputs.zip(@weights).map { |x, w| x * w }.sum + @bias
    sum  # No activation function (for ADALINE, it's linear)
  end

  # Train the model using gradient descent (MSE loss)
  def train(inputs, targets, epochs)
    losses = []

    epochs.times do |epoch|
      total_loss = 0.0

      inputs.each_with_index do |input, i|
        target = targets[i]

        # Calculate prediction (linear output)
        prediction = predict(input)

        # Calculate the error (target - prediction)
        error = target - prediction

        # Update weights and bias using gradient descent
        @weights = @weights.zip(input).map { |w, x| w + @learning_rate * error * x }
        @bias += @learning_rate * error

        # Calculate loss (Mean Squared Error)
        total_loss += error ** 2
      end

      # Record the loss for this epoch
      avg_loss = total_loss / inputs.length
      losses << avg_loss

      # Print progress
      if (epoch + 1) % 100 == 0
        puts "Epoch #{epoch + 1}/#{epochs}, Loss: #{avg_loss}"
      end
    end

    # Return loss for visualization
    losses
  end
end

# Create a simple AND-like dataset (linearly separable data)
inputs = [
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0]
]
targets = [0.0, 0.0, 0.0, 1.0]  # AND output

# Create and train the ADALINE model
adaline = ADALINE.new(2, 0.1)  # 2 inputs (features), learning rate = 0.1
losses = adaline.train(inputs, targets, 1000)  # Train for 1000 epochs


=begin
# Plot the training loss (requires 'gruff' gem)
begin
  require 'gruff'
  g = Gruff::Line.new
  g.title = 'ADALINE Training Loss'
  g.data(:loss, losses)
  g.write('adaline_loss.png')
rescue LoadError
  puts "Install the 'gruff' gem for plotting: gem install gruff"
end
=end


# Test the model after training
puts "Final Weights: #{adaline.weights}"
puts "Final Bias: #{adaline.bias}"

# Test predictions on the same input data
inputs.each_with_index do |input, i|
  prediction = adaline.predict(input)
  puts "Input: #{input}, Prediction: #{prediction}, Target: #{targets[i]}"
end


puts "bye"