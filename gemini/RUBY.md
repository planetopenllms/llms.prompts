# More Google Gemini Sample Prompts (Ruby Edition)

Model - gemini-1.5-flash


## Q: generate a multi-layer perceptron in ruby code and train on logical xor

A:

```ruby
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
```

**Explanation:**

1. **Import Matrix:** This line imports the Matrix library for performing matrix operations.

2. **XOR Data:** The `xor_data` array stores the training data for the XOR problem. Each element is a pair of `[inputs, target]` where `inputs` are the two input values and `target` is the expected output.

3. **Sigmoid and Sigmoid Derivative:** These functions define the sigmoid activation function and its derivative.

4. **MLP Class:** The `MLP` class defines the multi-layer perceptron with the following attributes:
    - `weights1`: Weights between the input and hidden layer.
    - `weights2`: Weights between the hidden and output layer.
    - `bias1`: Biases for the hidden layer.
    - `bias2`: Biases for the output layer.

5. **Initialize:** The `initialize` method initializes the weights and biases with random values.

6. **Forward:** The `forward` method implements the forward pass of the network. It takes the input values, applies the weights and biases to each layer, and uses the sigmoid activation function to calculate the output.

7. **Train:** The `train` method implements the backpropagation algorithm to update the weights and biases based on the error. It calculates the error between the predicted and target values, and uses this error to adjust the weights and biases in the opposite direction of the error gradient.

8. **MSE:** The `mse` method calculates the mean squared error between the predicted and target values for the given dataset.

9. **Create and Train:** This code creates an instance of the `MLP` class with two hidden neurons and trains it for 10,000 epochs using the XOR dataset. The `learning_rate` controls the step size of the weight updates.

10. **Test:** This code tests the trained MLP by predicting the output for each input pair in the XOR dataset and printing the results.

**Running the Code:**

1. Save the code as a `.rb` file (e.g., `mlp_xor.rb`).
2. Run the code from the command line: `ruby mlp_xor.rb`

The output should show the mean squared error decreasing over the epochs and the trained MLP successfully predicting the XOR function.




## Q: generate (crypto) punks pixel art images in 24x24px and in .png. use a ruby script and the punks gem


A:

```ruby
require 'punks'

# Set the number of images to generate
num_images = 10

# Create a directory to store the images
Dir.mkdir("punks_images") unless Dir.exist?("punks_images")

# Generate the images
num_images.times do |i|
  # Generate a random punk
  punk = Punks::Punk.new(attributes: Punks::Attributes.random)

  # Create a pixel art image using the punk's attributes
  image = punk.to_pixel_art(size: 24, color: :greyscale)

  # Save the image as a PNG file
  image.save("punks_images/punk_#{i + 1}.png")
end

puts "Generated #{num_images} CryptoPunk images."
```

**Explanation:**

1. **Install the `punks` gem:**
   ```bash
   gem install punks
   ```

2. **Create a Ruby script:**
   Create a new Ruby file (e.g., `generate_punks.rb`) and paste the code above into it.

3. **Run the script:**
   Navigate to the directory where you saved the script and run it using:
   ```bash
   ruby generate_punks.rb
   ```

**Output:**

This script will generate `num_images` (default: 10) CryptoPunk pixel art images in the `punks_images` directory. Each image will be 24x24 pixels and saved as a PNG file. The images will be in greyscale.

**Customization:**

* **`num_images`:** Change the number of images to generate.
* **`color`:** Set the color palette for the images. The `punks` gem supports `:greyscale`, `:color`, and `:custom`. For `:custom`, you can provide a custom color palette.
* **`size`:** Change the image size.

**Example:**

```ruby
require 'punks'

# Generate 5 images with a custom color palette
num_images = 5
color_palette = [
  "#FF0000", # Red
  "#00FF00", # Green
  "#0000FF", # Blue
  "#FFFFFF", # White
  "#000000", # Black
]

# Create a directory to store the images
Dir.mkdir("punks_images") unless Dir.exist?("punks_images")

# Generate the images
num_images.times do |i|
  # Generate a random punk
  punk = Punks::Punk.new(attributes: Punks::Attributes.random)

  # Create a pixel art image using the punk's attributes
  image = punk.to_pixel_art(size: 24, color: :custom, palette: color_palette)

  # Save the image as a PNG file
  image.save("punks_images/punk_#{i + 1}.png")
end

puts "Generated #{num_images} CryptoPunk images."
```

This example will generate 5 CryptoPunk images with a custom color palette and save them in the `punks_images` directory.