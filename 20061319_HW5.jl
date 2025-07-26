#Question 1: Train a Convolutional Neural network on the CIFAR10 dataset using the LeNet5 architecture
using Pkg
Pkg.add(["Flux", "MLDatasets", "Statistics", "CUDA", "Random", "Optimisers", "DataLoaders"])
using Flux
using Flux: DataLoader, flatten, logitcrossentropy, onecold, onehotbatch
using MLDatasets
using CUDA
using Statistics

# Check CUDA device
DEVICE = CUDA.has_cuda() ? gpu : cpu

# Load data
train_x, train_y = CIFAR10(:train)[:]
test_x, test_y = CIFAR10(:test)[:]

# Preprocess inputs
train_x = Float32.(train_x) ./ 255 |> DEVICE
test_x = Float32.(test_x) ./ 255 |> DEVICE

# One-hot encode labels
train_y = Flux.onehotbatch(train_y .+ 1, 1:10) |> DEVICE
test_y = Flux.onehotbatch(test_y .+ 1, 1:10) |> DEVICE

# Use Flux's DataLoader with batch size and shuffle
train_data = DataLoader((train_x, train_y), batchsize=128, shuffle=true)
test_data = DataLoader((test_x, test_y), batchsize=128)

# Define LeNet5 model
model = Chain(
    Conv((5,5), 3=>6, relu),
    MaxPool((2,2)),
    Conv((5,5), 6=>16, relu),
    MaxPool((2,2)),
    flatten,
    Dense(400, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10)
) |> DEVICE

# Optimizer
optimizer = Flux.Adam(1e-5)

# Loss function
loss(x, y) = logitcrossentropy(model(x), y)

# Accuracy function
function accuracy(x, y)
    ŷ = model(x)
    return mean(onecold(ŷ) .== onecold(y))
end

# Training loop
for epoch in 1:10
    println("\nEpoch ", epoch)
    for (x, y) in train_data
        x, y = x |> DEVICE, y |> DEVICE
        grads = gradient(() -> loss(x, y), Flux.params(model))
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)
    end

    acc = mean([accuracy(x |> DEVICE, y |> DEVICE) for (x, y) in test_data])
    println("Test Accuracy after epoch $epoch: $(round(acc * 100, digits=2))%")
end

#Question 2: 
#=Try to estimate the effect that new examples have on the performance of a neural network
compared to showing the same example several times. Train the network on a subset of 10000
examples for 6 epochs and evaluate the test set afterward. Now train on 20000 and 30000
examples while keeping the number of training steps identical i.e. 3 epoch for 20000 and 2
epochs for 30000 examples. Plot the final test accuracies and describe the observed behavior.=#
using Pkg
Pkg.add(["Flux", "MLDatasets", "Statistics", "CUDA", "Random", "Optimisers"])

using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, flatten, DataLoader
using MLDatasets
using CUDA
using Statistics
using Random

DEVICE = CUDA.has_cuda() ? gpu : cpu

train_x, train_y = CIFAR10(:train)[:]
test_x, test_y = CIFAR10(:test)[:]

train_x = Float32.(train_x) ./ 255 |> DEVICE
test_x = Float32.(test_x) ./ 255 |> DEVICE
train_y = onehotbatch(train_y .+ 1, 1:10) |> DEVICE
test_y = onehotbatch(test_y .+ 1, 1:10) |> DEVICE

function get_model()
    Chain(
        Conv((5,5), 3=>6, relu),
        MaxPool((2,2)),
        Conv((5,5), 6=>16, relu),
        MaxPool((2,2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10)
    ) |> DEVICE
end

function accuracy(model, data)
    mean([mean(onecold(model(x)) .== onecold(y)) for (x, y) in data])
end

function train_subset(subset_size, epochs)
    model = get_model()
    opt = Flux.Adam(1e-4)
    idx = randperm(size(train_x, 4))[1:subset_size]
    x_sub = train_x[:,:,:,idx]
    y_sub = train_y[:,idx]
    loader = DataLoader((x_sub, y_sub), batchsize=128, shuffle=true)

    for epoch in 1:epochs
        for (x, y) in loader
            x, y = x |> DEVICE, y |> DEVICE
            grads = gradient(() -> logitcrossentropy(model(x), y), Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end
    end

    test_loader = DataLoader((test_x, test_y), batchsize=128, shuffle=false)
    return accuracy(model, test_loader)
end


acc_10000 = train_subset(10000, 6)
acc_20000 = train_subset(20000, 3)
acc_30000 = train_subset(30000, 2)

println("Test Accuracy with 10000 samples, 6 epochs: $(round(acc_10000*100, digits=2))%")
println("Test Accuracy with 20000 samples, 3 epochs: $(round(acc_20000*100, digits=2))%")
println("Test Accuracy with 30000 samples, 2 epochs: $(round(acc_30000*100, digits=2))%")

using Plots

accuracies = [acc_10000, acc_20000, acc_30000]
labels = ["10k x 6", "20k x 3", "30k x 2"]

bar(labels, accuracies .* 100, legend=false, ylabel="Test Accuracy (%)", xlabel="Subset x Epochs",
    title="Effect of New Examples vs Repetition", color=:skyblue)

#Observed behavior:
#=
The results show that training with 20,000 unique examples for 3 epochs achieved the highest test accuracy (19.62%), 
compared to both training with 10,000 examples repeated over 6 epochs (18.99%) and 30,000 examples over only 2 epochs (18.37%). 
This suggests that increasing the diversity of training data contributes more to model performance 
than simply repeating the same limited data multiple times. However, 
the drop in accuracy for the 30,000 example case indicates that insufficient training time (only 2 epochs) 
may hinder the model’s ability to learn from the additional data. Therefore, 
a balance between data variety and training duration is essential for optimal performance.
=#

#Question 3: 
#=Measure the effect of filter size: Instead of using (5,5) filters in the LeNet5, use (3,3) filters
and (7,7) filters respectively to make a ”LeNet3” and ”LeNet7”. Plot the final test accuracies
and describe the observed behavior.=#
using Pkg
Pkg.add(["Flux", "MLDatasets", "Statistics", "CUDA", "Random", "Optimisers", "Plots"])

using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, flatten, DataLoader
using MLDatasets
using CUDA
using Statistics
using Random
using Plots

DEVICE = CUDA.has_cuda() ? gpu : cpu

train_x, train_y = CIFAR10(:train)[:]
test_x, test_y = CIFAR10(:test)[:]

train_x = Float32.(train_x) ./ 255 |> DEVICE
test_x = Float32.(test_x) ./ 255 |> DEVICE
train_y = onehotbatch(train_y .+ 1, 1:10) |> DEVICE
test_y = onehotbatch(test_y .+ 1, 1:10) |> DEVICE

train_data = DataLoader((train_x, train_y), batchsize=128, shuffle=true)
test_data = DataLoader((test_x, test_y), batchsize=128, shuffle=false)

function make_model(filter_size)
    # Feature extractor
    conv_part = Chain(
        Conv((filter_size, filter_size), 3=>6, relu),
        MaxPool((2,2)),
        Conv((filter_size, filter_size), 6=>16, relu),
        MaxPool((2,2))
    ) |> DEVICE

    # Pass a dummy image through to get output size
    dummy = rand(Float32, 32, 32, 3, 1) |> DEVICE
    conv_out = conv_part(dummy)
    flat_size = length(conv_out)

    return Chain(
        conv_part,
        flatten,
        Dense(flat_size, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10)
    ) |> DEVICE
end


function train_model(model, train_data, test_data, epochs=5)
    opt = Flux.Adam(1e-4)
    for epoch in 1:epochs
        for (x, y) in train_data
            x, y = x |> DEVICE, y |> DEVICE
            grads = gradient(() -> logitcrossentropy(model(x), y), Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end
    end
    return accuracy(model, test_data)
end

acc_3 = train_model(make_model(3), train_data, test_data)
acc_5 = train_model(make_model(5), train_data, test_data)
acc_7 = train_model(make_model(7), train_data, test_data)

println("Test Accuracy with (3,3) filters (LeNet3): $(round(acc_3 * 100, digits=2))%")
println("Test Accuracy with (5,5) filters (LeNet5): $(round(acc_5 * 100, digits=2))%")
println("Test Accuracy with (7,7) filters (LeNet7): $(round(acc_7 * 100, digits=2))%")

labels = ["LeNet3 (3x3)", "LeNet5 (5x5)", "LeNet7 (7x7)"]
accuracies = [acc_3, acc_5, acc_7]

bar(labels, accuracies .* 100, legend=false, ylabel="Test Accuracy (%)",
    title="Effect of Filter Size on LeNet Performance", color=:lightblue)

#Observed behavior:
#=
The test accuracies reveal that filter size has a measurable but modest impact on performance. 
LeNet5, using (5×5) filters, achieved the highest accuracy at 26.43%, 
suggesting it provides an effective receptive field size for capturing mid-level features in the CIFAR-10 images. 
LeNet3, with smaller (3×3) filters, performed slightly worse at 25.76%, likely due to its limited spatial context, 
which may hinder feature abstraction. LeNet7, using larger (7×7) filters, reached 26.09%, 
but may suffer from over-smoothing local patterns and increased parameter count without proportional learning benefit. T
hese results indicate that moderate-sized filters strike a balance between spatial resolution and model complexity, 
making (5×5) filters optimal for this architecture and dataset.
=#

#Question 4: 
#=Investigate the learned features of the convolution layers. Using any 3 samples, 
draw images showing how the image transforms as it passes the convolution layers of the LeNet3
architecture.=#
using Pkg
Pkg.add(["Flux", "MLDatasets", "CUDA", "Images", "Plots"])

using Flux
using Flux: onehotbatch, onecold, flatten
using MLDatasets
using CUDA
using Images
using Plots

DEVICE = CUDA.has_cuda() ? gpu : cpu

# Load and preprocess CIFAR-10
train_x, _ = CIFAR10(:train)[:]
train_x = Float32.(train_x) ./ 255 |> DEVICE

# Define LeNet3 with (3,3) filters
function make_lenet3()
    return Chain(
        Conv((3,3), 3=>6, relu),
        MaxPool((2,2)),
        Conv((3,3), 6=>16, relu),
        MaxPool((2,2))
    ) |> DEVICE
end

lenet3 = make_lenet3()

# Pick 3 sample images
samples = train_x[:,:,:,[1, 2, 3]]

# Function to get intermediate outputs
function get_activations(model, x)
    acts = []
    current = x
    for layer in model
        current = layer(current)
        push!(acts, current)
    end
    return acts
end

# Visualize feature maps
function plot_feature_maps(acts, sample_index)
    n_layers = length(acts)
    for i in 1:n_layers
        feature_map = acts[i][:,:,:,:]  # size: H x W x C x N
        fmap = cpu(feature_map[:,:,:,sample_index])
        num_channels = size(fmap, 3)
        max_show = min(num_channels, 6)  # Show up to 6 channels

        plot_title = "Layer $i Activations (Sample $sample_index)"
        p = plot(layout=(1, max_show), size=(120 * max_show, 120))
        for j in 1:max_show
            img = clamp01.(fmap[:, :, j])
            plot!(p[j], heatmap(img; aspect_ratio=1, axis=nothing, colorbar=false))
        end
        display(p)
    end
end

# Run for all 3 samples
for i in 1:3
    acts = get_activations(lenet3, samples)
    plot_feature_maps(acts, i)
end

#Observed behavior:
#=
The visualizations of the learned features from LeNet3 indicate that the convolution
layers produced sparse or near-zero activations across most channels for the selected inputs. 
This suggests that the early layers of the network may not have learned sufficiently informative filters during training. 
Possible causes include underfitting due to limited training epochs, low learning rate, or insufficient model capacity. 
Additionally, the fact that only minimal or no structural patterns appear in the activations implies that the 
convolution filters failed to capture key edge, texture, or color patterns from the input images. 
This aligns with the relatively low test accuracy (~25.76%) reported earlier for LeNet3, reinforcing that the model 
struggles to extract discriminative features with smaller (3×3) filters under the current training configuration.
=#
