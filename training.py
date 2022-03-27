from warnings import filterwarnings
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import PIL.Image as Image
from progress.bar import IncrementalBar
from tqdm import tqdm

filterwarnings(action='ignore', category=UserWarning)

# Sets device to cuda if available, cpu if otherwise.
def set_device(print_device = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if print_device:
        print(f"Using {device} device")
    return device

# Primary model for image training (specify 1 vs 3 channel)
# CNN using a series of 3 convolutional layers + 
# network of linear and ELU layers.
class image_model(nn.Module):
    def __init__(self, num_categories, num_channels, width, height, device=set_device(), hidden_filter_count=64, hidden_filter_count2=128):
        super(image_model, self).__init__()

        self.norm = nn.LayerNorm(normalized_shape=(3,height, width),
                                 device = device,
                                 dtype=torch.float)
        
        self.conv2D = nn.Conv2d(in_channels=num_channels,
                                out_channels=hidden_filter_count,
                                kernel_size=3,
                                bias=True,
                                device=device,
                                dtype=torch.float)

        self.conv2D_hidden = nn.Conv2d(in_channels=hidden_filter_count,
                                    out_channels=hidden_filter_count2,
                                    kernel_size=3,
                                    bias=True,
                                    device=device,
                                    dtype=torch.float)

        self.conv2D_hidden2 = nn.Conv2d(in_channels=hidden_filter_count2,
                                    out_channels=hidden_filter_count2,
                                    kernel_size=3,
                                    bias=True,
                                    device=device,
                                    dtype=torch.float)


        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    dilation=1)

        self.elu = nn.ELU()

        self.conv_block = nn.Sequential(self.conv2D,
                                        self.elu,
                                        self.maxpool,

                                        self.conv2D_hidden,
                                        self.elu,
                                        self.maxpool,

                                        self.conv2D_hidden2,
                                        self.elu,
                                        self.maxpool
                                        )

        self.batch_flatten = nn.Flatten(start_dim=1)
        self.single_flatten = nn.Flatten()

        self.linear_elu_stack = nn.Sequential(nn.LazyLinear(out_features=10000,
                                               device=device),
                                               nn.ELU(),

                                               nn.Linear(10000,10000,
                                               device=device),
                                               nn.ELU(),

                                               nn.Linear(10000,1000,
                                               device=device),
                                               nn.ELU(),

                                               nn.Linear(1000,num_categories,
                                               device=device))

        self.test = nn.LazyLinear(out_features=num_categories)
        self.sigmoid = nn.Sigmoid()


    def forward_with_shape(self,x):
        print(f"original shape = {x.size()}")
        x = self.norm(x)
        print(f"post-norm shape = {x.size()}")
        x = self.conv_block(x)
        print(f"post-conv shape = {x.size()}")
        x = self.single_flatten(x).cuda()
        print(f"post-flatten shape = {x.size()}")
        x = self.linear_elu_stack(x)
        print(f"post-linear shape = {x.size()}")
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_block(x)
        x = self.single_flatten(x)
        x = self.linear_elu_stack(x)
        x = self.sigmoid(x)
        return x

# Tests a model's accuracy against a dataset and returns it as a decimal.
def test_accuracy(model, dataloader, ones_only=False, device=set_device()):
    with torch.no_grad():
        guesses = []
        answers = []
        
        #fill guesses with model's predictions on all test cases
        #fill answers with correct predictions based on labels
        batch_count = dataloader.__len__()
        pbar = IncrementalBar(f"Testing accuracy: ", max=batch_count, suffix="%(percent)d%% done | approx. %(eta)d sec left")
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device=device)

            predicted = model(images)
            batch_guesses = torch.round(predicted).cpu().numpy() #get numpy array of rounded guesses (0 or 1 for each attribute * batch_size)
            for cur_guess in batch_guesses:
                guesses.append(cur_guess)
            for label in labels:
                answers.append(label.cpu().numpy())
            pbar.next()
        pbar.finish()
        guess_array = np.array(guesses)
        answer_array = np.array(answers)
        #calculate percent correct based on predictions and correct answers
        num_correct = 0.0
        num_images = guess_array.shape[0]
        num_attr = guess_array.shape[1]
        if not ones_only:
            for i in range(num_images):
                for j in range(num_attr):
                    if guesses[i][j] == answers[i][j]:
                        num_correct +=1
            return num_correct/(num_images*num_attr)
        else:
            total = 0
            for i in range(num_images):
                for j in range(num_attr):
                    if answers[i][j] == 1:
                        total += 1
                        if guesses[i][j] == 1:
                            num_correct += 1
            return num_correct/total



# Returns accuracy and loss data by epoch and batch
def train_model(model, loss_calc, optimizer, train_dataloader, test_dataloader, epoch_count, device=set_device()):
    path = input("Please enter a path to save model (saved as \'path_epochNum\'): ")
    model.to(device=device)
    print("*****BEGIN TRAINING*****")
    #initializing data lists for plotting later
    accuracies_by_epoch = []
    accuracies_by_batch = []
    loss_by_batch = []
    loss_by_epoch = []

    for epoch in range(epoch_count):
        batch_count = train_dataloader.__len__()
        pbar = IncrementalBar(f"Training epoch {epoch+1}/{epoch_count}", max=batch_count, suffix="%(percent)d%% done | approx. %(eta)d sec left in epoch")
        for step, (images, labels) in enumerate(train_dataloader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            ##processing whole batches at a time
            predicted = model(images)
            loss = loss_calc(predicted, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_by_batch.append(loss.item())
            pbar.next()
            if step == batch_count-1:
                print(f"Current guess: {predicted[0]}")

        #closing progress bar and printing epoch loss/accuracy
        pbar.finish()
        average_epoch_loss = sum(loss_by_batch)/len(loss_by_batch)
        print(f"    Average epoch loss = {average_epoch_loss}")
        print(f"    Accuracy = ...", end="\r")
        loss_by_epoch.append(average_epoch_loss)
        accuracy = test_accuracy(model, test_dataloader, device)
        accuracies_by_epoch.append(accuracy*100)
        print(f"    Accuracy = {(accuracy*100):.2f}%\n")
        epoch_path = f"{path}_{epoch}"
        torch.save(model.state_dict(), epoch_path)

    return accuracies_by_epoch, accuracies_by_batch, loss_by_epoch, loss_by_batch

# Saves a trained model to given path.
def save_model(model):
    model_saved = False
    while not model_saved:
        path = 'default'#input("Please enter a path to save the model: ")
        try:
            torch.save(model.state_dict(), path)
            model_saved = True
        except:
            print("**ERROR** Provided save path is invalid!")

# Creates and saves a plot for accuracy/loss by epoch.
def plot_accuracy_and_loss(accuracies, losses, epoch_count):

    # Set up plot figure
    epochs = np.arange(1,epoch_count+1)
    fig, axs = plt.subplots(2)
    fig.suptitle("Model Loss & Accuracy")
    plt.xlabel("Epoch")

    # Create loss plot
    axs[0].plot(epochs, np.array(losses), 'r-')
    axs[0].set_ylabel("Loss", color='red')
    axs[0].set_xticks(np.arange(1,epoch_count+1,1))

    # Create accuracy plot
    axs[1].plot(epochs, np.array(accuracies), 'b-')
    axs[1].set_ylabel("Accuracy[%]", color='blue')
    axs[1].set_xticks(np.arange(1,epoch_count+1,1))

    # Save plot
    plot_saved = False
    while not plot_saved:
        path = input("Please enter a path to save the plot: ")
        try:
            plt.savefig(path)
            plot_saved = True
        except:
            try:
                plt.savefig(path + '.png') # try adding a file type extension
                plot_saved = True
            except:
                print("**ERROR** Provided save path is invalid!")
    plt.show()