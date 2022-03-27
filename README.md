# Facial Feature Detection using CNN

### Overview
This is a collection of Python scripts used to train and run a convolutional neural network that works on feature detection with faces. Over 200,000 images, each annotated with 40 different attributes, have been used as inputs to the network. Attribute examples include hair color, age, certain accessories like necklaces, and facial hair. Using an existing model allows the user to display either model predictions as percents or as rounded, definitive answers. Current model accuracy sits at around 80%. 

### Network Structure
The current network is structured as a 3-layer convolutional neural network followed by a dense linear and ELU stack, ending with a sigmoid layer. Convolutional layers are defined as a 2-dimensional convolution, ELU, and maxpool sequence. Logits are then passed into a series of four linear + ELU pairs, producing 40 final logits that are parsed through a sigmoid activation function.

### File Structure
#### Helper Functions
Included in this repo are several "helper" scripts that are used in training and testing. Specifically, `training.py` contains functions for training a model based on given hyperparameters, as well as saving model weight dicts and accuracy plots. For testing new images, `testing.py` has functions to load in a user-specified image and model weight dict. Included are two functions for getting predictions - `get_prediction()` and `get_prediction_percents()`. The percents function directly returns the model's predictions as decimal probabilities (i.e. multiply by 100 for an actual percent). This does not return actual label names. The regular one rounds these predictions, returning a list of the label names that have a probability higher than 50%. This returns a list of Strings rather than probabilities. 

#### Info on `main.py`
The included main script combines user interface with the helper scripts to either train or test a new model. `main.py` takes care of the actual data processing and loading, as well as displaying predictions. 

### Dataset
This project was made possible by the CelebA Dataset, which is a collection of hundreds of thousands of celebrity images, each annotated with detail for facial attributes, landmarks, identity, etc. Due to an issue in PyTorch's built in CelebA dataset class, `main.py` implements a custom dataset. The data constructors as defined later in the script expect the CelebA attributes file to be located at `\celeb_data\list_attr_celeba.txt` and the images to be located in `celeb_data\img_align_celeba`, under their original names (e.g. 000137.jpg). These can of course be modified to fit another specified file path. 80% of the images are used as a trainset, with 20% used for validation. Currently, the custom class only supports gathering attribute data, but is planned to be expanded to utilizing CelebA's other annotations (next priority is likely facial landmarks).






@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}
