import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##### TO BE ADDED
# Normalizing by channel and why do we need it
# MultiLabelImageClassificationBase (from Protein notebook)
# F_Score (from Protein notebook)


def denorm(img_tensors, data_stats):
    """
    Denormalize an image tensor.
    
    This method reverses the normalization process,
    recovering the "shape" of the original data.
    It uses the std_dev and mean of the original data.

    :param img_tensors: image tensors
    :type img_tensor: Pytorch tensor
    :param data_stats: Standard Deviation and Mean for the dataset
    :type data_stats: dict(mean, std_dev)
    """
    return img_tensors * data_stats["std_dev"] + data_stats["mean"]


def show_sample(img, target, invert=False):
    """
    Show a sample image with its labels
    
    invert: invert colors of the image. Default is False.
    """
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))


def show_images(images, nmax=64, denorm=False):
    """
    Display images in a grid

    :param images: a batch of images
    :type images: image batch
    :param nmax: max number of images to show
    :type nmax: int
    :param denorm: denormalize the images before display. Default=False
    :type denorm: bool
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    if not denorm:
        ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))
    else:
        ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(data_loader, nmax=64, denorm=False):
    """
    Display a batch of images

    :param data_loader: A DataLoader
    :type data_loader: Pytorch DataLoader
    :param nmax: max number of images to show
    :type nmax: int
    :param denorm: denormalize the images before display. Default=False
    :type denorm: bool
    """
    for images, _ in data_loader:
        show_images(images, nmax, denorm)
        break


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(_input, device):
    """
    Move input to selected device

    :param _input: data or model
    :type _input: Pytorch tensors
    :param device: device to be used
    :type device: Pytorch device
    """
    if isinstance(_input, (list, tuple)):
        return [to_device(x, device) for x in _input]
    return _input.to(device, non_blocking=True)


def images_to_video(fname, img_dir, sort=False):
    """
    Create a video of the images in a dir
    Uses OpenCV

    :param fname: output video file name
    :type fname: str
    :param img_dir: dir containing images
    :type img_dir: str
    :param sort: sort the files alphabetically
    :type sort: boolean (False by default)
    """
    fname = f'{fname}.avi'

    # create a list of files in the dir
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    if sort:
        files.sort()

    # create an output video writer object
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'MP4V'), 1, (530, 530))

    # write each file to the video writer
    [out.write(cv2.imread(fname)) for fname in files]
    out.release()  # save video


def get_class_counts(data):
    """
    Loop over the input data and Get counts of each class

    :param data: input data
    :type data: torchvision dataset
    :rtype dict()
    """
    class_counts = dict()
    for _, class_idx in data:
        if class_idx not in class_counts:
            # add class if not already in the dict
            class_counts[class_idx] = 1
        else:
            # update class count
            class_counts[class_idx] += 1

    return class_counts


def get_accuracy(outputs, labels):
    """
    Calculate accuracy

    :param outputs: predicted outputs
    :type outputs: Pytorch tensor
    :param labels: actual labels
    :type labels: Pytorch tensor
    :rtype Pytorch tensor
    """
    # torch.max returns a namedtuple (values, indices) where values is the
    # maximum value of each row of the input tensor in the given dimension dim.
    # And indices is the index location of each maximum value found (argmax).
    # Here we are only getting the argmax
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    """
    Evaluate a trained model

    :param model: trained model
    :type model: An ImageClassification model
    :param val_loader: validation loader
    :type val_loader: Pytorch DataLoader
    """
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """
    Fit the model on the training data

    :param epochs: number of epochs
    :type epochs: int
    :param lr: learning rate
    :param lr: float
    :param model: model to train
    :type model: An ImageClassification model
    :param train_loader: training data loader
    :type train_loader: Pytorch DataLoader
    :param val_loader: validation data loader
    :type val_loader: Pytorch DataLoader
    :param opt_func: optimizer function. Default = torch.optim.SGD
    :type opt_func: Pytorch optimizer (torch.optim.*)
    """
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        ###### Training Phase ######
        for batch in train_loader:
            # take a training step on the batch and compute the loss
            loss = model.training_step(batch)
            # propagate the loss backwards to get the gradients
            loss.backward()
            # optimize the model using the optimizer and lr
            optimizer.step()
            # zero-out the gradients before starting with next batch
            optimizer.zero_grad()

        ###### Validation phase ######
        result = evaluate(model, val_loader)  # evaluate the trained model
        # output the average val_loss and val_acc for this epoch
        model.epoch_end(epoch, result)
        # append this epoch to the history and continue with remaining epochs
        history.append(result)
    return history


def plot_losses(history):
    """
    Plot losses from the training history

    :param history: model training history
    :type history: list
    """
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')


def plot_accuracies(history):
    """
    Plot accuracies from the training history

    :param history: model training history
    :type history: list
    """
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def dataframe_to_arrays(dataframe):
    """
    Convert a pandas dataframe to numpy arrays

    :param dataframe: a pandas dataframe containing the data
    :type dataframe: pandas dataframe
    """
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array


def train_val_dataset_split(dataset, val_split=0.25):
    """
    split the data into training and validation set
    @msminhas903 https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4
    Note, the use needs to specify at least one transform

    Use:
    dataset = ImageFolder(DATA_DIR, transform=tt.Compose([tt.Resize((224,224))]))
    print(f"Total Dataset Length: {len(dataset)}")
    datasets = train_val_dataset_split(dataset, val_split=0.10)
    print(f"Length of Training Data: {len(datasets['train'])}")
    print(f"Length of Validation Data: {len(datasets['val'])}")
    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                        test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class TransformedSubset(Subset):
    """
    Apply transforms to Subsets
    @ptrblck https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/3

    Use:
    After splitting the dataset using train_val_dataset_split

    train_ds = TransformedSubset(datasets['train'], transform=training_trfms)
    val_ds = TransformedSubset(datasets['val'], transform=valid_trfms)
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.data_loader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.data_loader)


class ImageClassificationBase(nn.Module):
    """
    Base class for Image Classification
    
    :param loss_fn_train: loss function for training
    :type loss_fn_train: 'str', {'CE', 'BCE', 'BCELogits', 'L1', 'MSE'}
    :param loss_fn_val: loss function for validation.
                        If not specified, use the same function as training.
    :type loss_fn_val: 'str', {'CE', 'BCE', 'BCELogits', 'L1', 'MSE'}
    """
    def __init__(self, loss_fn_train, loss_fn_val=None):
        self.loss_fn_train = loss_fn_train
        if loss_fn_val:
            self.loss_fn_val = loss_fn_val
        else:
            self.loss_fn_val = loss_fn_train

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions

        # Calculate training loss
        if self.loss_fn_train == 'CE':
            # cross entropy loss
            loss = F.cross_entropy(out, labels)
        elif self.loss_fn_train == 'BCE':
            # binary cross entropy loss
            loss = F.binary_cross_entropy(out, labels)
        elif self.loss_fn_train == 'BCELogits':
            # binary cross entropy with logits loss
            loss = F.binary_cross_entropy_with_logits(out, labels)
        elif self.loss_fn_train == 'L1':
            # smooth L1 loss. Uses a squared term if the absolute
            # element-wise error falls below 1 and an L1 term otherwise
            loss = F.smooth_l1_loss(out, labels)
        elif self.loss_fn_train == 'MSE':
            # mean squared error loss
            loss = F.mse_loss(out, labels)
        else:
            print(f"Training loss function not supported!")
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        
        # Calculate validation loss
        if self.loss_fn_val == 'CE':
            # cross entropy loss
            loss = F.cross_entropy(out, labels)
        elif self.loss_fn_val == 'BCE':
            # binary cross entropy loss
            loss = F.binary_cross_entropy(out, labels)
        elif self.loss_fn_val == 'BCELogits':
            # binary cross entropy with logits loss
            loss = F.binary_cross_entropy_with_logits(out, labels)
        elif self.loss_fn_val == 'L1':
            # smooth L1 loss. Uses a squared term if the absolute
            # element-wise error falls below 1 and an L1 term otherwise
            loss = F.smooth_l1_loss(out, labels)
        elif self.loss_fn_val == 'MSE':
            # mean squared error loss
            loss = F.mse_loss(out, labels)
        else:
            print(f"Validation loss function not supported!")

        acc = get_accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, \
                val_acc: {result['val_acc']:.4f}")
