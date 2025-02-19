# # from deeplib.datasets import load_mnist, train_valid_loaders
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from torchvision.transforms import ToTensor
# # import numpy as np
# # import torch
# # import random
# # from deeplib.visualization import plot_images
# # import torch
# # import time
# # import matplotlib.pyplot as plt

# # mnist, mnist_test = load_mnist()
# # mnist.transform = ToTensor()
# # mnist_test.transform = ToTensor()

# # class ConnectedRN(nn.Module):
# #     def __init__(self, in_val, out_val, n_hidden_layer):
# #         super().__init__()

# #         self.hidden_layers = nn.ModuleList([nn.Linear(in_val, in_val) for _ in range(n_hidden_layer)])
# #         self.output_layer = nn.Linear(in_val, out_val)

# #     def forward(self, go):
# #         go = go.flatten(1)
# #         for layer in self.hidden_layers:
# #             go = F.relu(layer(go))  # Utilisation de la fonction ReLU comme activation
# #         go = self.output_layer(go)
# #         return go


# # class ConnectedRN_Softmax(nn.Module):
# #     def __init__(self, input_size, hidden_sizes, output_size):
# #         super(ConnectedRN_Softmax, self).__init__()
# #         self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for hidden_size in range(hidden_sizes)])
# #         self.output_layer = nn.Linear(input_size, output_size)

# #     def forward(self, go):
# #         go = go.flatten(1)
# #         for layer in self.hidden_layers:
# #             go = F.relu(layer(go))  # Appliquer la couche linéaire et la fonction ReLU en une seule ligne
# #         go = self.output_layer(go)
# #         go = F.softmax(go, dim=1)
# #         return go
 
 
# #     def accuracy_score(true_val, pred_val):
# #         correct_predictions = sum(1 for x, y in zip(true_val, pred_val) if x == y)
# #         return correct_predictions / len(true_val)



# # def validation(model, criterion, data, is_mse=False, verbose=False):

# #     true_val = []
# #     predict_val = []

# #     predictions_img = []

# #     valid_loss = []

# #     model.eval()
# #     with torch.no_grad():
# #         for inputs, target in data:
# #             inputs = inputs.cuda()
# #             target = target.cuda()

# #             output = model(inputs)
# #             pred = output.max(dim=1)[1]

# #             if verbose:
# #                 for val, p in zip(inputs, pred):
# #                     el = {'img': val.cpu().numpy().squeeze(0), 'label': p.cpu().numpy()}
# #                     predictions_img.append(el)

# #             target_test = F.one_hot(target, num_classes=10).float() if is_mse else target

# #             loss = criterion(output, target_test)

# #             valid_loss.append(loss.item())

# #             true_val += target.cpu().numpy().tolist()
# #             predict_val += pred.cpu().numpy().tolist()

# #     accuracy = accuracy_score(true_val, predict_val)
# #     mean_loss = np.mean(valid_loss)

# #     return accuracy, mean_loss, predictions_img


# # batch_size = 128
# # n_epoch = 18
# # learning_rate = 0.1


# # def training_model(model, is_mse=False):
# #     model.cuda()

# #     criterion = nn.CrossEntropyLoss()

# #     if is_mse:
# #         criterion = nn.MSELoss()

# #     optimizer = optim.SGD(params=model.parameters(), momentum=0.2, lr=learning_rate)

# #     train_loader, valid_loader = train_valid_loaders(mnist, batch_size)

# #     train_accuracy = []
# #     test_accuracy = []
# #     test_losses = []
# #     train_losses = []
# #     start_global = time.time()

# #     model.train()
# #     for epoch in range(1, n_epoch + 1):
# #         start_time = time.time()

# #         train_loss = []
# #         true_val = []
# #         predict_val = []

# #         for data, target in train_loader:
# #             data = data.cuda()
# #             labels = target.cuda()

# #             optimizer.zero_grad()

# #             output = model(data)

# #             if is_mse:
# #                 labels = F.one_hot(target.cuda(), num_classes=10)
# #                 labels = labels.float()
# #                 output = output.float()

# #             loss = criterion(output, labels)
# #             pred = output.max(dim=1)[1]

# #             loss.backward()

# #             optimizer.step()

# #             train_loss.append(loss.item())
# #             true_val += target.cpu().numpy().tolist()
# #             predict_val += pred.cpu().numpy().tolist()

# #         train_acc = accuracy_score(true_val, predict_val)
# #         # train_acc, train_loss, _ = validation(model, criterion, train_loader, is_mse=True)
# #         val_acc, test_loss, _ = validation(model, criterion, valid_loader, is_mse=True)

# #         train_accuracy.append(round((1 - train_acc)*100, 1))
# #         test_accuracy.append(round((1 - val_acc)*100, 1))
# #         test_losses.append(test_loss)
# #         train_losses.append(np.mean(train_loss))


# #         print(f'Epoch {epoch}  - Train acc: {train_acc*100:.2f} - Val acc: {val_acc*100:.2f} - Train loss: {np.mean(train_loss):.4f} - {time.time() - start_time:2f}')

# #     print(f'time training {time.time() - start_global}')

# #     return train_accuracy, test_accuracy, test_losses, train_losses



# # def testing_model(train_accuracy, test_accuracy, name_file):
# #     epochs = np.arange(1, len(train_accuracy) + 1)

# #     plt.figure(figsize=(10, 6))

# #     plt.plot(epochs, train_accuracy, label='Erreur en entraînement', marker='o')
# #     plt.plot(epochs, test_accuracy, label='Erreur en validation', marker='o')

# #     plt.xticks(np.arange(0, len(train_accuracy) + 1, step=4))

# #     plt.xlabel('Époques')
# #     plt.ylabel('Taux d\'erreur en %')
# #     plt.title('Taux d\'erreur en entraînement et en validation')
# #     plt.legend()
# #     plt.grid(True)

# #     plt.savefig(name_file)
# #     plt.show()

# # def displayer_model(train_accuracy, test_accuracy, name_file):
# #     epochs = np.arange(1, len(train_accuracy) + 1)

# #     plt.figure(figsize=(10, 6))

# #     plt.plot(epochs, train_accuracy, label='perte en entraînement', marker='o')
# #     plt.plot(epochs, test_accuracy, label='perte en validation', marker='o')

# #     plt.xticks(np.arange(0, len(train_accuracy) + 1, step=4))

# #     plt.xlabel('Époques')
# #     plt.ylabel('Perte')
# #     plt.title('Perte en entraînement et en validation')
# #     plt.legend()
# #     plt.grid(True)

# #     plt.savefig(name_file)
# #     plt.show()
    
    
    
# # in_val = 28*28
# # out_val = 10
# # n_hidden_layer = 4


# # model = ConnectedRN(in_val, out_val, n_hidden_layer)

# # train_acc, test_acc, test, train = training_model(model)

# # testing_model(train_acc, test_acc, 'Question_4')

# # displayer_model(train, test, 'loss_4')


# # # Test avec MSE**

# # model = FullConnectedNetwork(28*28, 10, 4)

# # train_acc, test_acc = training_loop(model, is_mse=True)

# # testing_loop(train_acc, test_acc, 'Q4_accuracy_mse')




  
   
# from deeplib.datasets import load_mnist, train_valid_loaders
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision.transforms import ToTensor
# import numpy as np
# import torch
# import time
# import matplotlib.pyplot as plt
# from deeplib.visualization import plot_images

# # Chargement des données
# mnist, mnist_test = load_mnist()
# mnist.transform = ToTensor()
# mnist_test.transform = ToTensor()

# class ConnectedRN(nn.Module):
#     """Réseau de neurones entièrement connecté avec activation ReLU."""
    
#     def __init__(self, input_size, output_size, num_hidden_layers):
#         super().__init__()
#         self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_hidden_layers)])
#         self.output_layer = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         x = x.flatten(1)
#         for layer in self.hidden_layers:
#             x = F.relu(layer(x))  # Activation ReLU
#         return self.output_layer(x)


# class FullyConnectedNetworkSoftmax(nn.Module):
#     """Réseau de neurones entièrement connecté avec activation Softmax."""
    
#     def __init__(self, input_size, num_hidden_layers, output_size):
#         super().__init__()
#         self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_hidden_layers)])
#         self.output_layer = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         x = x.flatten(1)
#         for layer in self.hidden_layers:
#             x = F.relu(layer(x))
#         return F.softmax(self.output_layer(x), dim=1)
    
#     @staticmethod
#     def accuracy_score(true_values, predicted_values):
#         """Calcule la précision du modèle."""
#         return np.mean(np.array(true_values) == np.array(predicted_values))


# def validation(model, criterion, data_loader, is_mse=False, verbose=False):
#     """Évalue le modèle sur un ensemble de validation."""
    
#     model.eval()
#     true_values, predicted_values, predictions_img, losses = [], [], [], []

#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             inputs, targets = inputs.cuda(), targets.cuda()

#             outputs = model(inputs)
#             predictions = outputs.max(dim=1)[1]

#             if verbose:
#                 predictions_img.extend([{'img': inp.cpu().numpy().squeeze(0), 'label': pred.cpu().numpy()}
#                                         for inp, pred in zip(inputs, predictions)])

#             targets = F.one_hot(targets, num_classes=10).float() if is_mse else targets
#             loss = criterion(outputs, targets)
#             losses.append(loss.item())

#             true_values.extend(targets.cpu().numpy().tolist())
#             predicted_values.extend(predictions.cpu().numpy().tolist())

#     accuracy = FullyConnectedNetworkSoftmax.accuracy_score(true_values, predicted_values)
#     return accuracy, np.mean(losses), predictions_img


# def training_model(model, is_mse=False, batch_size=128, n_epoch=18, learning_rate=0.1):
#     """Entraîne le modèle et enregistre les performances."""
    
#     #model.cuda()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     criterion = nn.MSELoss() if is_mse else nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)

#     train_loader, valid_loader = train_valid_loaders(mnist, batch_size)

#     train_acc_list, test_acc_list, test_losses, train_losses = [], [], [], []
#     start_global = time.time()

#     model.train()
#     for epoch in range(1, n_epoch + 1):
#         start_time = time.time()
#         train_loss, true_values, predicted_values = [], [], []

#         for inputs, targets in train_loader:
#             inputs, targets = inputs.cuda(), targets.cuda()

#             optimizer.zero_grad()
#             outputs = model(inputs)

#             if is_mse:
#                 targets = F.one_hot(targets, num_classes=10).float()
#                 outputs = outputs.float()

#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             train_loss.append(loss.item())
#             predicted_values.extend(outputs.max(dim=1)[1].cpu().numpy().tolist())
#             true_values.extend(targets.cpu().numpy().tolist())

#         train_acc = FullyConnectedNetworkSoftmax.accuracy_score(true_values, predicted_values)
#         val_acc, test_loss, _ = validation(model, criterion, valid_loader, is_mse)

#         train_acc_list.append(round((1 - train_acc) * 100, 1))
#         test_acc_list.append(round((1 - val_acc) * 100, 1))
#         test_losses.append(test_loss)
#         train_losses.append(np.mean(train_loss))

#         print(f'Epoch {epoch}: Train Acc: {train_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}% - '
#               f'Train Loss: {np.mean(train_loss):.4f} - Time: {time.time() - start_time:.2f}s')

#     print(f'Training time: {time.time() - start_global:.2f}s')
#     return train_acc_list, test_acc_list, test_losses, train_losses


# def plot_training_results(train_values, test_values, filename, ylabel, title):
#     """Affiche l'évolution de l'erreur ou de la perte."""
    
#     epochs = np.arange(1, len(train_values) + 1)
#     plt.figure(figsize=(10, 6))
    
#     plt.plot(epochs, train_values, label='Entraînement', marker='o')
#     plt.plot(epochs, test_values, label='Validation', marker='o')
    
#     plt.xticks(np.arange(0, len(train_values) + 1, step=4))
#     plt.xlabel('Époques')
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
    
#     plt.savefig(filename)
#     plt.show()


# # Initialisation du modèle et entraînement
# input_size = 28 * 28
# output_size = 10
# hidden_layers = 4

# model = ConnectedRN(input_size, output_size, hidden_layers)
# train_acc, test_acc, test_losses, train_losses = training_model(model)

# # Visualisation des résultats
# plot_training_results(train_acc, test_acc, 'accuracy_plot.png', 'Taux d\'erreur (%)', 'Évolution du taux d\'erreur')
# plot_training_results(train_losses, test_losses, 'loss_plot.png', 'Perte', 'Évolution de la perte')

# # Entraînement avec MSE
# model_mse = ConnectedRN(input_size, output_size, hidden_layers)
# train_acc_mse, test_acc_mse, _, _ = training_model(model_mse, is_mse=True)

# plot_training_results(train_acc_mse, test_acc_mse, 'accuracy_mse_plot.png', 'Taux d\'erreur (%)', 'MSE: Évolution du taux d\'erreur')

# # Entraînement avec Softmax
# model_softmax = FullyConnectedNetworkSoftmax(input_size, hidden_layers, output_size)
# train_acc_softmax, test_acc_softmax, _, _ = training_model(model_softmax, is_mse=True)

# plot_training_results(train_acc_softmax, test_acc_softmax, 'accuracy_softmax_plot.png', 'Taux d\'erreur (%)', 'Softmax: Évolution du taux d\'erreur')
   
   
   
   
from deeplib.datasets import load_mnist, train_valid_loaders
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from deeplib.visualization import plot_images

# Chargement des données
mnist, mnist_test = load_mnist()
mnist.transform = ToTensor()
mnist_test.transform = ToTensor()

class ConnectedRN(nn.Module):
    """Réseau de neurones entièrement connecté avec activation ReLU."""
    
    def __init__(self, input_size, output_size, num_hidden_layers):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # Activation ReLU
        return self.output_layer(x)


class FullyConnectedNetworkSoftmax(nn.Module):
    """Réseau de neurones entièrement connecté avec activation Softmax."""
    
    def __init__(self, input_size, num_hidden_layers, output_size):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.softmax(self.output_layer(x), dim=1)
    
    @staticmethod
    def accuracy_score(true_values, predicted_values):
        """Calcule la précision du modèle."""
        return np.mean(np.array(true_values) == np.array(predicted_values))


def validation(model, criterion, data_loader, is_mse=False, verbose=False):
    """Évalue le modèle sur un ensemble de validation."""
    model.eval()
    true_values, predicted_values, validation_losses = [], [], []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            loss = criterion(outputs, F.one_hot(targets, num_classes=10).float() if is_mse else targets)
            validation_losses.append(loss.item())
            
            true_values.extend(targets.cpu().tolist())
            predicted_values.extend(predictions.cpu().tolist())
    
    return FullyConnectedNetworkSoftmax.accuracy_score(true_values, predicted_values), np.mean(validation_losses)


batch_size, n_epoch, learning_rate = 128, 18, 0.1

def train_model(model, is_mse=False):
    """Entraîne le modèle."""
    model.cuda()
    criterion = nn.MSELoss() if is_mse else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)
    train_loader, valid_loader = train_valid_loaders(mnist, batch_size)

    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    
    model.train()
    for epoch in range(1, n_epoch + 1):
        epoch_losses, true_values, predicted_values = [], [], []
        
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            
            outputs = model(data)
            labels = F.one_hot(target, num_classes=10).float() if is_mse else target
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            predicted_values.extend(outputs.argmax(dim=1).cpu().tolist())
            true_values.extend(target.cpu().tolist())
        
        train_acc = FullyConnectedNetworkSoftmax.accuracy_score(true_values, predicted_values)
        val_acc, val_loss = validation(model, criterion, valid_loader, is_mse)
        
        train_accs.append(train_acc * 100)
        test_accs.append(val_acc * 100)
        train_losses.append(np.mean(epoch_losses))
        test_losses.append(val_loss)
        
        print(f'Epoch {epoch}: Train acc: {train_acc * 100:.2f}% - Val acc: {val_acc * 100:.2f}% - Train loss: {np.mean(epoch_losses):.4f}')
    
    return train_accs, test_accs, train_losses, test_losses


def plot_training_results(train_acc, test_acc, filename):
    """Affiche les résultats de l'entraînement."""
    epochs = np.arange(1, len(train_acc) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Erreur entraînement', marker='o')
    plt.plot(epochs, test_acc, label='Erreur validation', marker='o')
    plt.xticks(np.arange(0, len(train_acc) + 1, step=4))
    plt.xlabel('Époques')
    plt.ylabel('Erreur (%)')
    plt.title("Évolution de l'erreur en entraînement et validation")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_loss(train_losses, test_losses, filename):
    """Affiche l'évolution de la perte."""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Perte entraînement', marker='o')
    plt.plot(epochs, test_losses, label='Perte validation', marker='o')
    plt.xticks(np.arange(0, len(train_losses) + 1, step=4))
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Évolution de la perte en entraînement et validation")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Entraînement du modèle
input_size, output_size, num_hidden_layers = 28*28, 10, 4
model = ConnectedRN(input_size, output_size, num_hidden_layers)
train_acc, test_acc, train_losses, test_losses = train_model(model)

# Affichage des résultats
plot_training_results(train_acc, test_acc, "accuracy_plot.png")
plot_loss(train_losses, test_losses, "loss_plot.png")

























































from deeplib.datasets import load_mnist, train_valid_loaders
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np
import torch
import random
from deeplib.visualization import plot_images
import torch
import time
import matplotlib.pyplot as plt

mnist, mnist_test = load_mnist()
mnist.transform = ToTensor()
mnist_test.transform = ToTensor()

class ConnectedRN(nn.Module):
    def __init__(self, in_val, out_val, n_hidden_layer):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(in_val, in_val) for _ in range(n_hidden_layer)])
        self.output_layer = nn.Linear(in_val, out_val)

    def forward(self, go):
        go = go.flatten(1)
        for layer in self.hidden_layers:
            go = F.relu(layer(go))  # Utilisation de la fonction ReLU comme activation
        go = self.output_layer(go)
        return go


class ConnectedRN_Softmax(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ConnectedRN_Softmax, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size) for hidden_size in range(hidden_sizes)])
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, go):
        go = go.flatten(1)
        for layer in self.hidden_layers:
            go = F.relu(layer(go))  # Appliquer la couche linéaire et la fonction ReLU en une seule ligne
        go = self.output_layer(go)
        go = F.softmax(go, dim=1)
        return go
def accuracy_score(true_val, pred_val):
    correct_predictions = sum(1 for x, y in zip(true_val, pred_val) if x == y)
    return correct_predictions / len(true_val)


def validation(model, criterion, data, is_mse=False, verbose=False):

    true_val = []
    predict_val = []

    predictions_img = []

    valid_loss = []

    model.eval()
    with torch.no_grad():
        for inputs, target in data:
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            pred = output.max(dim=1)[1]

            if verbose:
                for val, p in zip(inputs, pred):
                    el = {'img': val.cpu().numpy().squeeze(0), 'label': p.cpu().numpy()}
                    predictions_img.append(el)

            target_test = F.one_hot(target, num_classes=10).float() if is_mse else target

            loss = criterion(output, target_test)

            valid_loss.append(loss.item())

            true_val += target.cpu().numpy().tolist()
            predict_val += pred.cpu().numpy().tolist()

    accuracy = accuracy_score(true_val, predict_val)
    mean_loss = np.mean(valid_loss)

    return accuracy, mean_loss, predictions_img

batch_size = 128
n_epoch = 18
learning_rate = 0.1


def training_model(model, is_mse=False):
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    if is_mse:
        criterion = nn.MSELoss()

    optimizer = optim.SGD(params=model.parameters(), momentum=0.2, lr=learning_rate)

    train_loader, valid_loader = train_valid_loaders(mnist, batch_size)

    train_accuracy = []
    test_accuracy = []
    test_losses = []
    train_losses = []
    start_global = time.time()

    model.train()
    for epoch in range(1, n_epoch + 1):
        start_time = time.time()

        train_loss = []
        true_val = []
        predict_val = []

        for data, target in train_loader:
            data = data.cuda()
            labels = target.cuda()

            optimizer.zero_grad()

            output = model(data)

            if is_mse:
                labels = F.one_hot(target.cuda(), num_classes=10)
                labels = labels.float()
                output = output.float()

            loss = criterion(output, labels)
            pred = output.max(dim=1)[1]

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
            true_val += target.cpu().numpy().tolist()
            predict_val += pred.cpu().numpy().tolist()

        train_acc = accuracy_score(true_val, predict_val)
        # train_acc, train_loss, _ = validation(model, criterion, train_loader, is_mse=True)
        val_acc, test_loss, _ = validation(model, criterion, valid_loader, is_mse=True)

        train_accuracy.append(round((1 - train_acc)*100, 1))
        test_accuracy.append(round((1 - val_acc)*100, 1))
        test_losses.append(test_loss)
        train_losses.append(np.mean(train_loss))


        print(f'Epoch {epoch}  - Train acc: {train_acc*100:.2f} - Val acc: {val_acc*100:.2f} - Train loss: {np.mean(train_loss):.4f} - {time.time() - start_time:2f}')

    print(f'time training {time.time() - start_global}')

    return train_accuracy, test_accuracy, test_losses, train_losses

def testing_model(train_accuracy, test_accuracy, name_file):
    epochs = np.arange(1, len(train_accuracy) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_accuracy, label='Erreur en entraînement', marker='o')
    plt.plot(epochs, test_accuracy, label='Erreur en validation', marker='o')

    plt.xticks(np.arange(0, len(train_accuracy) + 1, step=4))

    plt.xlabel('Époques')
    plt.ylabel('Taux d\'erreur en %')
    plt.title('Taux d\'erreur en entraînement et en validation')
    plt.legend()
    plt.grid(True)

    plt.savefig(name_file)
    plt.show()

def displayer_model(train_accuracy, test_accuracy, name_file):
    epochs = np.arange(1, len(train_accuracy) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_accuracy, label='perte en entraînement', marker='o')
    plt.plot(epochs, test_accuracy, label='perte en validation', marker='o')

    plt.xticks(np.arange(0, len(train_accuracy) + 1, step=4))

    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.title('Perte en entraînement et en validation')
    plt.legend()
    plt.grid(True)

    plt.savefig(name_file)
    plt.show()

in_val = 28*28
out_val = 10
n_hidden_layer = 4


model = ConnectedRN(in_val, out_val, n_hidden_layer)

train_acc, test_acc, test, train = training_model(model)

testing_model(train_acc, test_acc, 'Question_4')

displayer_model(train, test, 'loss_4')

#Test avec MSE**        
in_val = 28*28
out_val = 10
n_hidden_layer = 4

model = ConnectedRN(in_val, out_val, n_hidden_layer)

# train_acc, test_acc = training_loop(model, is_mse=True)

# testing_loop(train_acc, test_acc, 'Q4_accuracy_mse')

train_acc, test_acc, test, train = training_model(model, is_mse=True)

testing_model(train_acc, test_acc, 'Q4_accuracy_mse')

displayer_model(train, test, 'loss_mse_4')


input_size = 28*28
output_size = 10
hidden_sizes = 4

model = ConnectedRN_Softmax(input_size, hidden_sizes, output_size)

train_acc, test_acc, test, train = training_model(model, is_mse=True)


testing_model(train_acc, test_acc, 'Q4_accuracy_mse_softmax')

displayer_model(train, test, 'loss_mse_4_softmax')