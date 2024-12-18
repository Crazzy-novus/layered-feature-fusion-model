import csv
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm  # Assuming you are using a ResNet18 model, change as needed
from utils import CustomDataset
from main_method import HifuseModel as HifuseClassifier
# Define the path to the model and the validation dataset
torch.manual_seed(42)
def evaluateModel (model=None, data_loader=None):

    if model is None:
        model_path = r"E:\research papers\coding\zkvasirmodels\model\model_save_path.pth"
        checkpoint = torch.load(model_path, weights_only=True)
        hifusemodel = HifuseClassifier(num_classes=num_classes)
        hifusemodel.load_state_dict(checkpoint['net'])
    else:
        hifusemodel = model

    if data_loader is None:
    # validation_data_path = "E:/research papers/coding/kvasir-dataset-split/train"
        validation_data_path = r"E:\research papers\coding\zkvasirmodels\data\kvasir-dataset-split_2\test"
        batch_size = 32
        num_classes = 8

        evaluate_dataset = CustomDataset(root_dir=validation_data_path, transform="validate")
        print("train path lenfth:", len(evaluate_dataset))


        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        evaluateData_loader = torch.utils.data.DataLoader(evaluate_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=evaluate_dataset.collate_fn)
    else:
        evaluateData_loader = data_loader
        
    hifusemodel.eval()

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    hifusemodel

    # Function to calculate accuracy
    def calculate_accuracy(model, data_loader, device):
        loss_function = torch.nn.CrossEntropyLoss()
        
        accu_num = torch.zeros(1)
        accu_loss = torch.zeros(1)
        sample_num = 0

        correct = 0
        total = 0
        with torch.no_grad():
            data_loader = tqdm(data_loader, file=sys.stdout)
        
            for iteration, data in enumerate(data_loader, 0):
                images, labels = data
                sample_num += images.shape[0]
                images, labels = images, labels
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                accu_num += torch.eq(predicted, labels).sum()

                loss = loss_function(outputs, labels)
                accu_loss += loss
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # output_file_path = './evaluating_output.txt'
                output_file_path = r'E:\research papers\coding\zkvasirmodels\output\evaluating_result\epoch_10.csv'
                file_exists = os.path.isfile(output_file_path)

                with open(output_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # Write the header
                        writer.writerow(['Iteration', 'Loss', 'Accuracy'])
                    
                    # Write the training results
                    writer.writerow([
                        iteration,
                        accu_loss.item() / (iteration + 1),
                        accu_num.item() / sample_num,
                        
                    ])



                data_loader.desc = "\nloss: {:.3f}, acc: {:.3f}\n".format(
                            
                            accu_loss.item() / (iteration + 1),
                            accu_num.item() / sample_num,
                            
                        )

        return 100 * correct / total

    # Calculate and print the accuracy
    accuracy = calculate_accuracy(hifusemodel, evaluateData_loader, device)
    print(f'Accuracy of the model on the validation dataset: {accuracy:.4f}')
    return accuracy



if __name__ == '__main__':
    evaluateModel()