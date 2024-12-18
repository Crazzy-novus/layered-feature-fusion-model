import csv
import os
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from utils import get_image_paths_and_labels, CustomDataset, load_image
from main_method import HifuseModel as HifuseClassifier



def main(imagepath, model_save_path,output_file_path, batch_size=32, model_load_path = "", num_classes=2):

    print ("training Started")

    device = torch.device("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    
    train_dataset = CustomDataset(root_dir=imagepath, transform="train")
    print("train path lenfth:", len(train_dataset))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    if model_load_path:
        hifusemodel = HifuseClassifier(num_classes=num_classes).to(device)
        hifusemodel.load_state_dict(torch.load(model_load_path, weights_only=True))
    else:
        hifusemodel = HifuseClassifier(num_classes=num_classes)
        hifusemodel = hifusemodel.to(device)

    ###### Training Setting and Epoch 1 #######

    hifusemodel.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer = optim.AdamW(hifusemodel.parameters(), lr=0.001, weight_decay=0.01)
    optimizer.zero_grad()

    sample_num = 0
    print("\nDataLoader Length:",data_loader.__len__())
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for epoch in range(1):
        running_loss = 0.0
        for iteration, data in enumerate(data_loader, 0):
            images, labels = data
            sample_num += images.shape[0]

            preditions = hifusemodel(images.to(device))
            pred_classes = torch.max(preditions, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()


            loss = loss_function(preditions, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()

           
            # with open(output_file_path, 'a') as f:
            #     f.write("\n[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}\n".format(
            #         iteration,
            #         accu_loss.item() / (iteration + 1),
            #         accu_num.item() / sample_num,
            #         optimizer.param_groups[0]["lr"]
            #     ))

                        
            # Check if the file exists to write the header only once
            # file_exists = os.path.isfile(output_file_path)

            # with open(output_file_path, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     if not file_exists:
            #         # Write the header
            #         writer.writerow(['Epoch', 'Iteration', 'Loss', 'Accuracy', 'Learning Rate'])
                
            #     # Write the training results
            #     writer.writerow([
            #         epoch,
            #         iteration,
            #         accu_loss.item() / (iteration + 1),
            #         accu_num.item() / sample_num,
            #         optimizer.param_groups[0]["lr"]
            #     ])


            data_loader.desc = "\n[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}\n".format(
                        epoch,
                        accu_loss.item() / (iteration + 1),
                        accu_num.item() / sample_num,
                        optimizer.param_groups[0]["lr"]
                    )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if iteration % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 200 :.5f}')
                running_loss = 0.0
    print ('Finished Training')
    # torch.save(hifusemodel.state_dict(), model_save_path)


if __name__ == '__main__':
    model_load_path = r""
    imagepath=r"E:\research papers\coding\zkvasirmodels\data\kvasir-dataset-split\test"
    model_save_path = r"E:\research papers\coding\hifuse-model-image-classifiaction\novelty\results\kvsir\models\kvasir_model_Epoch1_Local.pth"
    # model_load_path = r"E:\research papers\coding\novelty\results\kvsir\models\kvasir_model_Epoch1.pth"
    output_file_path = r'E:\research papers\coding\hifuse-model-image-classifiaction\novelty\results\kvsir\output\training\epoch_1_Local.csv'
    num_classes = 8
    main(imagepath=imagepath, model_save_path=model_save_path,output_file_path=output_file_path, batch_size=32,model_load_path=model_load_path, num_classes=num_classes)
