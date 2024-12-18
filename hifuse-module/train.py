import csv
import os
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from utils import create_lr_scheduler, CustomDataset, get_params_groups
from main_method import HifuseModel as HifuseClassifier
from evaluate_model import evaluateModel

torch.manual_seed(42)

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
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    start_epoch = 0
    hifusemodel = HifuseClassifier(num_classes=num_classes)
    pg = get_params_groups(hifusemodel, weight_decay=0.01)
    
    optimizer = optim.AdamW(pg, lr=0.001, weight_decay=0.01)
    lr_scheduler = create_lr_scheduler(optimizer, len(data_loader), 100,
                                       warmup=True, warmup_epochs=1)
    if model_load_path:
        checkpoint = torch.load(model_load_path)
        hifusemodel.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
    else:
        hifusemodel = hifusemodel
        
    ###### Training Setting and Epoch 1 #######
    
    print("\nDataLoader Length:",data_loader.__len__())
    
    for epoch in range(start_epoch, start_epoch + 6):
        hifusemodel.train()
        loss_function = torch.nn.CrossEntropyLoss()
        accu_loss = torch.zeros(1)
        accu_num = torch.zeros(1)
        optimizer.zero_grad()
        running_loss = 0.0
        sample_num = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        data_loader = tqdm(data_loader, file=sys.stdout)
        
        for iteration, data in enumerate(data_loader, 0):
            images, labels = data
            sample_num += images.shape[0]

            preditions = hifusemodel(images)
            pred_classes = torch.max(preditions, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()


            loss = loss_function(preditions, labels)
            loss.backward()
            accu_loss += loss.detach()

                        
            # Check if the file exists to write the header only once
            file_exists = os.path.isfile(output_file_path)

            with open(output_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write the header
                    writer.writerow(['Epoch', 'Iteration', 'Loss', 'Accuracy', 'Learning Rate'])
                
                # Write the training results
                writer.writerow([
                    start_epoch + 1,
                    iteration,
                    round(accu_loss.item() / (iteration + 1), 4),  # Round loss to 4 decimal places
                    round(accu_num.item() / sample_num, 4),       # Round accuracy to 4 decimal places
                    round(optimizer.param_groups[0]["lr"], 6)
                ])


            data_loader.desc = "\n[train epoch {}] loss: {:.3f}, acc: {:.9f}, lr: {:.9f}\n".format(
                        start_epoch + 1,
                        accu_loss.item() / (iteration + 1),
                        accu_num.item() / sample_num,
                        optimizer.param_groups[0]["lr"]
                    )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            running_loss += loss.item()
        accuracy = evaluateModel(hifusemodel)

    print ('Finished Training')
    torch.save(hifusemodel.state_dict(), model_save_path)
    print('epoch:', epoch)
    print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
    checkpoint = {
        "net": hifusemodel.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'lr_schedule': lr_scheduler.state_dict()
    }
   
    torch.save(checkpoint, model_save_path)



if __name__ == '__main__':
    model_load_path = r""
    imagepath=r"E:\research papers\coding\zkvasirmodels\data\kvasir-dataset-split_2\train"
    model_save_path = r"E:\research papers\coding\zkvasirmodels\model\kvasir_model_epoch_10.pth"
    # model_load_path = r"E:\research papers\coding\novelty\results\kvsir\models\kvasir_model_Epoch4_set2.pth"
    model_load_path = r"E:\research papers\coding\zkvasirmodels\model\kvasir_model_epoch_4_new.pth"
    output_file_path = r'E:\research papers\coding\zkvasirmodels\output\training_result\epoch_10.csv'
    num_classes = 8
    main(imagepath=imagepath, model_save_path=model_save_path,output_file_path=output_file_path, batch_size=32,model_load_path=model_load_path, num_classes=num_classes)
