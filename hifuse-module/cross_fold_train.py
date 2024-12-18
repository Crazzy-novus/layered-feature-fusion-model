import csv
import os
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils import adjust_learning_rate, clip_gradients, create_lr_scheduler, get_image_paths_and_labels, CustomDataset, get_params_groups, load_image
from main_method import HifuseModel as HifuseClassifier
from evaluate_model import evaluateModel

def main(imagepath, model_save_path, output_file_path, batch_size=32, model_load_path="", num_classes=2):
    print("Training Started")

    device = torch.device("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load dataset
    dataset = CustomDataset(root_dir=imagepath, transform="train")
    print("Dataset length:", len(dataset))

    # Initialize KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")

        # Create data loaders for the current fold
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

        # Initialize model, optimizer, and scheduler
        hifusemodel = HifuseClassifier(num_classes=num_classes).to(device)
        optimizer = optim.AdamW(hifusemodel.parameters(), lr=0.001, weight_decay=0.01)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs=10, warmup=True, warmup_epochs=1)

        # Load checkpoint if specified
        start_epoch = 0
        if model_load_path:
            checkpoint = torch.load(model_load_path, map_location=device)
            hifusemodel.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            lr_scheduler.load_state_dict(checkpoint['lr_schedule'])

        # Training loop
        for epoch in range(start_epoch, start_epoch + 10):
            hifusemodel.train()
            loss_function = torch.nn.CrossEntropyLoss()
            accu_loss = torch.zeros(1).to(device)
            accu_num = torch.zeros(1).to(device)
            optimizer.zero_grad()
            running_loss = 0.0
            sample_num = 0
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loader = tqdm(train_loader, file=sys.stdout)

            for iteration, data in enumerate(train_loader, 0):
                images, labels = data
                sample_num += images.shape[0]
                print("-----------------------------------", sample_num)

                predictions = hifusemodel(images.to(device))
                pred_classes = torch.max(predictions, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()

                loss = loss_function(predictions, labels.to(device))
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
                        epoch,
                        iteration,
                        round(accu_loss.item() / (iteration + 1), 4),  # Round loss to 4 decimal places
                        round(accu_num.item() / sample_num, 4),       # Round accuracy to 4 decimal places
                        round(optimizer.param_groups[0]["lr"], 6)
                    ])

                train_loader.desc = "\n[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}\n".format(
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
                lr_scheduler.step()
                running_loss += loss.item()

            evaluateModel(hifusemodel)

        print('Finished Training Fold', fold + 1)
        torch.save(hifusemodel.state_dict(), f"{model_save_path}_fold{fold + 1}.pth")
        print('epoch:', epoch)
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        checkpoint = {
            "net": hifusemodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_scheduler.state_dict()
        }
        torch.save(checkpoint, f"{model_save_path}_fold{fold + 1}_checkpoint.pth")

if __name__ == '__main__':
    model_load_path = r""
    imagepath = r"E:\research papers\coding\zkvasirmodels\data\kvasir-dataset-split_2\train"
    model_save_path = r"E:\research papers\coding\zkvasirmodels\model\kvasir_model_set2_epoch_12"
    output_file_path = r'E:\research papers\coding\zkvasirmodels\output\training_result\set2_epoch_12.csv'
    num_classes = 8
    main(imagepath=imagepath, model_save_path=model_save_path, output_file_path=output_file_path, batch_size=32, model_load_path=model_load_path, num_classes=num_classes)