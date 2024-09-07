import torch.optim
from torch.utils.data import Dataset, DataLoader, random_split
from Dataset import MyDataset
from torch.utils.data import DataLoader
from Model import MLP
from torch.nn import MSELoss

if __name__ == "__main__":
    dataset = MyDataset("上海")
    dataloader = DataLoader(dataset)

    # split train dataset & test dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    model = MLP(12, 5, 1)


    criterion = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # train
    num_epochs = 1000
    for epoch in range(num_epochs):
        for data in train_dataset:
            # compute loss
            input, target = data
            output = model(input)
            loss = criterion(output, target)

            # back-propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            total_loss = 0
            total_num = 0
            for data in test_dataset:
                input, target = data
                output = model(input)
                loss = criterion(output, target)
                total_loss = total_loss + loss
                total_num = total_num + 1
            print("average test loss",total_loss/total_num)
