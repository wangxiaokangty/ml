import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = self.hidden_layer2(x)
        return x



if __name__ == '__main__':
    model = MLP(12,20,1)
    input = torch.tensor([ 0.2025, -1.1217, -0.4381, -0.6385, -0.1965, -0.2966, -0.8847, -0.8847,
                    -2.0672, -2.0672, -0.9016, -0.9016], dtype=torch.float32)
    print(model(input))