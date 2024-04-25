#目的為查看權重檔的變數
import torch
lz = torch.load('./best_modelloss0.8.pt')
for parameter in lz.parameters():
    print(parameter)
    print(parameter.size())