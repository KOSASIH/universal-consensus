# Universal Consensus

This repository contains the implementation of the Universal Consensus project, a decentralized data exchange protocol that enables secure, transparent, and efficient data sharing and monetization.

## Prerequisites

To install requirements:

```setup
pip install -r requirements.txt
```

- Python 3.6
- GPU Memory: 10GB
- Pytorch 1.4.0
- Getting Started
- Download the dataset: Office-31, OfficeHome, VisDA, DomainNet.

## Data Folder structure:

```
Your dataset DIR:
|-Office/domain_adaptation_images
| |-amazon
| |-webcam
| |-dslr
|-OfficeHome
| |-Art
| |-Product
| |-...
|-VisDA
| |-train
| |-validation
|-DomainNet
| |-clipart
| |-painting
| |-...
```

You need to modify the data_path in config files, i.e., config.root

## Training 

Train on one transfer of Office:

```
CUDA_VISIBLE_DEVICES=0 python office_run.py note=EXP_NAME setting=uda/osda/pda source=amazon target=dslr
```

To train on six transfers of Office:

```
CUDA_VISIBLE_DEVICES=0 python office_run.py note=EXP_NAME setting=uda/osda/pda transfer_all=1
```

Train on OfficeHome:

```
CUDA_VISIBLE_DEVICES=0 python office_run.py note=EXP_NAME setting=uda/osda/pda source=Art target=Product
```

office_run.py

```python
1. import torch
2. import torch.nn as nn
3. import torch.optim as optim
4. from torch.utils.data import DataLoader
5. from torchvision import datasets, transforms
6. 
7. # Define the model, loss function, and optimizer
8. model = nn.Sequential(...)
9. criterion = nn.CrossEntropyLoss()
10. optimizer = optim.SGD(model.parameters(), lr=0.001)
11. 
12. # Load the dataset
13. transform = transforms.Compose([transforms.ToTensor()])
14. dataset = datasets.ImageFolder('path/to/dataset', transform=transform)
15. dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
16. 
17. # Train the model
18. for epoch in range(10):
19.    for batch in dataloader:
20.        inputs, labels = batch
21.        optimizer.zero_grad()
22.        outputs = model(inputs)
23.        loss = criterion(outputs, labels)
24.        loss.backward()
25.        optimizer.step()
26.    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

config.json

```json
1. {
2.    "root": "path/to/dataset",
3.    "uda": {
4.        "source": "amazon",
5.       "target": "dslr"
6.    },
7.    "osda": {
8.        "source": "Art",
9.        "target": "Product"
10.    },
11.    "pda": {
12.        "source": "clipart",
13.        "target": "painting"
14.    }
15. }
```

Please note that this is just a basic implementation, and you may need to modify the code to suit your specific requirements.
