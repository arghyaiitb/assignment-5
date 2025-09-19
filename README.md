# MNIST Digit Classification - Neural Network Evolution

This project demonstrates the evolution of a Convolutional Neural Network (CNN) for MNIST digit classification, showcasing the journey from a basic model achieving 98.92% accuracy to an optimized model achieving **99.40% accuracy**.

## Model Evolution Journey

### Starting Point: Model 1 (`model_1.ipynb`)

#### Architecture Overview
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)      # 1→8 channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)     # 8→16 channels  
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3)     # 16→8 channels
        self.fc1 = nn.Linear(5*5*8, 80)                  # 200→80 neurons
        self.fc2 = nn.Linear(80, 10)                     # 80→10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))        # 28×28→26×26
        x = F.relu(self.conv2(x))        # 26×26→24×24
        x = F.max_pool2d(x, 2)           # 24×24→12×12
        x = F.relu(self.conv3(x))        # 12×12→10×10
        x = F.max_pool2d(x, 2)           # 10×10→5×5
        x = x.view(-1, 200)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

#### Model 1 Parameters Breakdown
| Layer | Input Shape | Output Shape | Parameters | Calculation |
|-------|-------------|--------------|------------|-------------|
| Conv1 | (1, 28, 28) | (8, 26, 26) | **80** | (3×3×1×8) + 8 = 80 |
| Conv2 | (8, 26, 26) | (16, 24, 24) | **1,168** | (3×3×8×16) + 16 = 1,168 |
| Conv3 | (16, 24, 24) | (8, 10, 10) | **1,160** | (3×3×16×8) + 8 = 1,160 |
| FC1 | (200,) | (80,) | **16,080** | (200×80) + 80 = 16,080 |
| FC2 | (80,) | (10,) | **810** | (80×10) + 10 = 810 |
| **Total** | | | **19,298** | |

#### Model 1 Configuration
- **Batch Size**: 100
- **Optimizer**: SGD with momentum=0.9, lr=0.01
- **Scheduler**: StepLR (step_size=15, gamma=0.1)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 15
- **Final Accuracy**: 98.92%

#### Model 1 Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

---

## Optimized Model: Model 5 (`model_5.ipynb`)

### Key Improvements Made

#### 1. **Enhanced Architecture with Batch Normalization**
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Added padding
        self.bn1 = nn.BatchNorm2d(16)                            # Added BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)                            # Added BatchNorm
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3)
        self.fc1 = nn.Linear(5*5*8, 60)                         # Reduced neurons
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))              # BatchNorm after conv1
        x = F.relu(self.bn2(self.conv2(x)))  # BatchNorm after conv2
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5*5*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

#### 2. **Model 5 Parameters Breakdown**
| Layer | Input Shape | Output Shape | Parameters | Calculation |
|-------|-------------|--------------|------------|-------------|
| Conv1 | (1, 28, 28) | (16, 28, 28) | **160** | (3×3×1×16) + 16 = 160 |
| BatchNorm2d | (16, 28, 28) | (16, 28, 28) | **32** | 16×2 = 32 |
| Conv2 | (16, 28, 28) | (32, 26, 26) | **4,640** | (3×3×16×32) + 32 = 4,640 |
| BatchNorm2d | (32, 26, 26) | (32, 26, 26) | **64** | 32×2 = 64 |
| Conv3 | (32, 13, 13) | (8, 11, 11) | **2,312** | (3×3×32×8) + 8 = 2,312 |
| FC1 | (200,) | (60,) | **12,060** | (200×60) + 60 = 12,060 |
| FC2 | (60,) | (10,) | **610** | (60×10) + 10 = 610 |
| **Total** | | | **19,878** | |

#### 3. **Advanced Data Augmentation**
```python
train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,           # Rotation range
        translate=(0.1, 0.1), # Translation
        scale=(0.9, 1.1),     # Scaling
        shear=10              # Shearing
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

#### 4. **Optimized Hyperparameters**
- **Batch Size**: 128 (increased from 100)
- **Optimizer**: SGD with momentum=0.9, lr=0.01
- **Scheduler**: StepLR (step_size=5, gamma=0.005) - more aggressive decay
- **Loss Function**: NLL Loss (F.nll_loss)
- **Epochs**: 19 (increased training duration)
- **Final Accuracy**: **99.40%**

## Key Architectural Improvements

### 1. **Batch Normalization Integration**
- Added `BatchNorm2d` layers after first two convolutions
- Helps with gradient flow and training stability
- Reduces internal covariate shift

### 2. **Padding Strategy**
- Added `padding=1` to first convolution layer
- Preserves spatial dimensions: 28×28 → 28×28 (instead of 28×28 → 26×26)
- Better feature preservation at borders

### 3. **Channel Progression Optimization**
- Model 1: 1 → 8 → 16 → 8 channels
- Model 5: 1 → 16 → 32 → 8 channels
- More aggressive feature extraction in early layers

### 4. **Fully Connected Layer Optimization**
- Reduced FC1 neurons from 80 to 60
- More efficient parameter usage
- Slight reduction in overfitting potential

### 5. **Enhanced Data Augmentation**
- **Model 1**: Basic rotation + center crop
- **Model 5**: Comprehensive affine transformations
  - Rotation: ±15 degrees
  - Translation: ±10% in both directions
  - Scale: 0.9 to 1.1
  - Shear: ±10 degrees

### 6. **Training Strategy Improvements**
- Increased batch size for better gradient estimates
- More aggressive learning rate decay (gamma=0.005 vs 0.1)
- Shorter decay intervals (step_size=5 vs 15)
- Extended training (19 vs 15 epochs)

## Performance Comparison

| Metric | Model 1 | Model 5 | Improvement |
|--------|---------|---------|-------------|
| **Test Accuracy** | 98.92% | 99.40% | +0.48% |
| **Parameters** | 19,298 | 19,878 | +580 (+3%) |
| **Architecture** | Basic CNN | CNN + BatchNorm | Enhanced |
| **Augmentation** | Basic | Advanced Affine | Comprehensive |
| **Training Epochs** | 15 | 19 | Extended |

## Training Results (Model 5)

The model achieved consistent improvement throughout training:
- **Epoch 1**: 97% accuracy
- **Epoch 2**: 99% accuracy  
- **Final (Epoch 19)**: 99.40% accuracy

The training showed stable convergence with the enhanced architecture and augmentation strategy.

## Key Learnings

1. **Batch Normalization** significantly improved training stability and convergence
2. **Padding preservation** helped maintain spatial information
3. **Advanced augmentation** improved model generalization
4. **Optimized learning rate scheduling** enabled better fine-tuning
5. **Strategic channel progression** enhanced feature extraction capability

This evolution demonstrates how systematic architectural improvements and hyperparameter tuning can push model performance from good (98.92%) to excellent (99.40%) while maintaining parameter efficiency.