import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *
from models import *


# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}, 测试集大小: {len(test_data)}")

# 创建数据加载器
batch_sizes = [64, 128]
for batch_size in batch_sizes:
    print(f"批次大小：{batch_size}")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 加载模型
    models = {
       "model1": Model1(),
       "model2": Model2(),
     }
    for model_name, model in models.items():
        print(f"训练模型：{model_name}")
        if torch.cuda.is_available():
            model = model.cuda()

        # 定义损失函数和优化器
        loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        learning_rates = [1e-2, 1e-3]
        for learning_rate in learning_rates:
            print(f"学习率：{learning_rate}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练参数
            total_train_step = 0
            total_val_step = 0
            total_test_step = 0
            epoch = 10
            writer = SummaryWriter("logs_train")
            start_time = time.time()

            for i in range(epoch):
                print("-----第 {} 轮训练开始-----".format(i + 1))
                model.train()
                for data in train_dataloader:
                    imgs, targets = data
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        targets = targets.cuda()
                    output = model(imgs)
                    loss = loss_fn(output, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_step += 1
                    if total_train_step % 100 == 0:
                        end_time = time.time()
                        print(f"训练次数：{total_train_step}，loss：{loss.item()}")
                        writer.add_scalar('train_loss', loss.item(), total_train_step)

                # 评估验证集
                model.eval()
                total_val_loss = 0
                total_val_accuracy = 0
                best_val_accuracy = 0.00
                with torch.no_grad():
                    for data in val_dataloader:
                        imgs, targets = data
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            targets = targets.cuda()
                        output = model(imgs)
                        loss = loss_fn(output, targets)
                        total_val_loss += loss.item()
                        accuracy = (output.argmax(1) == targets).sum()
                        total_val_accuracy += accuracy
                val_accuracy = total_val_accuracy / len(val_data)
                print(f"验证集上的loss：{total_val_loss}")
                print(f"验证集上的正确率: {val_accuracy}")
                writer.add_scalar('val_loss', total_val_loss, total_val_step)
                writer.add_scalar('val_accuracy', val_accuracy, total_val_step)
                total_val_step += 1
                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), f'best_{model_name}_bs{batch_size}_lr{learning_rate}.pth')
                    print('最佳模型已保存')
            writer.close()
            # 加载最佳模型并在测试集上评估
            best_model = model
            best_model.load_state_dict(torch.load(f'best_{model_name}_bs{batch_size}_lr{learning_rate}.pth'))
            if torch.cuda.is_available():
                best_model = best_model.cuda()

            best_model.eval()
            total_test_loss = 0
            total_test_accuracy = 0
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for data in test_dataloader:
                    imgs, targets = data
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        targets = targets.cuda()
                    output = best_model(imgs)
                    predicted = output.argmax(1)
                    loss = loss_fn(output, targets)
                    total_test_loss += loss.item()
                    accuracy = (output.argmax(1) == targets).sum()
                    total_test_accuracy += accuracy
                    all_labels.extend(targets.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            cm = confusion_matrix(all_labels, all_preds)
            label_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
            title = f"{model_name} Confusion Matrix (bs={batch_size}, lr={learning_rate})"
            plot_confusion_matrix(cm, label_names, title)
            print(f"测试集上的loss：{total_test_loss}")
            print(f"测试集上的正确率: {total_test_accuracy / len(test_data)}")


