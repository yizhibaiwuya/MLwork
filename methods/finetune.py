from torch import log
from utils.train_utils import select_model, select_optimizer
from utils.data_loader import ImageDataset
import random
from torch.utils.data import DataLoader
import pandas as pd
import logging
import torch.nn as nn
import torch

logger = logging.getLogger()

class FullModule(nn.Module):
    def __init__(self, feature_extractor, feature_size, num_classes):
        super().__init__()
        self.featrue_extractor = feature_extractor
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, num_classes, bias=False)

    def forward(self, x):
        x = self.featrue_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

class Finetune:
    def __init__(self, criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs):
        self.num_learned_class = 0
        self.num_learning_class = init_class
        self.n_classes = n_classes
        self.device = device
        self.criterion = criterion
        self.model_name = kwargs['model_name']
        self.opt_name = kwargs['opt_name']
        self.sched_name = kwargs['sched_name']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.n_epoch = kwargs['n_epoch']
        self.n_woker = kwargs['n_worker']
        self.dataset = kwargs['dataset']
        self.topk = 1

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_list = None
        self.test_list = None
        self.memory_list = []
        self.memory_size = kwargs['memory_size']

        self.feature_extractor, self.feature_size = select_model(self.model_name)
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.model = FullModule(self.feature_extractor, self.feature_size, self.num_learning_class)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)


    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.train_list = train_datalist
        self.test_list = test_datalist

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = ImageDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader

    def before_task(self, datalist):
        logger.info("Apply before_task")
        incoming_classes = pd.DataFrame(datalist)["label"].unique().tolist()
        incoming_classes = list(set(incoming_classes))
        incoming_classes = 1 + max(incoming_classes)
        logger.info("Increasing fc layer:  {}  ---->  {}".format(self.num_learned_class, incoming_classes))
        self.num_learning_class = incoming_classes
        
        # update classifier
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        weight = self.model.fc.weight.data

        self.model.fc = nn.Linear(
                in_features, self.num_learning_class, bias=False
            )

        # keep weights for the old classes
        self.model.fc.weight.data[:out_features] = weight

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.model = self.model.to(self.device)

        # reinitialize the optimizer and scheduler
        logger.info("Reset the optimizer and scheduler states")
        self.optimizer, self.scheduler = select_optimizer(
            self.opt_name, self.lr, self.model, self.sched_name
        )

    def train(self, cur_iter):
        logger.info("#" * 10 + "Start Training" + "#" * 10)
        train_list = self.train_list + self.memory_list
        test_list  = self.test_list
        train_loader, test_loader = self.get_dataloader(
            self.batch_size, self.n_woker, train_list, test_list
        )

        logger.info(f"New training samples: {len(self.train_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        best_acc = 0.0
        eval_dict = dict()
        n_batches = len(train_loader)
        for epoch in range(self.n_epoch):
            if epoch > 0:
                self.scheduler.step()
            
            total_loss, correct, num_data = 0.0, 0.0, 0.0
            self.model.train()
            for i, data in enumerate(train_loader):
                x = data['image'].to(self.device)
                y = data['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                logit = self.model(x)
                loss = self.criterion(logit, y)
                preds = torch.argmax(logit, dim=-1)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y).item()
                num_data += y.size(0)
            
            eval_dict = self.evaluation(test_loader=test_loader, criterion=self.criterion)

            cls_acc = "cls_acc: ["
            for _ in eval_dict['cls_acc']:
                cls_acc += format(_, '.3f') + ', '
            cls_acc += ']'

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{self.n_epoch} | lr {self.optimizer.param_groups[0]['lr']:.4f} | train_loss {total_loss/n_batches:.4f} | train_acc {correct/num_data:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} |"
            )

            # 输出每个类别的准确率，但是cifar100类别数目一多，输出就太乱了，取决你们自己
            #logger.info(cls_acc)


            best_acc = max(best_acc, eval_dict["avg_acc"])
            
        return best_acc, eval_dict


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.num_learning_class)
        num_data_l = torch.zeros(self.num_learning_class)
        label = []

        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
                #if (i+1)%5 == 0:
                #    print("y = {},   logit = {}".format(y, logit))
                #print("y = ", y)
                #print("logit = ", logit)
                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)

                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()

                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)

                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()


        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret
        
    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.num_learning_class)
        ret_corrects = torch.zeros(self.num_learning_class)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects


    def after_task(self, cur_iter):
        self.num_learned_class = self.num_learning_class
        # update memory list if needed
        

        # random sample

        k = self.memory_size // self.num_learning_class
        tmp = [[] for _ in range(self.num_learning_class)]
        for _ in self.memory_list + self.train_list:
            tmp[_['label']].append(_)
        self.memory_list = []
        for _ in tmp:
        #    print(_)
            self.memory_list.extend(_[:k])

