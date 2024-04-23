import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from model.loss.con_loss import SimCLRLoss, SupConLoss
from model.models import stratified_layerNorm
from torch.utils.data import DataLoader
from data.data_process import *
from data.dataset import PDataset, FACED_Dataset
from io_utils import *
from reorder_vids import *
import math
from captum.attr import (
    IntegratedGradients,
)




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # if k == 1:
            #     correct_k = correct_k - eq_num
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


    



class TrainFramework(object):   #cross training framework with Contrastive learning loss and supervised classification loss
    def __init__(self, args, extractor, mlp, opt_ext=None, opt_mlp=None, log_dir='', sche_ext=None, sche_mlp=None, temperature=0.07):
        self.args = args
        self.extractor = extractor
        self.mlp = mlp
        self.opt_ext = opt_ext
        self.opt_mlp = opt_mlp
        self.sche_ext = sche_ext
        self.sche_mlp = sche_mlp
        self.log_dir = log_dir
        self.SupCon = SupConLoss(temperature, 'all', temperature)
        self.SimCLRLoss = SimCLRLoss(temperature)
        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.device = device
        self.SupCon = self.SupCon.to(device)
        self.SimCLRLoss = self.SimCLRLoss.to(device)
        self.CEL = self.CEL.to(device)
        return self

    def accuracy(self, output, mask):
        # acc 一行正确率的平均值
        # acc_allright 一行（一个anchor对应同类）全对的正确率
        with torch.no_grad():
            acc_ave = 0
            acc_allright = 0
            for i in range(output.shape[0]):
                _, pre = output[i].topk(int(mask[i].sum().item()))
                acc_one = mask[i,pre].sum().item()/mask[i].sum().item()
                acc_ave += acc_one
                if acc_one==1:
                    acc_allright+=1
            acc_ave /= (i+1)
            acc_allright /= (i+1)
            return acc_ave, acc_allright

    def normal_train_fn(self, train_loader, val_loader, train_loader2, val_loader2):
        self.extractor_train_fn_normal(train_loader, val_loader, self.args.epoch_ext_normal)
        if self.args.use_best_pretrain:
            checkpoint = torch.load(os.path.join(self.log_dir, 'best_ext_cp.pth.tar'), map_location=self.device)
            state_dict = checkpoint['state_dict']
            self.extractor.load_state_dict(state_dict, strict=True)

        best_acc, best_loss, best_train_acc, best_train_loss = self.mlp_train_fn_normal(
                                                            train_loader2, val_loader2, self.args.epoch_mlp_normal)
        return best_acc, best_loss, best_train_acc, best_train_loss

    def cross_train_fn(self, train_loader, val_loader, train_loader2, val_loader2):
        rounds = self.args.rounds
        max_tol = self.args.max_tol
        
        # bad_count = 0
        round_best_acc = -1

        epoch_ext = self.args.epoch_ext
        epoch_mlp = self.args.epoch_mlp

        for r in range(rounds):
            print('round:',r)
            epoch_ext -= 1
            epoch_mlp += 1
            if self.args.para_evolve1:
                Lambda1 = max(2*(rounds-r)/rounds-1,0)
            else:
                Lambda1 = self.args.Lambda1
            
            if self.args.para_evolve2:
                Lambda2 = r/rounds
            else:   
                Lambda2 = self.args.Lambda2
            start_time = time.time()
            self.extractor_train_fn(train_loader, val_loader,Lambda1, Lambda2, epoch_ext, r)
            best_acc, best_loss, best_train_acc, best_train_loss, best_epoch = self.mlp_train_fn(
                                                            train_loader2, val_loader2, epoch_mlp, r)
            end_time = time.time()
            print('cross round time consumed:', end_time - start_time)

            if best_acc > round_best_acc:
                bad_count = 0
                round_best_loss = best_loss
                round_best_acc = best_acc
                round_best_train_loss = best_train_loss
                round_best_train_acc = best_train_acc
                round_best_epoch = best_epoch
                best_round = r

            else:
                bad_count+=1
            
            if bad_count > max_tol:
                print('best round %d, train loss: %.4f, train acc: %.3f, val loss: %.4f, val acc: %.3f' % (best_round,round_best_train_loss,round_best_train_acc,round_best_loss,round_best_acc))
                break
        
        print('best round %d, train loss: %.4f, train acc: %.3f, val loss: %.4f, val acc: %.3f' % (best_round,round_best_train_loss,round_best_train_acc,round_best_loss,round_best_acc))
        
        checkpoint_name = 'ext_cp_r{:02d}.pth.tar'.format(best_round)
        save_path = os.path.join(self.log_dir, checkpoint_name)
        copy_path = os.path.join(self.log_dir, 'best_ext_cp.pth.tar')
        copy_checkpoint(save_path,copy_path)
        checkpoint_name = 'mlp_cp_r{:02d}_e{:02d}.pth.tar'.format(best_round,round_best_epoch)
        save_path = os.path.join(self.log_dir, checkpoint_name)
        copy_path = os.path.join(self.log_dir, 'best_mlp_cp.pth.tar')
        copy_checkpoint(save_path,copy_path)
        return round_best_train_acc, round_best_acc

    def attribution_analysis(self, val_loader2, target_label, onesub_n_samples):
        # 对初始data做train norm

        # 对fea做norm和lds  生成数据迭代器
        print('extract data pretrain feature!')
        self.extractor.eval()
        self.extractor.set_saveFea(True)
        self.extractor.set_stratified = []
        pfea_val, label_val = self.ext_fea_process_independent(val_loader2, self.args.val_sub, self.args.val_vid_inds, 
                                                               self.args.n_samples2, self.args.n_vids,self.args.extract_mode, self.args.norm_decay_rate)

        print('data_val_min:', np.min(pfea_val))
        pdataset_val = PDataset(pfea_val,label_val)
        ploader_val = DataLoader(dataset=pdataset_val, batch_size=onesub_n_samples, shuffle=False, num_workers=8)

        self.extractor.set_saveFea(False)
        self.extractor.set_stratified = self.args.stratified

        print('starting attribution analysis...')
        self.mlp.eval()
        attributions_all = torch.zeros((onesub_n_samples * len(self.args.val_sub), self.args.fea_dim))
        for counter, (data, labels) in enumerate(ploader_val):
            # print('counter:', counter)
            data = data.to(self.device)
            labels = labels.to(self.device)

            baseline = np.min(pfea_val)*torch.ones_like(data).to(self.device)
            # print(baseline)

            ig = IntegratedGradients(self.mlp)
            attributions, delta = ig.attribute(data, baseline, target=target_label, return_convergence_delta=True)
            # print('IG Attributions:', attributions)
            # print('Convergence Delta:', delta)

            attributions_all[onesub_n_samples * counter: onesub_n_samples * (counter+1), :] = attributions.cpu()
        
        attrs = attributions_all.numpy()
        
        return attrs


    def extractor_train_fn(self,train_loader,val_loader,Lambda1, Lambda2, epochs, current_round):
        for epoch in range(epochs):
            start_time = time.time()
            print(f'round: {current_round}   extractor training epoch {epoch}')

            self.extractor.train()
            self.extractor.set_stratified = self.args.stratified
            self.mlp.eval()
            
            # self.extractor.add_debug()

            for count, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
            
                proj, fea = self.extractor(data)

                proj = F.normalize(proj, dim=1)
                SimCLR_loss,_,_ = self.SimCLRLoss(proj)
                SupCon_loss, exp_logits, mask = self.SupCon(proj.reshape(proj.shape[0],1,-1),labels)

                # 做me平均操作+stratified norm后接mlp,计算分类损失
                if self.args.extract_mode == 'me':
                    fea = torch.mean(fea, axis=3).reshape(fea.shape[0],fea.shape[1],fea.shape[2],1)
                elif self.args.extract_mode == 'de': 
                    fea = 0.5*torch.log(2*math.pi*math.e*(torch.var(fea, 3))).reshape(fea.shape[0],fea.shape[1],fea.shape[2],1)


                fea = stratified_layerNorm(fea,int(fea.shape[0]/2)).reshape(fea.shape[0],-1)

                logits = self.mlp(fea)
                SupCls_loss = self.CEL(logits, labels)

                loss = SupCon_loss + Lambda1*SimCLR_loss + Lambda2*SupCls_loss

                self.opt_ext.zero_grad()
                loss.backward()
                self.opt_ext.step()
            
            self.extractor.eval()
            acc_ave,acc_allright = 0,0
            for count, (data,labels) in enumerate(val_loader):
                with torch.no_grad():
                    data = data.to(self.device)
                    labels = labels.to(self.device)

                    proj, fea = self.extractor(data)

                    proj = F.normalize(proj, dim=1)
                    # SimCLR_loss = self.SimCLRLoss(proj)
                    SupCon_loss, exp_logits, mask = self.SupCon(proj.reshape(proj.shape[0],1,-1),labels)

                    t_acc_ave, t_acc_allright = self.accuracy(exp_logits,mask)
                    acc_ave += t_acc_ave
                    acc_allright += t_acc_allright
            
            acc_ave /= (count+1)
            acc_allright /= (count+1)

            print('acc_ave:',acc_ave,'acc_allright:',acc_allright)
            end_time = time.time()
            print('extract training epoch time consumed:',end_time - start_time)

        checkpoint_name = 'ext_cp_r{:02d}.pth.tar'.format(current_round)
        save_checkpoint(self.extractor,self.opt_ext,save_path=os.path.join(self.log_dir, checkpoint_name))


    def extractor_train_fn_normal(self,train_loader,val_loader, epochs):
        best_acc = -1
        print('extractor training......')
        for epoch in range(epochs):
            start_time = time.time()


            self.extractor.train()
            self.extractor.set_stratified = self.args.stratified
            
            # self.extractor.add_debug()
            train_loss = 0
            train_acc = 0
            train_acc5 = 0
            for count, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
            
                proj, fea = self.extractor(data)

                proj = F.normalize(proj, dim=1)
                loss, logits, labels = self.SimCLRLoss(proj)
                top1, top5 = accuracy(logits, labels, topk=(1,5))



                self.opt_ext.zero_grad()
                loss.backward()
                self.opt_ext.step()

                train_loss = train_loss + loss.data.cpu().numpy()
                train_acc = train_acc + top1[0]
                train_acc5 = train_acc5 + top5[0]
            
            train_loss = train_loss / (count + 1)
            train_acc = train_acc / (count + 1)
            train_acc5 = train_acc5 / (count + 1)
            
            self.extractor.eval()
            val_loss = 0
            val_acc = 0
            val_acc5 = 0
            for count, (data,labels) in enumerate(val_loader):
                with torch.no_grad():
                    data = data.to(self.device)
                    labels = labels.to(self.device)

                    proj, fea = self.extractor(data)

                    proj = F.normalize(proj, dim=1)
                    loss, logits, labels = self.SimCLRLoss(proj)

                    top1, top5 = accuracy(logits, labels, topk=(1,5))

                    val_loss = val_loss + loss.data.cpu().numpy()
                    val_acc = val_acc + top1[0]
                    val_acc5 = val_acc5 + top5[0]

            val_loss = val_loss / (count + 1)
            val_acc = val_acc / (count + 1)
            val_acc5 = val_acc5 / (count + 1)

            print(f"Epoch: {epoch}   Train loss: {train_loss}   Top1 accuracy: {train_acc}   Top5 accuracy: {train_acc5}")
            print(f"\tVal loss: {val_loss}   Top1 accuracy: {val_acc}   Top5 accuracy: {val_acc5}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_train_loss = train_loss
                best_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'best_ext_cp.pth.tar'
                save_checkpoint(self.extractor,self.opt_ext,save_path=os.path.join(self.log_dir, checkpoint_name))
            else:
                bad_count += 1


            end_time = time.time()
            print('extract training epoch time consumed:',end_time - start_time)
            
            if bad_count > self.args.max_tol_normal_ext:
                break
        # checkpoint_name = 'ext_cp_r{:02d}.pth.tar'.format(current_round)
        # save_checkpoint(self.extractor,self.opt_ext,save_path=os.path.join(self.log_dir, checkpoint_name))
        # print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc}   Train loss: {best_train_loss}    Val accuracy: {best_acc}   Val loss: {best_loss}")

    def extractor_train_fn_normal_find_ave_best_epoch(self,train_loader,val_loader, epochs):
        best_acc = -1
        print('extractor training......')
        val_acc_list =  np.zeros(epochs)
        for epoch in range(epochs):
            start_time = time.time()


            self.extractor.train()
            self.extractor.set_stratified = self.args.stratified
            
            # self.extractor.add_debug()
            train_loss = 0
            train_acc = 0
            train_acc5 = 0
            for count, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
            
                proj, fea = self.extractor(data)

                proj = F.normalize(proj, dim=1)
                loss, logits, labels = self.SimCLRLoss(proj)
                top1, top5 = accuracy(logits, labels, topk=(1,5))



                self.opt_ext.zero_grad()
                loss.backward()
                self.opt_ext.step()

                train_loss = train_loss + loss.data.cpu().numpy()
                train_acc = train_acc + top1[0]
                train_acc5 = train_acc5 + top5[0]
            
            train_loss = train_loss / (count + 1)
            train_acc = train_acc / (count + 1)
            train_acc5 = train_acc5 / (count + 1)
            
            self.extractor.eval()
            val_loss = 0
            val_acc = 0
            val_acc5 = 0
            for count, (data,labels) in enumerate(val_loader):
                with torch.no_grad():
                    data = data.to(self.device)
                    labels = labels.to(self.device)

                    proj, fea = self.extractor(data)

                    proj = F.normalize(proj, dim=1)
                    loss, logits, labels = self.SimCLRLoss(proj)

                    top1, top5 = accuracy(logits, labels, topk=(1,5))

                    val_loss = val_loss + loss.data.cpu().numpy()
                    val_acc = val_acc + top1[0]
                    val_acc5 = val_acc5 + top5[0]

            val_loss = val_loss / (count + 1)
            val_acc = val_acc / (count + 1)
            val_acc5 = val_acc5 / (count + 1)
            val_acc_list[epoch] = val_acc

            print(f"Epoch: {epoch}   Train loss: {train_loss}   Top1 accuracy: {train_acc}   Top5 accuracy: {train_acc5}")
            print(f"\tVal loss: {val_loss}   Top1 accuracy: {val_acc}   Top5 accuracy: {val_acc5}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_train_loss = train_loss
                best_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'best_ext_cp.pth.tar'
                save_checkpoint(self.extractor,self.opt_ext,save_path=os.path.join(self.log_dir, checkpoint_name))
            else:
                bad_count += 1


            end_time = time.time()
            print('extract training epoch time consumed:',end_time - start_time)
            
            if bad_count > self.args.max_tol_normal_ext:
                break
        # checkpoint_name = 'ext_cp_r{:02d}.pth.tar'.format(current_round)
        # save_checkpoint(self.extractor,self.opt_ext,save_path=os.path.join(self.log_dir, checkpoint_name))
        print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        return val_acc_list


    def mlp_train_fn(self, train_loader2, val_loader2, epochs, current_round):
        
        # 对初始data做train norm

        # 对fea做norm和lds  生成数据迭代器
        self.extractor.eval()
        self.extractor.set_saveFea(True)
        self.extractor.set_stratified = []
        pfea_train, label_train, pfea_val, label_val = self.ext_fea_process(train_loader2, val_loader2, self.args.extract_mode, self.args.norm_decay_rate)

        pdataset_train = PDataset(pfea_train,label_train)
        ploader_train = DataLoader(dataset=pdataset_train, batch_size=self.args.mlp_train_bs, shuffle=True, num_workers=8)
        pdataset_val = PDataset(pfea_val,label_val)
        ploader_val = DataLoader(dataset=pdataset_val, batch_size=self.args.mlp_val_bs, shuffle=False, num_workers=8)

        self.extractor.set_saveFea(False)
        self.extractor.set_stratified = self.args.stratified
        
        best_acc = -1

        for epoch in range(epochs):
            start_time = time.time()
            print(f'round: {current_round} mlp training epoch {epoch}')

            train_acc = 0
            train_loss = 0
            self.mlp.train()
            for counter, (data, labels) in enumerate(ploader_train):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                self.opt_mlp.zero_grad()
                loss.backward()
                self.opt_mlp.step()
                
                top1 = accuracy(logits, labels, topk=(1,))
                train_acc += top1[0]
                train_loss += loss.data.cpu().numpy()

            train_acc /= (counter + 1)
            train_loss /= (counter + 1)
            
            val_acc = 0
            val_loss = 0
            self.mlp.eval()
            for counter, (data, labels) in enumerate(ploader_val):
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                top1 = accuracy(logits, labels, topk=(1,))
                val_acc += top1[0]
                val_loss += loss.data.cpu().numpy()
            val_acc /= (counter + 1)
            val_loss /= (counter + 1)

            # print(f"Round {current_round}  Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            print(f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")

            if val_acc > best_acc:
                best_loss = val_loss
                best_acc = val_acc
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'mlp_cp_r{:02d}_e{:02d}.pth.tar'.format(current_round,best_epoch)
                save_checkpoint(self.mlp,self.opt_mlp,save_path=os.path.join(self.log_dir, checkpoint_name))
            
            end_time = time.time()
            print('mlp training epoch time consumed:',end_time - start_time)
        # if epoch != 0:
        print(f"Round {current_round}   Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        return  best_acc, best_loss, best_train_acc, best_train_loss, best_epoch

    def mlp_train_fn_normal(self, train_loader2, val_loader2, epochs):
        
        # 对初始data做train norm

        # 对fea做norm和lds  生成数据迭代器
        self.extractor.eval()
        self.extractor.set_saveFea(True)
        self.extractor.set_stratified = []
        pfea_train, label_train, pfea_val, label_val = self.ext_fea_process(train_loader2, val_loader2, self.args.extract_mode, self.args.norm_decay_rate)

        pdataset_train = PDataset(pfea_train,label_train)
        ploader_train = DataLoader(dataset=pdataset_train, batch_size=self.args.mlp_train_bs, shuffle=True, num_workers=8)
        pdataset_val = PDataset(pfea_val,label_val)
        ploader_val = DataLoader(dataset=pdataset_val, batch_size=self.args.mlp_val_bs, shuffle=False, num_workers=8)

        self.extractor.set_saveFea(False)
        self.extractor.set_stratified = self.args.stratified
        
        best_acc = -1
        print('mlp training......')
        for epoch in range(epochs):
            start_time = time.time()


            train_acc = 0
            train_loss = 0
            self.mlp.train()
            for counter, (data, labels) in enumerate(ploader_train):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                self.opt_mlp.zero_grad()
                loss.backward()
                self.opt_mlp.step()
                
                top1 = accuracy(logits, labels, topk=(1,))
                train_acc += top1[0]
                train_loss += loss.data.cpu().numpy()

            train_acc /= (counter + 1)
            train_loss /= (counter + 1)
            
            val_acc = 0
            val_loss = 0
            self.mlp.eval()
            for counter, (data, labels) in enumerate(ploader_val):
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                top1 = accuracy(logits, labels, topk=(1,))
                val_acc += top1[0]
                val_loss += loss.data.cpu().numpy()
            val_acc /= (counter + 1)
            val_loss /= (counter + 1)

            # print(f"Round {current_round}  Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            # print(f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            print(f"Epoch {epoch}    Train accuracy {train_acc}    Val accuracy: {val_acc}    Train loss {train_loss}    Val loss: {val_loss}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'best_mlp_cp.pth.tar'
                save_checkpoint(self.mlp,self.opt_mlp,save_path=os.path.join(self.log_dir, checkpoint_name))
            
            else:
                bad_count += 1

            end_time = time.time()
            print('mlp training epoch time consumed:',end_time - start_time)

            if bad_count > self.args.max_tol_normal_mlp:
                break

            
        # print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc}   Train loss: {best_train_loss}    Val accuracy: {best_acc}   Val loss: {best_loss}")
        return  best_acc, best_loss, best_train_acc, best_train_loss

    def mlp_train_fn_normal_SEEDV(self, train_loader2, val_loader2, epochs):
        
        # 对初始data做train norm

        # 对fea做norm和lds  生成数据迭代器
        self.extractor.eval()
        self.extractor.set_saveFea(True)
        self.extractor.set_stratified = []
        pfea_train, label_train, pfea_val, label_val = self.ext_fea_process_SEEDV(train_loader2, val_loader2, self.args.trainset_len2, self.args.valset_len2, self.args.extract_mode, self.args.norm_decay_rate)

        pdataset_train = PDataset(pfea_train,label_train)
        ploader_train = DataLoader(dataset=pdataset_train, batch_size=self.args.mlp_train_bs, shuffle=True, num_workers=8)
        pdataset_val = PDataset(pfea_val,label_val)
        ploader_val = DataLoader(dataset=pdataset_val, batch_size=self.args.mlp_val_bs, shuffle=False, num_workers=8)

        self.extractor.set_saveFea(False)
        self.extractor.set_stratified = self.args.stratified
        
        best_acc = -1
        print('mlp training......')
        for epoch in range(epochs):
            start_time = time.time()


            train_acc = 0
            train_loss = 0
            self.mlp.train()
            for counter, (data, labels) in enumerate(ploader_train):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                self.opt_mlp.zero_grad()
                loss.backward()
                self.opt_mlp.step()
                
                top1 = accuracy(logits, labels, topk=(1,))
                train_acc += top1[0]
                train_loss += loss.data.cpu().numpy()

            train_acc /= (counter + 1)
            train_loss /= (counter + 1)
            
            val_acc = 0
            val_loss = 0
            self.mlp.eval()
            for counter, (data, labels) in enumerate(ploader_val):
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                top1 = accuracy(logits, labels, topk=(1,))
                val_acc += top1[0]
                val_loss += loss.data.cpu().numpy()
            val_acc /= (counter + 1)
            val_loss /= (counter + 1)

            # print(f"Round {current_round}  Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            # print(f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            print(f"Epoch {epoch}    Train accuracy {train_acc}    Val accuracy: {val_acc}    Train loss {train_loss}    Val loss: {val_loss}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'best_mlp_cp.pth.tar'
                save_checkpoint(self.mlp,self.opt_mlp,save_path=os.path.join(self.log_dir, checkpoint_name))
            
            else:
                bad_count += 1

            end_time = time.time()
            print('mlp training epoch time consumed:',end_time - start_time)

            if bad_count > self.args.max_tol_normal_mlp:
                break

            
        # print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc}   Train loss: {best_train_loss}    Val accuracy: {best_acc}   Val loss: {best_loss}")
        return  best_acc, best_loss, best_train_acc, best_train_loss



    def mlp_train_fn_normal_find_ave_best_epoch(self, train_loader2, val_loader2, epochs):
        
        # 对初始data做train norm

        # 对fea做norm和lds  生成数据迭代器
        self.extractor.eval()
        self.extractor.set_saveFea(True)
        self.extractor.set_stratified = []
        pfea_train, label_train, pfea_val, label_val = self.ext_fea_process(train_loader2, val_loader2, self.args.extract_mode, self.args.norm_decay_rate)

        pdataset_train = PDataset(pfea_train,label_train)
        ploader_train = DataLoader(dataset=pdataset_train, batch_size=self.args.mlp_train_bs, shuffle=True, num_workers=8)
        pdataset_val = PDataset(pfea_val,label_val)
        ploader_val = DataLoader(dataset=pdataset_val, batch_size=self.args.mlp_val_bs, shuffle=False, num_workers=8)

        self.extractor.set_saveFea(False)
        self.extractor.set_stratified = self.args.stratified
        
        best_acc = -1
        val_acc_list =  np.zeros(epochs)
        print('mlp training......')
        for epoch in range(epochs):
            start_time = time.time()


            train_acc = 0
            train_loss = 0
            self.mlp.train()
            for counter, (data, labels) in enumerate(ploader_train):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                self.opt_mlp.zero_grad()
                loss.backward()
                self.opt_mlp.step()
                
                top1 = accuracy(logits, labels, topk=(1,))
                train_acc += top1[0]
                train_loss += loss.data.cpu().numpy()

            train_acc /= (counter + 1)
            train_loss /= (counter + 1)
            
            val_acc = 0
            val_loss = 0
            self.mlp.eval()
            for counter, (data, labels) in enumerate(ploader_val):
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = self.mlp(data)
                loss = self.CEL(logits, labels)

                top1 = accuracy(logits, labels, topk=(1,))
                val_acc += top1[0]
                val_loss += loss.data.cpu().numpy()
            val_acc /= (counter + 1)
            val_loss /= (counter + 1)
            val_acc_list[epoch] = val_acc

            # print(f"Round {current_round}  Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")
            print(f"Epoch {epoch}    Train accuracy {train_acc.item()}    Val accuracy: {val_acc.item()}    Train loss {train_loss}    Val loss: {val_loss}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_epoch = epoch
                checkpoint_name = 'best_mlp_cp.pth.tar'
                save_checkpoint(self.mlp,self.opt_mlp,save_path=os.path.join(self.log_dir, checkpoint_name))
            
            else:
                bad_count += 1

            end_time = time.time()
            print('mlp training epoch time consumed:',end_time - start_time)

            if bad_count > self.args.max_tol_normal_mlp:
                break

            
        print(f"Best Epoch {best_epoch}   Train accuracy: {best_train_acc.item()}   Train loss: {best_train_loss}    Val accuracy: {best_acc.item()}   Val loss: {best_loss}")
        return  best_acc, best_loss, best_train_acc, best_train_loss, val_acc_list

    def ext_fea_process(self, train_loader2, val_loader2, extract_mode='me',decay_rate=0.990):
        # 处理训练集
        for counter, (x_batch, y_batch) in enumerate(train_loader2):
            train_bs = x_batch.shape[0]
            x_batch = x_batch.to(self.device)
            train_label = y_batch
            fea = self.extractor(x_batch)
            fea = fea.detach().cpu().numpy()

            # print('test:',extract_mode)

            if extract_mode == 'me':
                # print('ext_mode:me')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                mean = np.mean(fea, axis=3).reshape(train_bs,-1)
                # print(mean.shape)
                if counter == 0:
                    fea_ = mean
                else:
                    fea_ = np.concatenate((fea_,mean),0)
               
            elif extract_mode == 'de':
                # print('ext_mode:de')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3))+1e-3)
                # print(fea)
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(fea, 3))).reshape(train_bs,-1)
                if counter == 0:
                    fea_ = de
                else:
                    fea_ = np.concatenate((fea_,de),0)

        # 数据修正
        # fea_[np.isnan(fea_)] = -30
        if(np.isnan(fea_).any()):
            print('train_fea_ exsits nan!')
            print(fea_)
            if np.any(~np.isnan(fea_)):
                print(np.min(fea_[~np.isnan(fea_)]))
                fea_[np.isnan(fea_)] = np.min(fea_[~np.isnan(fea_)])

        data_mean = np.mean(fea_, axis=0)
        data_var = np.var(fea_, axis=0)

        trian_fea_ = fea_.reshape(-1,train_bs,fea_.shape[-1])

        # reorder before running norm
        vid_order = video_order_load(28)
        train_vid_order = vid_order[self.args.train_sub]
        train_fea_reorder, vid_play_order_new = reorder_vids_sepVideo(trian_fea_, train_vid_order, self.args.train_vid_inds, self.args.n_vids)

        train_fea_processed_reorder = running_norm(train_fea_reorder,data_mean,data_var,decay_rate)

        # order back
        train_fea_processed = reorder_vids_back(train_fea_processed_reorder, self.args.n_vids_train, vid_play_order_new)
        train_fea_processed = train_fea_processed.reshape(-1,self.args.n_samples2,train_fea_processed.shape[-1])
        # LDS
        for vid in range(train_fea_processed.shape[0]):
            train_fea_processed[vid] = LDS(train_fea_processed[vid])
        train_fea_processed = train_fea_processed.reshape(-1,train_fea_processed.shape[-1])

        # 数据修正
        if(np.isnan(train_fea_processed).any()):
            print('train_fea_processed exsits nan!')
            print(train_fea_processed)
            if np.any(~np.isnan(train_fea_processed)):
                print(np.min(train_fea_processed[~np.isnan(train_fea_processed)]))
                train_fea_processed[np.isnan(train_fea_processed)] = np.min(train_fea_processed[~np.isnan(train_fea_processed)])

        train_label = np.tile(train_label, len(self.args.train_sub))

        if (len(val_loader2)==0)or(val_loader2==None):
            return train_fea_processed, train_label, np.array([]), np.array([])
        # 处理测试集
        for counter, (x_batch, y_batch) in enumerate(val_loader2):
            val_bs = x_batch.shape[0]
            x_batch = x_batch.to(self.device)
            val_label = y_batch
            fea = self.extractor(x_batch)
            fea = fea.detach().cpu().numpy()

            # print('test:',extract_mode)

            if extract_mode == 'me':
                # print('ext_mode:me')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                mean = np.mean(fea, axis=3).reshape(val_bs,-1)
                # print(mean.shape)
                if counter == 0:
                    fea_ = mean
                else:
                    fea_ = np.concatenate((fea_,mean),0)
               
            elif extract_mode == 'de':
                # print('ext_mode:de')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3))+1e-3)
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(fea, 3))).reshape(val_bs,-1)
                if counter == 0:
                    fea_ = de
                else:
                    fea_ = np.concatenate((fea_,de),0)

        # 数据修正
        # fea_[np.isnan(fea_)] = -30
        if(np.isnan(fea_).any()):
            print('val_fea_ exsits nan!')
            print(fea_)
            if np.any(~np.isnan(fea_)):
                print(np.min(fea_[~np.isnan(fea_)]))
                fea_[np.isnan(fea_)] = np.min(fea_[~np.isnan(fea_)])
        
        val_fea_ = fea_.reshape(-1,val_bs,fea_.shape[-1])

        # reorder before running norm
        # vid_order = video_order_load(self.args.n_vids)
        val_vid_order = vid_order[self.args.val_sub]
        val_fea_reorder, vid_play_order_new = reorder_vids_sepVideo(val_fea_, val_vid_order, self.args.val_vid_inds, self.args.n_vids)

        val_fea_processed_reorder = running_norm(val_fea_reorder,data_mean,data_var,decay_rate)

        # order back
        val_fea_processed = reorder_vids_back(val_fea_processed_reorder, self.args.n_vids_val, vid_play_order_new)
        val_fea_processed = val_fea_processed.reshape(-1,self.args.n_samples2,val_fea_processed.shape[-1])
        # LDS
        for vid in range(val_fea_processed.shape[0]):
            val_fea_processed[vid] = LDS(val_fea_processed[vid])
        val_fea_processed = val_fea_processed.reshape(-1,val_fea_processed.shape[-1])


        # 数据修正
        if(np.isnan(val_fea_processed).any()):
            print('val_fea_processed exsits nan!')
            print(val_fea_processed)
            if np.any(~np.isnan(val_fea_processed)):
                print(np.min(val_fea_processed[~np.isnan(val_fea_processed)]))
                val_fea_processed[np.isnan(val_fea_processed)] = np.min(val_fea_processed[~np.isnan(val_fea_processed)])

        val_label = np.tile(val_label, len(self.args.val_sub))

        
        return train_fea_processed, train_label, val_fea_processed, val_label


    def ext_fea_process_independent(self, train_loader2, train_sub, train_vid_inds, n_samples2, n_vids, extract_mode='me',decay_rate=0.990):
        # 处理训练集
        for counter, (x_batch, y_batch) in enumerate(train_loader2):
            train_bs = x_batch.shape[0]
            x_batch = x_batch.to(self.device)
            train_label = y_batch
            fea = self.extractor(x_batch)
            fea = fea.detach().cpu().numpy()

            # print('test:',extract_mode)

            if extract_mode == 'me':
                # print('ext_mode:me')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                mean = np.mean(fea, axis=3).reshape(train_bs,-1)
                # print(mean.shape)
                if counter == 0:
                    fea_ = mean
                else:
                    fea_ = np.concatenate((fea_,mean),0)
               
            elif extract_mode == 'de':
                # print('ext_mode:de')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3))+1e-3)
                # print(fea)
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(fea, 3))).reshape(train_bs,-1)
                if counter == 0:
                    fea_ = de
                else:
                    fea_ = np.concatenate((fea_,de),0)

        # 数据修正
        # fea_[np.isnan(fea_)] = -30
        if(np.isnan(fea_).any()):
            print('train_fea_ exsits nan!')
            print(fea_)
            if np.any(~np.isnan(fea_)):
                print(np.min(fea_[~np.isnan(fea_)]))
                fea_[np.isnan(fea_)] = np.min(fea_[~np.isnan(fea_)])

        data_mean = np.mean(fea_, axis=0)
        data_var = np.var(fea_, axis=0)

        trian_fea_ = fea_.reshape(-1,train_bs,fea_.shape[-1])

        # reorder before running norm
        vid_order = video_order_load(28)
        train_vid_order = vid_order[train_sub]
        train_fea_reorder, vid_play_order_new = reorder_vids_sepVideo(trian_fea_, train_vid_order, train_vid_inds, n_vids)

        train_fea_processed_reorder = running_norm(train_fea_reorder,data_mean,data_var,decay_rate)

        # order back
        n_vids_train = len(train_vid_inds)
        train_fea_processed = reorder_vids_back(train_fea_processed_reorder, n_vids_train, vid_play_order_new)
        train_fea_processed = train_fea_processed.reshape(-1,n_samples2,train_fea_processed.shape[-1])
        # LDS
        for vid in range(train_fea_processed.shape[0]):
            train_fea_processed[vid] = LDS(train_fea_processed[vid])
        train_fea_processed = train_fea_processed.reshape(-1,train_fea_processed.shape[-1])

        # 数据修正
        if(np.isnan(train_fea_processed).any()):
            print('train_fea_processed exsits nan!')
            print(train_fea_processed)
            if np.any(~np.isnan(train_fea_processed)):
                print(np.min(train_fea_processed[~np.isnan(train_fea_processed)]))
                train_fea_processed[np.isnan(train_fea_processed)] = np.min(train_fea_processed[~np.isnan(train_fea_processed)])

        train_label = np.tile(train_label, len(train_sub))

        return train_fea_processed, train_label


    def ext_fea_process_SEEDV(self, train_loader2, val_loader2, trainset_len, valset_len, extract_mode='me',decay_rate=0.990):
        # 处理训练集
        fea_ = np.empty((trainset_len,self.args.fea_dim),float)
        train_label_all = np.empty((trainset_len,),int)
        pos = 0
        # print(trainset_len)
        for counter, (x_batch, y_batch) in enumerate(train_loader2):
            train_bs = x_batch.shape[0]
            x_batch = x_batch.to(self.device)
            train_label = y_batch
            fea = self.extractor(x_batch)
            fea = fea.detach().cpu().numpy()

            # print('test:',extract_mode)

            if extract_mode == 'me':
                # print('ext_mode:me')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                mean = np.mean(fea, axis=3).reshape(train_bs,-1)
                # print(mean.shape)
                # if counter == 0:
                #     fea_ = mean
                # else:
                #     fea_ = np.concatenate((fea_,mean),0)
                fea_[pos:pos+train_bs] = mean
            
               
            elif extract_mode == 'de':
                # print('ext_mode:de')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3))+1e-3)
                # print(fea)
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(fea, 3))).reshape(train_bs,-1)
                # if counter == 0:
                #     fea_ = de
                # else:
                #     fea_ = np.concatenate((fea_,de),0)
                fea_[pos:pos+train_bs] = de
            train_label_all[pos:pos+train_bs] = train_label
            pos += train_bs
        # print(pos)
        # 数据修正
        # fea_[np.isnan(fea_)] = -30
        if(np.isnan(fea_).any()):
            print('train_fea_ exsits nan!')
            print(fea_)
            if np.any(~np.isnan(fea_)):
                print(np.min(fea_[~np.isnan(fea_)]))
                fea_[np.isnan(fea_)] = np.min(fea_[~np.isnan(fea_)])

        data_mean = np.mean(fea_, axis=0)
        data_var = np.var(fea_, axis=0)

        trian_fea_ = fea_.reshape(len(self.args.train_sub),-1,fea_.shape[-1])
        n_sample_sum_sessions = np.sum(self.args.n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))

        train_fea_processed = np.zeros_like(trian_fea_)
        for sub in range(len(self.args.train_sub)):
            for s in  range(len(n_sample_sum_sessions)):
                train_fea_processed[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesub(
                        trian_fea_[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],data_mean,data_var,decay_rate)

        # train_fea_processed.shape = (n_train_sub,all_samples(45 vid),fea_dim)

        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(self.args.n_samples2_onesub)))
        # LDS
        for sub in range(len(self.args.train_sub)):
            for vid in range(len(self.args.n_samples2_onesub)):
                train_fea_processed[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(train_fea_processed[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]])
        train_fea_processed = train_fea_processed.reshape(-1,train_fea_processed.shape[-1])

        # 数据修正
        if(np.isnan(train_fea_processed).any()):
            print('train_fea_processed exsits nan!')
            print(train_fea_processed)
            if np.any(~np.isnan(train_fea_processed)):
                print(np.min(train_fea_processed[~np.isnan(train_fea_processed)]))
                train_fea_processed[np.isnan(train_fea_processed)] = np.min(train_fea_processed[~np.isnan(train_fea_processed)])

        train_label = np.array(train_label_all)

        if (len(val_loader2)==0)or(val_loader2==None):
            return train_fea_processed, train_label, np.array([]), np.array([])
        # 处理测试集
        pos = 0
        fea_ = np.empty((valset_len,self.args.fea_dim),float)
        val_label_all = np.empty((valset_len,),int)
        # print(valset_len)
        for counter, (x_batch, y_batch) in enumerate(val_loader2):
            val_bs = x_batch.shape[0]
            x_batch = x_batch.to(self.device)
            val_label = y_batch
            fea = self.extractor(x_batch)
            fea = fea.detach().cpu().numpy()

            # print('test:',extract_mode)

            if extract_mode == 'me':
                # print('ext_mode:me')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                mean = np.mean(fea, axis=3).reshape(val_bs,-1)
                # print(mean.shape)
                # if counter == 0:
                #     fea_ = mean
                # else:
                #     fea_ = np.concatenate((fea_,mean),0)
                fea_[pos:pos+val_bs] = mean
                
               
            elif extract_mode == 'de':
                # print('ext_mode:de')
                # de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3))+1e-3)
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(fea, 3))).reshape(val_bs,-1)
                # if counter == 0:
                #     fea_ = de
                # else:
                #     fea_ = np.concatenate((fea_,de),0)
                fea_[pos:pos+val_bs] = de
            
            val_label_all[pos:pos+val_bs] = val_label
            pos += val_bs
        # print(pos)
        # 数据修正
        # fea_[np.isnan(fea_)] = -30
        if(np.isnan(fea_).any()):
            print('val_fea_ exsits nan!')
            print(fea_)
            if np.any(~np.isnan(fea_)):
                print(np.min(fea_[~np.isnan(fea_)]))
                fea_[np.isnan(fea_)] = np.min(fea_[~np.isnan(fea_)])


        val_fea_ = fea_.reshape(len(self.args.val_sub),-1,fea_.shape[-1])





        val_fea_processed = np.zeros_like(val_fea_)
        for sub in range(len(self.args.val_sub)):
            for s in  range(len(n_sample_sum_sessions)):
                val_fea_processed[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesub(
                        val_fea_[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],data_mean,data_var,decay_rate)
    
        # val_fea_processed.shape = (n_val_sub,all_samples(45 vid),fea_dim)

        # LDS
        for sub in range(len(self.args.val_sub)):
            for vid in range(len(self.args.n_samples2_onesub)):
                val_fea_processed[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(val_fea_processed[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]])
        val_fea_processed = val_fea_processed.reshape(-1,val_fea_processed.shape[-1])

        # 数据修正
        if(np.isnan(val_fea_processed).any()):
            print('val_fea_processed exsits nan!')
            print(val_fea_processed)
            if np.any(~np.isnan(val_fea_processed)):
                print(np.min(val_fea_processed[~np.isnan(val_fea_processed)]))
                val_fea_processed[np.isnan(val_fea_processed)] = np.min(val_fea_processed[~np.isnan(val_fea_processed)])

        val_label = np.array(val_label_all)

        
        return train_fea_processed, train_label, val_fea_processed, val_label



