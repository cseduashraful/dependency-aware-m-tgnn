import torch
import time
from sklearn.metrics import average_precision_score, roc_auc_score

def _get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def train_new(model, dataloader, sampler, criterion, optimizer, args, dataloader2):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    data_loader_stime = time.time()
    dl_tt = 0
    tr_tt = 0
    for blocks in dataloader2:
        print(blocks)
        print(blocks[0].ndata)
        print(blocks[0].edges())
        # data_loader_etime = time.time()
        # dt = time.time()-data_loader_stime
        # # print("Data loader Batch: ", batch_cnt, "Time: ", dt, " dl_total: ", dl_tt, " tr_total: ", tr_tt)
        # dl_tt = dl_tt + dt
        # optimizer.zero_grad()
        # # print("Combined: ............................................................................................")
        # # model.printEdges(negative_pair_g)
        # pred_pos, pred_neg = model.embed(
        #     blocks, blocks, blocks)
        # loss = criterion(pred_pos, torch.ones_like(pred_pos))
        # print("Loss:", float(loss))
        # loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        # total_loss += float(loss)*args.batch_size
        
        # retain_graph = True if batch_cnt == 0 else False
        # loss.backward(retain_graph=retain_graph)
        # optimizer.step()

        # model.detach_memory()
        # # if not args.not_use_memory:
        # #     model.update_memory(positive_pair_g)


        # # if args.fast_mode:
        # #     sampler.attach_last_update(model.memory.last_update_t)
        # dt = time.time()-last_t
        # # print("Batch: ", batch_cnt, "Time: ", dt)
        # tr_tt  = tr_tt + dt
        # last_t = time.time()
        # batch_cnt += 1
        # data_loader_stime = time.time()
    return total_loss, dl_tt, tr_tt






def trainvsajfsdjfsjkdf2(model, dataloader, sampler, criterion, optimizer, args, dataloader2):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    data_loader_stime = time.time()
    dl_tt = 0
    tr_tt = 0

    for _, blocks in dataloader2:
        print(blocks)

    input("input something to old dataloader")
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        # data_loader_etime = time.time()
        dt = time.time()-data_loader_stime
        # print("Data loader Batch: ", batch_cnt, "Time: ", dt, " dl_total: ", dl_tt, " tr_total: ", tr_tt)
        dl_tt = dl_tt + dt
        optimizer.zero_grad()
        # print("Combined: ............................................................................................")
        # model.printEdges(negative_pair_g)
        pred_pos, pred_neg = model.embed(
            positive_pair_g, negative_pair_g, blocks)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        print("Loss:", float(loss))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss)*args.batch_size
        
        retain_graph = True if batch_cnt == 0 else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        model.detach_memory()
        # if not args.not_use_memory:
        #     model.update_memory(positive_pair_g)


        # if args.fast_mode:
        #     sampler.attach_last_update(model.memory.last_update_t)
        dt = time.time()-last_t
        # print("Batch: ", batch_cnt, "Time: ", dt)
        tr_tt  = tr_tt + dt
        last_t = time.time()
        batch_cnt += 1
        data_loader_stime = time.time()
    return total_loss, dl_tt, tr_tt


# def updateMemory(model, positive_pair_g):
#     model.detach_memory()
#     model.update_memory(positive_pair_g)

def train(model, dataloader, sampler, criterion, optimizer, args):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    data_loader_stime = time.time()
    dl_tt = 0
    tr_tt = 0
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        # data_loader_etime = time.time()
        dt = time.time()-data_loader_stime
        # print("Data loader Batch: ", batch_cnt, "Time: ", dt, " dl_total: ", dl_tt, " tr_total: ", tr_tt)
        dl_tt = dl_tt + dt
        optimizer.zero_grad()
        # print("Combined: ............................................................................................")
        # model.printEdges(negative_pair_g)
        pred_pos, pred_neg = model.embed(
            positive_pair_g, negative_pair_g, blocks)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        print("Loss:", float(loss))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss)*args.batch_size
        
        retain_graph = True if batch_cnt == 0 else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        model.detach_memory()
        # if not args.not_use_memory:
        #     model.update_memory(positive_pair_g)


        # if args.fast_mode:
        #     sampler.attach_last_update(model.memory.last_update_t)
        dt = time.time()-last_t
        # print("Batch: ", batch_cnt, "Time: ", dt)
        tr_tt  = tr_tt + dt
        last_t = time.time()
        batch_cnt += 1
        data_loader_stime = time.time()
    return total_loss, dl_tt, tr_tt


def test_val(model, dataloader, sampler, criterion, args):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for _, postive_pair_g, negative_pair_g, blocks in dataloader:
            pred_pos, pred_neg = model.embed(
                postive_pair_g, negative_pair_g, blocks)
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss)*batch_size
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            
            # if not args.not_use_memory:
            #     model.update_memory(postive_pair_g)
            
            # if args.fast_mode:
            #     sampler.attach_last_update(model.memory.last_update_t)
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


def train2(model, dataloader, sampler, criterion, optimizer, args):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    data_loader_stime = time.time()
    dl_tt = 0
    tr_tt = 0
    for blocks in dataloader:
        # data_loader_etime = time.time()
        dt = time.time()-data_loader_stime
        # print("Data loader Batch: ", batch_cnt, "Time: ", dt, " dl_total: ", dl_tt, " tr_total: ", tr_tt)
        dl_tt = dl_tt + dt
        optimizer.zero_grad()
        # print("Combined: ............................................................................................")
        # model.printEdges(negative_pair_g)
        pred_pos, pred_neg = model.embed(
            blocks, blocks, blocks)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        print("Loss:", float(loss))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss)*args.batch_size
        
        retain_graph = True if batch_cnt == 0 else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        model.detach_memory()
        # if not args.not_use_memory:
        #     model.update_memory(positive_pair_g)


        # if args.fast_mode:
        #     sampler.attach_last_update(model.memory.last_update_t)
        dt = time.time()-last_t
        # print("Batch: ", batch_cnt, "Time: ", dt)
        tr_tt  = tr_tt + dt
        last_t = time.time()
        batch_cnt += 1
        data_loader_stime = time.time()
    return total_loss, dl_tt, tr_tt


def test_val2(model, dataloader, sampler, criterion, args):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for blocks in dataloader:
            pred_pos, pred_neg = model.embed(
                blocks, blocks, blocks)
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss)*batch_size
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            
            # if not args.not_use_memory:
            #     model.update_memory(postive_pair_g)
            
            # if args.fast_mode:
            #     sampler.attach_last_update(model.memory.last_update_t)
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())



# import torch
from torch.profiler import profile, ProfilerActivity

def profileModel(model, dataloader):
    for blocks in dataloader:
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
            # Run your model
            output = model.embed(blocks, blocks, blocks)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        break