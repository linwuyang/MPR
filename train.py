import tqdm
import copy
import torch
import os
import pandas as pd
from helper import utils
from main_model.utils.mpr_perturber import Perturber
from helper.continual_learner import ContinualLearner
from data.loader import data_loader, data_dset
from torch.autograd import Variable
import torch.nn.functional as F
from helper.buffer import GlobalReplay
from helper.utils import l2_loss
from helper.ewc import ElasticWeightConsolidation
from helper import evaluate
import shutil





def train(args, model, train_loader, optimizer, epoch, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]

        ###################################################
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
        pred_traj_fake_rel = model(model_input, seq_start_end)
        l2_loss_rel = utils.l2_loss(
            pred_traj_fake_rel,
            model_input[-args.pred_len:],
            loss_mask,
            mode="average",
        )

        loss = l2_loss_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)

def train_cl(args, best_ade, model, train_datasets, val_datasets, test_datasets, replay_model="none", iters=2, batch_size=32,
             generator=None, fake_generator=None, gen_iters=0, gen_loss_cbs= list(), fake_gen_loss_cbs=list(), loss_cbs=list(), val_loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             metric_cbs=list()):
    '''
    Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]           <nn.Module> main model to optimize across all tasks
    [train_datasets]  <list> with for each task the training <DataSet>
    [replay_mode]     <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]        <str>, choice from "task", "domain" and "class"
    [iters]           <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]       None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]           <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None
    pre_model = None
    memory_batch = []

    # global x_rel_val, y_rel_val, seq_start_end_val

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    # Loop over all tasks.
    Buffer = GlobalReplay()  # 全局唯一缓存
    all_metrics = {
        'epoch': [],
        'ade': [],
        'fde': [],
    }
    for task, train_dataset in enumerate(train_datasets, 1):
        

        training_dataset = data_loader(args, train_dataset, args.batch_size)
        batch_num = len(training_dataset)

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None   # -> for Domain-IL scenario, always all classes are active

        # Reset state of optimizer(s) for every task (if requested) todo

        # Initialize # iters left on current data-loader(s)
        # iters_left = iters_left_previous = 1
        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters_to_use + 1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters + 1))
        # replay previous data for validation
        if fake_generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters * batch_num + 1))
        if args.val:
            x_rel_val = None
            y_rel_val = None
            seq_start_end_val = None

        ##-----REPLAYED BATCH------##
        if not Exact and not Generative and not Current:
            x_rel_ = y_rel_ = seq_start_end_ = x_rel_gr = y_rel_gr = seq_start_end_gr = None  # -> if no replay



        # run epoch
        current_step = 0
        for epoch in range(1, iters_to_use+1):
            losses_dict_main = {'loss_total':[], 'loss_current':[], 'loss_replay':[], 'pred_traj':[], 'pred_traj_r':[]}
            losses_dict_generative = {'loss_total':[], 'reconL':[], 'variatL':[], 'reconL_r':[], 'variatL_r':[]}
            
 
            for batch_index, batch in enumerate(training_dataset):
                current_step += 1
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    pred_traj_gt,
                    obs_traj_rel,
                    pred_traj_gt_rel,
                    non_linear_ped,
                    loss_mask,
                    seq_start_end,
                ) = batch

                

                #-------------Collect data----------------#
                ##------CURRENT BATCH-------##
                # y = y - class_per_task*(task-1) if scenario="task" else y    # --> ITL: adjust y-targets
                # print('\nout', len(out))
                if args.main_model == 'lstm':
                    model_input = obs_traj_rel
                else:
                    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
                x_rel = obs_traj_rel
                y_rel = pred_traj_gt_rel
                seq_start_end = seq_start_end
                loss_mask = loss_mask


                # x_rel, y_rel = out[2].to(device), out[3].to(device)                            # --> transfer them to correct device
                # seq_start_end = out[6].to(device)
                # loss_mask = out[5].to(device)

                # ------ Exact Replay ----- #
                if Exact:
                    x_rel_, y_rel_, logits_list, seq_start_end_ = Buffer.get_all()
                    #x_rel_, y_rel_ = Buffer.get_all()
                
                

                # ----> Train Main model
                if batch_index <= iters*batch_num:

                    # Train the main model with this batch
                    # Set model to training-mode
                    model.train()

                    # Reset optimizer
                    model.optimizer.zero_grad()
                    #--(1)-- REPLAYED DATA---#

                    y_hat_all = None
                    y_hat_rel = None
                    if x_rel_ is not None:
                        pert(x_rel_, y_rel_, seq_start_end_)
                        y_ = [y_rel_]
                        n_replays = len(y_) if (y_ is not None) else None

                        # Prepare lists to store losses for each replay
                        loss_replay = [None]*n_replays
                        pred_traj_r = [None]*n_replays
                        distill_r = [None]*n_replays

                        # Loop to evaluate predictions on replay according to each previous task
                        y_hat_all = model(x_rel_, seq_start_end_)
                        # y_hat_all = model(model_input, seq_start_end_)

                        for replay_id in range(n_replays):
                            y_hat = y_hat_all

                            # Calculate losses
                            if (y_rel_ is not None) and (y_[replay_id] is not None):
                                pred_traj_r[replay_id] = l2_loss(y_hat, y_[replay_id], mode="average")

                            # Weigh losses
                            loss_replay[replay_id] = pred_traj_r[replay_id]
                        loss_replay = sum(loss_replay) / n_replays
                        
                       
                        if replay_model == 'der':
                            with torch.no_grad():
                                y_old = logits_list.detach()
                            loss_distill = F.mse_loss(y_hat_all, y_old)
                            loss_replay = loss_distill
                    else:
                        loss_replay = None

                    #--(2)-- CURRENT DATA --#

                    if x_rel is not None:
                        
                        # Run model
                        y_hat_rel = model(model_input, seq_start_end)
                        
                        pred_traj = None if y_rel is None else l2_loss(y_hat_rel, y_rel, mode="average")
                        loss_cur = pred_traj

                    # Combine loss from current and replayed batch
                    rnt = 0.5
                    if x_rel_ is None:
                        loss_total = loss_cur
                    else:
                        # rnt = torch.rand(10, device=loss_cur.device)
                        # weighted_losses = rnt * loss_cur + (1 - rnt) * loss_replay
                        # loss_total = weighted_losses.mean()
                        loss_total = loss_replay if (x_rel is None) else rnt*loss_cur+(1-rnt)*loss_replay
                    
                    
                    

                    #--(3)-- ALLOCATION LOSSES --#

                    # Add SI-loss (Zenke et al., 2017)
                    surrogate_loss = model.surrogate_loss()
                    if model.si_c > 0:
                        loss_total += model.si_c * surrogate_loss

                    if replay_model != 'ewc':
                        if replay_model == 'lwf' and task > 1 and pre_model is not None:
                            y_old = pre_model(x_rel, seq_start_end)
                            y_hat_old = model(x_rel, seq_start_end)
                            loss_distill = F.mse_loss(y_hat_old, y_old)
                            loss_total = loss_total + 0.5 * loss_distill
                        # Backpropagate errors (if not yet done)
                        loss_total.backward()

                        # Take optimization-step
                        model.optimizer.step()

                    # if current_step % 20 == 0:
                    #     model.cbp.selective_reset()

                    # Returen the dictionary with different training-loss split in categories
                    loss_dict_main = {
                        'loss_total': loss_total.item(),
                        'loss_current':loss_cur.item() if x_rel is not None else 0,
                        'loss_replay': loss_replay.item() if (loss_replay is not None) and (x_rel is not None) else 0,
                        'pred_traj': pred_traj.item() if pred_traj is not None else 0,
                        'pred_traj_r': sum(pred_traj_r).item()/n_replays if (x_rel_ is not None and pred_traj_r[0] is not None) else 0,
                        'si_loss': surrogate_loss.item(),
                    }
                    #loss_dict_main = model.train_a_batch(x_rel, y_rel, seq_start_end, x_rel_=x_rel_, y_rel_=y_rel_, seq_start_end_=seq_start_end_, loss_mask=loss_mask, rnt=1./task)

                    main_loss_file = open("{}/loss_main_model_{}_{}_{}_{}.txt".format(args.r_dir, args.iters, args.batch_size, args.replay, args.val_class), 'a')
                    main_loss_file.write('{}: {}\n'.format(batch_index, loss_dict_main['loss_total']))
                    main_loss_file.close()
                    losses_dict_main['loss_total'].append(loss_dict_main['loss_total'])
                    losses_dict_main['loss_current'].append(loss_dict_main['loss_current'])
                    losses_dict_main['loss_replay'].append(loss_dict_main['loss_replay'])
                    losses_dict_main['pred_traj'].append(loss_dict_main['pred_traj'])
                    losses_dict_main['pred_traj_r'].append(loss_dict_main['pred_traj_r'])

                    # Update running parameter importance estimates in W
                    if isinstance(model, ContinualLearner) and (model.si_c > 0):
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    W[n].add_(-p.grad * (p.detach() - p_old[n]))
                                p_old[n] = p.detach().clone()

                # # -----> Train Fake Generator
                # if fake_generator is not None:
                #     # Train the generator with this batch
                #     loss_dict_fake_generative = fake_generator.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, rnt=1./task)
                if epoch == iters_to_use:
                    Buffer.add(batch, y_hat_rel)
                
                # -----> Train Generator
                if generator is not None and batch_index <= gen_iters*batch_num:

                    # Train the generator with this batch
                    loss_dict_generative = generator.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, rnt=1./task)
                    #loss_dict_generative = generator.train_a_batch(x_rel, y_rel, seq_start_end, seq_start_end_=seq_start_end_, rnt=1./task)
                    losses_dict_generative['loss_total'].append(loss_dict_generative['loss_total'])
                    losses_dict_generative['reconL'].append(loss_dict_generative['reconL'])
                    losses_dict_generative['variatL'].append(loss_dict_generative['variatL'])
                    losses_dict_generative['reconL_r'].append(loss_dict_generative['reconL_r'])
                    losses_dict_generative['variatL_r'].append(loss_dict_generative['variatL_r'])

            

            if args.val:
                if args.val_class == 'current':
                    val_dataset = data_loader(args, val_datasets[task-1], args.batch_size)
                    ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                    # save val loss
                    val_loss_file = open("{}/loss_val_{}_{}_{}_{}.txt".format(args.r_dir, args.iters, args.batch_size, args.replay, args.val_class), 'a')
                    val_loss_file.write('{}: {}\n'.format(epoch, loss_val))
                    val_loss_file.close()
                    loss_val_dict_main = {'loss_val': loss_val}
                    for val_loss_cb in val_loss_cbs:
                      if val_loss_cb is not None:
                         val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                    ade_val = ade_current
                    is_best = ade_val < best_ade
                    best_ade = min(ade_val, best_ade)
                    if is_best:
                        previous_model = copy.deepcopy(model)
                        file_dir = os.path.dirname(__file__) + "/chekpoint"
                        if os.path.exists(file_dir) is False:
                           os.mkdir(file_dir)
                        filename = os.path.join(file_dir,
                                               "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                   method=args.method, replay=args.replay, task=task,
                                                   order=args.dataset_order, batch_size=args.batch_size,
                                                   seed=args.seed, epoch=epoch,
                                                   val=args.val, val_class=args.val_class,
                                                   si=args.si, si_c= args.si_c))
                        torch.save(model.state_dict(), filename)
                        shutil.copyfile(filename, "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                   method=args.method, replay=args.replay, task=task,
                                                   order=args.dataset_order, batch_size=args.batch_size,
                                                   seed=args.seed,
                                                   val=args.val, val_class=args.val_class,
                                                   si=args.si, si_c= args.si_c))
                if args.val_class == 'all':
                    if generator is None:
                        val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(args, val_datasets[task -1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            for i in range(task - 1):
                                val_dataset_ = data_loader(args, val_datasets[i], args.batch_size)
                                ade_, _ = utils.validate_cl(args, model, val_dataset_, epoch)
                                ade_previous += ade_
                        else:
                            val_dataset = data_loader(args, val_datasets[task -1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                if args.val_class == 'replay':
                    if generator is None:
                        val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                    else:
                        if task >=2:
                            ade_previous = 0
                            val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = utils.validate_cl_replay(args, model, x_rel_val, y_rel_val, seq_start_end_val)
                        else:
                            val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
            else:
                val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                _, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                loss_val_dict_main = {'loss_val': loss_val}
                for val_loss_cb in val_loss_cbs:
                    if val_loss_cb is not None:
                        val_loss_cb(progress, epoch, loss_val_dict_main, task=task)

            # Main model
            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            # if args.val:
            #     model = copy.deepcopy(previous_model)

            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, epoch, losses_dict_main, task=task)
            if args.val:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(previous_model, epoch, task=task)
            else:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, epoch, task=task)
            if model.label == "VAE":
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(model, epoch, task=task)

            # # Generative model
            # # Fire callbacks on each iteration
            # for loss_cb in fake_gen_loss_cbs:
            #     if loss_cb is not None:
            #         loss_cb(progress_gen, epoch, loss_dict_fake_generative, task=task)
            # for sample_cb in sample_cbs:
            #     if sample_cb is not None:
            #         sample_cb(fake_generator, epoch, task=task)

            # Generative model
            # Fire callbacks on each iteration
            for loss_cb in gen_loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress_gen, epoch, losses_dict_generative, task=task)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(generator, epoch, task=task)
            
            if task > 1:
                ades = []
                fdes = []
                for i in range(1):
                    ade, fde = evaluate.validate(model, test_datasets[i])
                    ades.append(ade)
                    fdes.append(fde)
                average_ades = sum(ades)
                average_fdes = sum(fdes)
                all_metrics['epoch'].append(epoch + (task - 2) * iters_to_use)
                all_metrics['ade'].append(average_ades)
                all_metrics['fde'].append(average_fdes)

        # ----> UPON FINISHING EACH TASK...

        # Close progress-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        if args.val:
            model = copy.deepcopy(previous_model)
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        if args.val is False:
            previous_model = copy.deepcopy(model)
            file_dir = os.path.dirname(__file__)
            filename = os.path.join(file_dir, "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(method=args.method, replay=args.replay, task=task, order=args.dataset_order, batch_size=args.batch_size, seed=args.seed, val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c))
            torch.save(model.state_dict(), filename)

        best_ade = 200
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            model.update_omega(W, model.epsilon)
        
        if replay_model == 'der':
            pre_model = copy.deepcopy(model).eval()

        if replay_model == "der":
            Exact=True
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        f"{args.r_dir}/{replay_model}_mpr.csv", 
        index=False,
        float_format="%.4f"
        )