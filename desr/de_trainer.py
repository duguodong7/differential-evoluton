import os
import math
from decimal import Decimal

import utility
from utility import *
from utility import model_weights_as_vector, model_weights_as_dict
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import sade
from sade import model_dict_to_vector, model_vector_to_dict
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter

import data
import model as _model
import loss as _loss
from option import args

import logging

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
_logger = logging.getLogger('de_train')

logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
ckp = checkpoint

# loader = data.Data(args)
model = _model.Model(args, checkpoint)
loss = _loss.Loss(args, checkpoint) if not args.test_only else None

def main():
    writer=SummaryWriter("logs") #os.path.join(output_dir, 'summary.csv')
    setup_default_logging()
    # loader_train = loader.loader_train
    error_last = 1e8
    population = load_populaton(args.pop_init_dir, model, args.popsize)
    # PSNR_lst, SSIM_lst = os.path.join(args.pop_init_dir, 'args.yaml')
    # PSNR_lst, SSIM_lst = os.path.join(args.pop_init_dir, 'args.yaml')
    ssim_lst = [0 for i in range(args.popsize)]
    psnr_lst = [0 for i in range(args.popsize)]

    dim = len(model_weights_as_vector(model))
    bounds = 1
    paras = [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, [], []]
    #crm1, crm2, crm3, p1, p2, p3, p4, dyn_list_cr, dyn_list_nsf
    # if args.test_only: test()
    output_dir = args.output_dir

    eval_metrics = OrderedDict([('ssim', ssim_lst), ('psnr', psnr_lst)])
    loader_test = data.Data(args).loader_test
    ckp.write_log('loader test length:', len(loader_test))

    for epoch in range(args.de_epochs):
        seed = epoch + 2333
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        de_dataset = data.div2k.DIV2K(args, train=True, name='DIV2K', de=True)
        # loader_de_args['worker_seeding'] = epoch + 233
        # loader_de = create_loader(dataset_de, **loader_de_args)
        de_dataset.set_scale(0)
        loader_de = dataloader.DataLoader(de_dataset, batch_size=args.de_batch_size,
                                                            shuffle=False,
                                                            pin_memory=not args.cpu,
                                                            num_workers=args.n_threads,
                                                            persistent_workers=False)
        # if args.distributed:
        #     loader_de.sampler.set_epoch(epoch)
        eval_metrics = DE_epoch(
           epoch, model, loader_de, population, args, ckp, output_dir, loader_test, eval_metrics, writer)
        writer.close()
        # if saver is not None:
        #    # save proper checkpoint with eval metric
        #    save_metric = eval_metrics[eval_metric]
        #    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    # if best_metric is not None:
    #     ckp.write_log('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def DE_epoch(epoch, model, loader, population, args, ckp, output_dir, loader_test, eval_metrics, writer):
    de_paras = [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, [], []]
    dim = len(model_dict_to_vector(model))
    bounds = 1
    # epoch = self.optimizer.get_last_epoch() + 1
    ckp.write_log('[Epoch {}]\t'.format(epoch))
    model.eval()
    timer_de = utility.timer()
    timer_de.tic()
    score_lst = score_func(population)
    # best_score = -min(score_lst)
    bestidx = score_lst.index(min(score_lst))
    # de_iter_acc = [round(-i.item(), 4) for i in psnr_lst]
    fitness_train = [round(-j.item(), 4) for j in score_lst]
    ckp.write_log('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, fitness_train: {}'.format(0, min(score_lst), bestidx, fitness_train))

    train_metrics = OrderedDict([('iter', 0), ('bestidx', bestidx), ('train_loss', fitness_train)])

    popsize = len(population)
    de_iter_time_m = AverageMeter()
    end = time.time()

    for de_iter in range(1, args.de_iters+1):
        population, score_lst, bestidx, de_paras, change_label = sade.evolve4(
                       score_func, epoch, bounds, dim, popsize, population, score_lst, de_paras)
        solution = population[bestidx]
        best_score = score_lst[bestidx]
        model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
        model.load_state_dict(model_weights_dict)    

        # de_iter_acc = [round(-i.item(), 4) for i in psnr_lst]
        # fitness_train = [round(j.item(), 4) for j in score_lst]
        ckp.write_log('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, fitness_train: {}'.format(de_iter, best_score, bestidx, fitness_train))
        # eval_metrics_loss=[]
        for i in range(popsize): #!!!
            if change_label[i] == 1:
                solution = population[i]
                model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                model.load_state_dict(model_weights_dict)
                # eval_metrics_temp = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                eval_metrics_temp = validate(epoch, model, loader_test, args, ckp)
                # eval_metrics_loss.append(eval_metrics_temp['loss'])
                eval_metrics['ssim'][i] = round(eval_metrics_temp['ssim'], 4)
                eval_metrics['psnr'][i] = round(eval_metrics_temp['psnr'], 4)
                # eval_metrics['eval_loss'][i] = round(eval_metrics_temp['loss'], 4)
          
        ckp.write_log('eval_metrics_ssim: {}'.format(eval_metrics['ssim']))
        ckp.write_log('eval_metrics_psnr: {}'.format(eval_metrics['psnr']))
        # ckp.write_log('eval_metrics_acc1: {}'.format(eval_metrics['ssim']))
        # eval_metrics = OrderedDict([('loss', eval_metrics_loss), ('ssim', eval_metrics_acc1), ('psnr', eval_metrics_acc5)])
        # eval_metrics = OrderedDict([('ssim', 70.), ('psnr', 100)])
        # torch.cuda.synchronize()
        de_iter_time_m.update(time.time() - end)
        end = time.time()

        ckp.write_log(
                 'DE: {} [de_iter: {}]  '
                 'SSIM: {ssim:>7.4f}  '
                 'PSNR: {psnr:>7.4f}  '
                 'Iter_time: {de_iter_time.val:.3f}s, {rate:>7.2f}/s  '.format(
                     epoch, de_iter,
                     ssim = eval_metrics['ssim'][bestidx],
                     psnr = eval_metrics['psnr'][bestidx],
                     de_iter_time=de_iter_time_m,
                     rate= args.de_batch_size / de_iter_time_m.val))    #eval_metrics_temp

        # torch.save(population,'/root/declc/pth/population.pth')
        train_metrics = OrderedDict([('iter', de_iter), ('bestidx', bestidx), ('fitness_train', fitness_train)])
        writer.add_scalar("train_metrics", fitness_train, epoch*args.de_iters+de_iter)
        update_summary(
            epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
            write_header=True)

     return eval_metrics


    # timer_de.hold()
    # solution = self.population[bestidx]
    # best_score = self.score_lst[bestidx]
    # model_weights_dict = model_weights_as_dict(model = self.model, weights_vector = solution)
    # self.model.load_state_dict(model_weights_dict)    
    # self.optimizer.schedule()
    # self.ckp.write_log(
    #             '[{} x{}]\tSSIM: {:.4f} time: {:.1f}s epoch: {} crm: {},{},{}'.format(
    #                 'DIV2k',
    #                 self.scale,
    #                 -best_score,
    #                 timer_de.release(),
    #                 epoch,
    #                 self.paras[0],
    #                 self.paras[1],
    #                 self.paras[2]
    #             )
    #         )


def validate(epoch, model, loader_test, args, ckp):
    torch.set_grad_enabled(False)
    ckp.write_log('\nEvaluation:')
    # ckp.add_log(
    #     torch.zeros(1, len(loader_test), len(scale))
    # )
    # ckp.add_log2(
    #     torch.zeros(1, len(loader_test), len(scale))
    # )
    model.eval()

    timer_test = utility.timer()
    if args.save_results: ckp.begin_background()
    for idx_data, d in enumerate(loader_test):
        for idx_scale, scale in enumerate(args.scale):
            d.dataset.set_scale(idx_scale)
            for lr, hr, filename in d:
                lr, hr = prepare(lr, hr)
                # print('666', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
                sr = model(lr, idx_scale)
                sr = utility.quantize(sr, args.rgb_range)
                # print('777', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())

                save_list = [sr]
                psnr += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)
                ssim += utility.calc_ssim(sr, hr, scale, dataset=d)
            psnr /= len(d)
            ssim /= len(d)
                # ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                #     sr, hr, scale, args.rgb_range, dataset=d
                # )
                # ckp.log2[-1, idx_data, idx_scale] += utility.calc_ssim(
                #     sr, hr, scale, dataset=d
                # )
# 
                # if args.save_gt:
                #     save_list.extend([lr, hr])

                # if args.save_results:
                #     ckp.save_results(d, filename[0], save_list, scale)

            # ckp.log[-1, idx_data, idx_scale] /= len(d)
            # ckp.log2[-1, idx_data, idx_scale] /= len(d)
            # best = ckp.log.max(0)
            # best2 = ckp.log2.max(0)
            # ckp.write_log(
            #     '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
            #         d.dataset.name,
            #         scale,
            #         ckp.log[-1, idx_data, idx_scale],
            #         best[0][idx_data, idx_scale],
            #         best[1][idx_data, idx_scale] + 1
            #     )
            # )
            # ckp.write_log(
            #     '\tSSIM: {:.4f}(Best: {:.4f} @epoch {})'.format(
            #         ckp.log2[-1, idx_data, idx_scale],
            #         best2[0][idx_data, idx_scale],
            #         best2[1][idx_data, idx_scale] + 1
            #     )
            # )


    # ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
    # ckp.write_log('Saving...')

    # if args.save_results:
    #     ckp.end_background()

    # if not args.test_only:
    #     ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

    ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
    # torch.set_grad_enabled(True)
    metrics = OrderedDict([('psnr', psnr), ('ssim', ssim)])

    return metrics

def score_func(self, solution, model, population, loader, ckp):
    popsize = len(population)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    ssim_all = torch.zeros(popsize).tolist()
    psnr_all = torch.zeros(popsize).tolist()

    torch.set_grad_enabled(False)
    model.eval()
    end = time.time()
    for batch_idx, (lr, hr, filename) in enumerate(loader):
        if batch_idx >= 2: break
        # if batch_idx >= (args.de_batch_size//args.mini_batch_size): break
        data_time_m.update(time.time() - end)
        for i in range(0, popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            # if batch_idx==0 and args.local_rank < 1 and i == 0:
            if batch_idx==0 and i == 0:
                ckp.write_log('pop: {} input: {}'.format(i, input.flatten()[6000:6004]))
            lr, hr = prepare(lr, hr)
            sr = model(lr, 0)
            sr = utility.quantize(sr, 255)
            ssim = utility.calc_ssim(sr, hr, scale[0])
            psnr = utility.calc_psnr(sr, hr, scale[0], args.rgb_range)
            ssim_all[i] += ssim
            psnr_all[i] += psnr
        batch_time_m.update(time.time() - end)
        end = time.time()
        ckp.write_log('batch: {} '
              'data_time: {time1.val:.3f} ({time1.avg:.3f})  '
              'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(batch_idx, time1=data_time_m, time2=batch_time_m))                 
    revs_psnr_lst = [-i/(args.de_batch_size//args.mini_batch_size) for i in ssim_all]#!!!
    revs_ssim_lst = [-i/(args.de_batch_size//args.mini_batch_size) for i in psnr_all]#!!!
    return revs_ssim_lst, revs_psnr_lst#!!!


def load_populaton(pop_init_dir, model, popsize):
    population = []
    for file in os.listdir(pop_init_dir):
        if len(population) >= popsize: break
        elif file.split('_')[0] == 'model':
            resume_path = os.path.join(args.pop_init_dir, file)
            model.load_state_dict(resume_path, strict=False)
            solution = model_dict_to_vector(model).detach()
            population.append(solution)
    return population


def prepare(self, *args):
    if self.args.cpu:
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    def _prepare(tensor):
        if self.args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(a) for a in args]


if __name__ == '__main__':
    main()











