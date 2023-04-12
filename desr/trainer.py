import os
import math
from decimal import Decimal

import utility
from utility import model_weights_as_vector, model_weights_as_dict
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from GA_model import Get_Pop
import sade

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_de_train = loader.loader_de_train
        print('loader test length:', len(self.loader_test))
        self.model = my_model
        # self.model2 = my_model2
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        # self.GA_optimizer = utility.make_optimizer(args, self.model, self.score_function)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.get_pop = Get_Pop(model=self.model, popsize=args.popsize)
        # self.population = self.get_pop.population
        self.population = None
        self.score_lst = None
        self.dim = len(model_weights_as_vector(self.model))
        self.bounds = 1
        self.popsize = args.popsize
        self.paras = [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, [], []]
        #crm1, crm2, crm3, p1, p2, p3, p4, dyn_list_cr, dyn_list_nsf
        
    def score_func(self, solution):
        torch.set_grad_enabled(False)
        model = self.model
        model.eval()
        model_weights_dict = model_weights_as_dict(model=model,
                                                   weights_vector = solution)
        model.load_state_dict(model_weights_dict)
        aaa = 0
        All_score = 0
        self.loader_de_train.dataset.set_scale(0)
        time_score_function = utility.timer()
        time_score_function.tic()
        with torch.no_grad():
            d = self.loader_de_train
            for lr, hr, filename in d:
                lr, hr = self.prepare(lr, hr)
                # print('333', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
                sr = model(lr, 0)
                # print('444', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
                sr = utility.quantize(sr, 255)
                score = -utility.calc_ssim(sr, hr, self.scale[0])
                # print('555', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
                # score = utility.calc_psnr(sr, hr, self.scale[0], self.args.rgb_range)
                aaa += 1
                All_score += score
        time_score_function.hold()
        print('aaa:{}, score:{:.4f}, time_score_function:{:.1f}s'.format(aaa, -All_score/aaa, time_score_function.release()))
        return All_score/aaa

    def DE_train(self):
        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('[Epoch {}]\t'.format(epoch))
        self.model.eval()
        timer_de = utility.timer()
        timer_de.tic()

        self.population, self.score_lst, bestidx, self.paras = sade.evolve4(
                                                self.score_func, epoch, self.bounds, self.dim, self.popsize,
                                                self.population, self.score_lst, self.paras
                                                )
        timer_de.hold()
        solution = self.population[bestidx]
        best_score = self.score_lst[bestidx]
        model_weights_dict = model_weights_as_dict(model = self.model, weights_vector = solution)
        self.model.load_state_dict(model_weights_dict)    
        self.optimizer.schedule()
        self.ckp.write_log(
                    '[{} x{}]\tSSIM: {:.4f} time: {:.1f}s epoch: {} crm: {},{},{}'.format(
                        'DIV2k',
                        self.scale,
                        -best_score,
                        timer_de.release(),
                        epoch,
                        self.paras[0],
                        self.paras[1],
                        self.paras[2]
                    )
                )


    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_last_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            # print('111', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
            loss = self.loss(sr, hr)
            loss.backward()
            # print('222', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        solution = model_weights_as_vector(self.model).copy()
        score = self.score_func(solution)
        self.get_pop(score, self.model)

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.ckp.add_log2(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in d:
                    lr, hr = self.prepare(lr, hr)
                    # print('666', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # print('777', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log2[-1, idx_data, idx_scale] += utility.calc_ssim(
                        sr, hr, scale, dataset=d
                    )

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log2[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                best2 = self.ckp.log2.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                self.ckp.write_log(
                    '\tSSIM: {:.4f}(Best: {:.4f} @epoch {})'.format(
                        self.ckp.log2[-1, idx_data, idx_scale],
                        best2[0][idx_data, idx_scale],
                        best2[1][idx_data, idx_scale] + 1
                    )
                )


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

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

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

