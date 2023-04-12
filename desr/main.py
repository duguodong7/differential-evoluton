import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer



torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            # _model2 = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                if t.optimizer.get_last_epoch() < args.pretrain_epoch:
                    t.train()
                    t.test()
                    t.population = t.get_pop.population.tolist()
                    t.score_lst = t.get_pop.score_lst.tolist()
                else:
                    t.DE_train()
                    t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
