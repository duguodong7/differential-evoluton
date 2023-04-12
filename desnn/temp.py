model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
model.load_state_dict(model_weights_dict)

            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # if batch_idx==0 and args.local_rank == 0:#!!!
            #    _logger.info('validate, input: {}'.format(input.flatten()[6000:6005]))#!!!

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            # reduced_loss_all = torch.zeros(popsize).tolist()
            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
