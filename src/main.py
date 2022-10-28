from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def collate_fn(batch):
    data = [item[0] for item in batch]
    # 这里对我的target进行了reshape操作
    target = [torch.reshape(item[1], (-1,)) for item in batch]
    data = torch.stack(data)
    target = torch.stack(target)
    return [data, target]



def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  #torch.backends.cudnn.enabled = False
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  '''
  opt.arch: res\dla34\swint\...
  opt.heads: 'hm':num_class 'wh':2 'reg':2
  opt.head_conv: 'conv layer channels for output head'最后卷积输出的通道数
  '''
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print(model+"dddd")
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.937, nesterov=True)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1,#opt.batch_size
      shuffle=False,
      num_workers=opt.num_workers,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'),
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      # collate_fn=collate_fn,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    log_out = ''
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    log_out = log_out + 'epoch: {} |'.format(epoch)
    for k, v in log_dict_train.items():#train loss
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
      log_out = log_out + 'train_{} {:8f} | '.format(k, v)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:#val loss
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      log_val = ""
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
        log_val = log_val + 'val_{} {:8f} | '.format(k, v)
      if log_val != '':
        log_out = log_out + "\n-------------" + log_val + "\n"

      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)

    logger.write('\n')
    save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    print(log_out)
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)