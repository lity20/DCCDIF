import os
import sys
import json
import importlib
from datetime import datetime
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import Network
from dataset import ShapeNet


def prepare(config):
    os.makedirs(config['experiment']['dir'])
    with open(os.path.join(config['experiment']['dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    src_dir = os.path.join(config['experiment']['dir'], 'src')
    os.makedirs(src_dir)
    os.system('cp *.py ' + src_dir)
    os.makedirs(config['experiment']['ckpt_save_dir'])
    os.makedirs(config['experiment']['log_dir'])


class LossHelper:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.loss_dict = None
        self.count = 0
    
    def accumulate(self, loss_dict):
        if self.count == 0:
            self.loss_dict = {}
            for k, v in loss_dict.items():
                self.loss_dict[k] = v.item()
        else:
            for k, v in loss_dict.items():
                self.loss_dict[k] += v.item()
        self.count += 1
    
    def write_log(self, writer, step):
        if self.count == 0:
            return
        for k, v in self.loss_dict.items():
            v /= self.count
            writer.add_scalar(k, v, global_step=step)
            print(k, ':', v)


def load_ckpt(config, network, decoder_optimizer, latent_codes_optimizer, decoder_lr_scheduler, latent_codes_lr_scheduler):
    ckpt_decoder = torch.load(config['train']['pretrain_ckpt_decoder'])
    ckpt_decoder_optimizer = torch.load(config['train']['pretrain_ckpt_decoder_optimizer'])
    ckpt_decoder_lr_scheduler = torch.load(config['train']['pretrain_ckpt_decoder_lr_scheduler'])
    ckpt_latent_codes = torch.load(config['train']['pretrain_ckpt_latent_codes'])
    ckpt_latent_codes_optimizer = torch.load(config['train']['pretrain_ckpt_latent_codes_optimizer'])
    ckpt_latent_codes_lr_scheduler = torch.load(config['train']['pretrain_ckpt_latent_codes_lr_scheduler'])

    network.decoder.load_state_dict(ckpt_decoder['decoder'])
    network.code_cloud.load_state_dict(ckpt_latent_codes['latent_codes'])
    decoder_optimizer.load_state_dict(ckpt_decoder_optimizer['decoder_optimizer'])
    latent_codes_optimizer.load_state_dict(ckpt_latent_codes_optimizer['latent_codes_optimizer'])
    decoder_lr_scheduler.load_state_dict(ckpt_decoder_lr_scheduler['decoder_lr_scheduler'])
    latent_codes_lr_scheduler.load_state_dict(ckpt_latent_codes_lr_scheduler['latent_codes_lr_scheduler'])

    start_epoch = ckpt_decoder['epoch'] + 1
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Load pretrain model, start from epoch %d.' % start_epoch)
    return start_epoch


def save_ckpt(config, epoch, network, decoder_optimizer, latent_codes_optimizer, decoder_lr_scheduler, latent_codes_lr_scheduler):
    torch.save({'epoch': epoch, 'decoder': network.decoder.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-decoder.pth'%epoch))
    torch.save({'epoch': epoch, 'latent_codes': network.code_cloud.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-latent_codes.pth'%epoch))
    torch.save({'epoch': epoch, 'decoder_optimizer': decoder_optimizer.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-decoder_optimizer.pth'%epoch))
    torch.save({'epoch': epoch, 'latent_codes_optimizer': latent_codes_optimizer.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-latent_codes_optimizer.pth'%epoch))
    torch.save({'epoch': epoch, 'decoder_lr_scheduler': decoder_lr_scheduler.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-decoder_lr_scheduler.pth'%epoch))
    torch.save({'epoch': epoch, 'latent_codes_lr_scheduler': latent_codes_lr_scheduler.state_dict()}, os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-latent_codes_lr_scheduler.pth'%epoch))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Checkpoint saved to', os.path.join(config['experiment']['ckpt_save_dir'], 'epoch%d-*.pth'%epoch))


def train(config):
    prepare(config)
    loss_helper = LossHelper()
    writer = SummaryWriter(config['experiment']['log_dir'])
    dataset = ShapeNet(config['train_dataset'])
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    network = Network(config['network'], len(dataset)).cuda()
    decoder_optimizer = Adam(network.decoder.parameters(), lr=config['train']['decoder_init_lr'], weight_decay=config['train']['decoder_weight_decay'])
    decoder_lr_scheduler = StepLR(decoder_optimizer, step_size=config['train']['decoder_lr_decay_step'], gamma=config['train']['decoder_lr_decay_rate'])
    latent_codes_optimizer = Adam(network.code_cloud.parameters(), lr=config['train']['latent_codes_init_lr'])
    latent_codes_lr_scheduler = StepLR(latent_codes_optimizer, step_size=config['train']['latent_codes_lr_decay_step'], gamma=config['train']['latent_codes_lr_decay_rate'])

    start_epoch = 0
    if config['train']['pretrain']:
        start_epoch = load_ckpt(config, network, decoder_optimizer, latent_codes_optimizer, decoder_lr_scheduler, latent_codes_lr_scheduler)
    network.train()
    for epoch in range(start_epoch, config['train']['num_epoch']):
        print('****** %s ******\ntime: %s\nepoch: %d/%d' % (config['experiment']['name'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, config['train']['num_epoch']))
        loss_helper.clear()
        for batch_idx, (query_points, gt_sd, indices) in enumerate(data_loader):
            indices = indices.cuda()
            query_points = query_points.float().cuda()
            gt_sd = gt_sd.float().cuda()
            decoder_optimizer.zero_grad()
            latent_codes_optimizer.zero_grad()
            pred_sd = network(indices, query_points)
            loss_dict = network.loss(gt_sd)
            loss_helper.accumulate(loss_dict)
            loss_dict['total_loss'].backward()
            decoder_optimizer.step()
            latent_codes_optimizer.step()
            print('Current epoch progress: %d/%d...' % (batch_idx, len(data_loader)), end='\r')
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Current epoch progress: %d/%d. Done.' % (len(data_loader), len(data_loader)))
        
        decoder_lr_scheduler.step()
        latent_codes_lr_scheduler.step()
        loss_helper.write_log(writer, epoch)
        writer.add_scalar('decoder_lr', decoder_optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar('latent_codes_lr', latent_codes_optimizer.param_groups[0]['lr'], global_step=epoch)
        if epoch % config['train']['ckpt_save_frequency'] == 0:
            save_ckpt(config, epoch, network, decoder_optimizer, latent_codes_optimizer, decoder_lr_scheduler, latent_codes_lr_scheduler)

    save_ckpt(config, config['train']['num_epoch'], network, decoder_optimizer, latent_codes_optimizer, decoder_lr_scheduler, latent_codes_lr_scheduler)
    writer.close()


if __name__ == '__main__':
    config = importlib.import_module(sys.argv[1]).config
    train(config)
