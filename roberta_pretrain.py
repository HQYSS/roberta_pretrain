import torch
from torch import nn
import model
import dataset
from torch.utils.tensorboard import SummaryWriter


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y):
    # 前向传播
    _, mlm_Y_hat = net(tokens_X, valid_lens_x, pred_positions_X)
    # print(mlm_weights_X)
    ori_mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    # print("ori:", ori_mlm_l)
    mlm_l=ori_mlm_l * mlm_weights_X.reshape(-1)
    # print("mlm_l: ", mlm_l)
    # print("mlm_l.sum()", mlm_l.sum())
    # print('mlm_w', mlm_weights_X)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    return mlm_l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps, writer, from_checkpoint=False):
    net = net.to(devices[1])
    trainer = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.9,0.98), eps=1e-6, weight_decay=0.01)
    step = 0
    num_steps_reached = False
    if from_checkpoint==True:
        checkpoint=torch.load(state, '/home/lhy/bert/checkpoint.tar')
        step=checkpoint['step']
        net.load_state_dict(checkpoint['model'])
        trainer.load_state_dict(checkpoint['optimizer'])
    while step < num_steps and not num_steps_reached:
        print(f"第{step}次训练")
        for tokens_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y in train_iter:
            tokens_X = tokens_X.to(devices[1])
            valid_lens_x = valid_lens_x.to(devices[1])
            pred_positions_X = pred_positions_X.to(devices[1])
            mlm_weights_X = mlm_weights_X.to(devices[1])
            mlm_Y = mlm_Y.to(devices[1])
            trainer.zero_grad()
            l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y)
            writer.add_scalar("Loss", l, step)
            l.backward()
            trainer.step()

            if step%5000==0:
                state={
                    'step': step,
                    'model': net.state_dict(),
                    'optimizer': trainer.state_dict(),
                }
                torch.save(state, '/home/lhy/bert/checkpoint.tar')

            step += 1

            if step == num_steps:
                num_steps_reached = True
                break


if __name__=='__main__':

    writer=SummaryWriter('/home/lhy/bert/tensorboard')

    batch_size, max_len = 512, 512
    train_iter = dataset.load_data(batch_size, max_len)

    net = model.BERTModel(vocab_size=50265, num_hiddens=768, norm_shape=[768],
                        ffn_num_input=768, ffn_num_hiddens=3072, num_heads=12,
                        num_layers=12, dropout=0.1, key_size=768, query_size=768,
                        value_size=768, hid_in_features=768, mlm_in_features=768)
    devices = [torch.device(f'cuda:{i}')
                for i in range(torch.cuda.device_count())]
    loss = nn.CrossEntropyLoss(reduction='none')
    train_bert(train_iter, net, loss, vocab_size=50265, devices=devices, num_steps=128000, writer=writer)
    writer.close()
    torch.save(net.state_dict(), "/home/lhy/bert/model_saved")