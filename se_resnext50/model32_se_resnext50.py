from common import *
from net.imagenet_pretrain_model.se_resnext import *
# from torchvision import models



#<todo> bce, focal loss, direct top 5 sugorate
#<todo> 'smooth loss functions for deep top-k classification' - Leonard Berrada, Andrew Zisserman, M. Pawan Kumar
#       https://arxiv.org/abs/1802.07595



def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss


def metric(logit, truth, is_average=True):

    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        if is_average==True:
            # top-3 accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct/len(truth)

            top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
            precision = correct[0]/1 + correct[1]/2 + correct[2]/3
            return precision, top

        else:
            return correct



###########################################################################################3

class Net(nn.Module):

    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if 'last_linear' in key: continue
            state_dict[key] = pretrain_state_dict[key[11:]]
            print(key)
        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=340):
        super(Net,self).__init__()
        self.se_resnext = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                                  dropout_p=None, inplanes=64, input_3x3=False,
                                  downsample_kernel_size=1, downsample_padding=0,
                                  num_classes=num_class)

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x/255.0
        mean=[0.485, 0.456, 0.406] #rgb
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:,[0]]-mean[0])/std[0],
            (x[:,[1]]-mean[1])/std[1],
            (x[:,[2]]-mean[2])/std[2],
        ],1)

        logit = self.se_resnext(x)
        return logit


    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False





### run ##############################################################################



def run_check_net():

    batch_size = 32
    C,H,W = 3, 32, 32
    num_class = 5

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()


    #---
    criterion = softmax_cross_entropy_criterion
    net = Net(num_class).cuda()
    net.set_mode('train')
    # print(net)
    ## exit(0)

    net.load_pretrain('/root/share/project/kaggle/tgs/data/model/resnet34-333f7ec4.pth')

    logit = net(input)
    loss  = criterion(logit, truth)
    precision, top = metric(logit, truth)

    print('loss    : %0.8f  '%(loss.item()))
    print('correct :(%0.8f ) %0.8f  %0.8f '%(precision.item(), top[0].item(),top[-1].item()))
    print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    print('        loss  | prec      top      ')
    print('[iter ]       |           1  ... k ')
    print('-------------------------------------')
    while i<=500:

        logit   = net(input)
        loss    = criterion(logit, truth)
        precision, top = metric(logit, truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
            ))
        i = i+1





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')