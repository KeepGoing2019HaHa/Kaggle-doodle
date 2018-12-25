from common import *
from data   import *
from model32_resnet50 import *

from tqdm import tqdm


def train_augment(drawing, label, index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache

def valid_augment(drawing, label, index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache


batch_size = 2048

# train_dataset = DoodleDataset('train', 'train_0', train_augment)
# train_loader  = DataLoader(
#                     train_dataset,
#                     sampler     = SequentialSampler(train_dataset),
#                     batch_size  = batch_size,
#                     drop_last   = True,  # no ...
#                     num_workers = 2,
#                     pin_memory  = True,
#                     collate_fn  = null_collate)
valid_dataset = DoodleDataset('train', 'valid_1',  valid_augment)
valid_loader  = DataLoader(
                    valid_dataset,
                    sampler     = SequentialSampler(valid_dataset),
                    batch_size  = batch_size,
                    drop_last   = False,
                    num_workers = 2,
                    pin_memory  = True,
                    collate_fn  = null_collate)



net = Net().cuda()
net.load_state_dict(torch.load('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/checkpoint/00220000_model.pth'))
# net = nn.DataParallel(net)
net.set_mode('test')


probs    = []
truths   = []

for input, truth, cache in tqdm(valid_loader):
    input = input.cuda()
    truth = truth.cuda()

    with torch.no_grad():
        logit   = net(input)
        prob    = F.softmax(logit,1)
        # _,index = prob.max(1)
    prob = prob.data.cpu().numpy()
    truth = truth.data.cpu().numpy()
    probs.append(prob[range(prob.shape[0]),truth])
    # pass
    # probs.append(prob.data.cpu().numpy())
    # truths.append(truth.data.cpu().numpy())


pickle.dump(probs, open('val.probs.pkl', 'wb'))
# pickle.dump(truths, open('val.truths.pkl', 'wb'))