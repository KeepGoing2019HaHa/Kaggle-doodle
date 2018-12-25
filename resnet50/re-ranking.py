# use 2nd cls for re-ranking
from common import *
from data   import *
from resnet50.model32_resnet50 import *
import pandas as pd
import numpy as np

def test_augment(drawing, label, index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache


ori = pd.read_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k.submit.csv')

def re_ranking(ori, re_ranking_class, ckpt):

    # 'broccoli', 'tree'
    res = []
    for i in range(len(ori)):
        pred = ori['word'][i].split(' ')
        if re_ranking_class[0] in pred and re_ranking_class[1] in pred:
            res.append(i)

    re_ranking_dataset = ReRankingDataset(res, test_augment)
    re_ranking_loader  = DataLoader(
                            re_ranking_dataset,
                            sampler     = SequentialSampler(re_ranking_dataset),
                            batch_size  = 512,
                            drop_last   = False,
                            num_workers = 2,
                            pin_memory  = True,
                            collate_fn  = null_collate)

    net = Net().cuda()
    # net.load_state_dict(torch.load('/coer/yiwei/kaggle/doodle/results-cnn-11-07/broccoli-tree/checkpoint/00001000_model.pth'))
    net.load_state_dict(torch.load(ckpt))
    net.set_mode('test')

    probs    = []
    for input, truth, cache in re_ranking_loader:
        input = input.cuda()
        with torch.no_grad():
            logit   = net(input)
            prob    = F.softmax(logit,1)
        probs.append(prob.data.cpu().numpy())
    probs = np.concatenate(probs)

    cnt = 0
    for i,v in enumerate(res):
        pred = ori['word'][v].split(' ')
        idx1,idx2 = pred.index(re_ranking_class[0]), pred.index(re_ranking_class[1])
        idx1,idx2 = sorted([idx1,idx2])
        # print(probs[i][0], probs[i][1])
        if 0.98<probs[i][0]+probs[i][1]<1.01:
            pass
        else:
            print(str(probs[i][0]+probs[i][1]))

        if probs[i][0]>probs[i][1]:
            pred[idx1]=re_ranking_class[0]
            pred[idx2]=re_ranking_class[1]
        # ori.loc[v]['word']=' '.join(pred)

        if ' '.join(pred)!=ori.loc[v]['word']: cnt+=1
        # print(ori.loc[v]['word'])
        ori.set_value(v,'word',' '.join(pred))
        # print(ori.loc[v]['word'])
        assert ' '.join(pred)==ori.loc[v]['word']

    return ori, cnt

# check on validation set ?
# ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m1.submit.csv',
#            index=False, columns=['key_id', 'word'])
# pass

ori, cnt1 = re_ranking(ori, ['broccoli', 'tree'], '/coer/yiwei/kaggle/doodle/results-cnn-11-07/broccoli-tree/checkpoint/00001000_model.pth')
ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m1.submit.csv', index=False, columns=['key_id', 'word'])
ori, cnt2 = re_ranking(ori, ['hurricane', 'tornado'], '/coer/yiwei/kaggle/doodle/results-cnn-11-07/hurricane-tornado/checkpoint/00001000_model.pth')
ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m2.submit.csv', index=False, columns=['key_id', 'word'])
ori, cnt3 = re_ranking(ori, ['school_bus', 'bus'], '/coer/yiwei/kaggle/doodle/results-cnn-11-07/school_bus-bus/checkpoint/00005000_model.pth')
ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m3.submit.csv', index=False, columns=['key_id', 'word'])
ori, cnt4 = re_ranking(ori, ['cake', 'birthday_cake'], '/coer/yiwei/kaggle/doodle/results-cnn-11-07/cake-birthday_cake/checkpoint/00001000_model.pth')
ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m4.submit.csv', index=False, columns=['key_id', 'word'])
ori, cnt5 = re_ranking(ori, ['coffee_cup', 'mug'], '/coer/yiwei/kaggle/doodle/results-cnn-11-07/coffee_cup-mug/checkpoint/00002000_model.pth')
ori.to_csv('/coer/yiwei/kaggle/doodle/results-cnn-2/resnet50-fold-final/test/test-null-220k-m5.submit.csv', index=False, columns=['key_id', 'word'])



# print(cnt1,cnt2,cnt3)