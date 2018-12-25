from common import *
from data   import *


def prob_to_csv(prob, key_id, csv_file):
    top = np.argsort(-prob,1)[:,:3]
    word = []
    for (t0,t1,t2) in top:
        word.append(
            CLASS_NAME[t0] + ' ' + \
            CLASS_NAME[t1] + ' ' + \
            CLASS_NAME[t2]
        )
    df = pd.DataFrame({ 'key_id' : key_id , 'word' : word}).astype(str)
    df.to_csv(csv_file, index=False, columns=['key_id', 'word'])


complexity='simplified'
TEST_DF = pd.read_csv(DATA_DIR + '/csv/test_%s.csv'%(complexity))
key_id = TEST_DF['key_id'].values



step_probs = []
for step in ['330k', '328k', '326k']:
    prob_null_file = '/datanew/DATASET/doodle/results/2018-12-1/test/test-%s-null.prob.uint8.npy'%(step)
    prob_null = np_uint8_to_float32(np.load(prob_null_file))
    prob_flip_file = '/datanew/DATASET/doodle/results/2018-12-1/test/test-%s-flip.prob.uint8.npy'%(step)
    prob_flip = np_uint8_to_float32(np.load(prob_flip_file))
    prob = 0.6*prob_flip+prob_null
    step_probs.append(prob)

# prob = sum(step_probs)/len(step_probs)
prob = (6*step_probs[0]+3*step_probs[1]+3*step_probs[2])/12
prob_to_csv(prob, key_id, '/datanew/DATASET/doodle/results/2018-12-1/test/ensemble.submit.csv')

        
        
        
# # TTA
# prob_null_file = '/datanew/DATASET/doodle/results/2018-12-1/test/test-326k-null.prob.uint8.npy'
# prob_null = np_uint8_to_float32(np.load(prob_null_file))
# print(prob_null.shape)
#
# prob_flip_file = '/datanew/DATASET/doodle/results/2018-12-1/test/test-326k-flip.prob.uint8.npy'
# prob_flip = np_uint8_to_float32(np.load(prob_flip_file))
# print(prob_flip.shape)
#
# # prob = 1.0*prob_flip+prob_null
# # prob = 0.8*prob_flip+prob_null
# prob = 0.6*prob_flip+prob_null
# # prob = 0.4*prob_flip+prob_null
# # prob = 0.2*prob_flip+prob_null  # maybe due to training has not include flip aug
#
# prob_to_csv(prob, key_id, '/datanew/DATASET/doodle/results/2018-12-1/test/tta0.6-326k.submit.csv')