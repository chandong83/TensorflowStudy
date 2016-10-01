#encoding:Utf-8
'''
A Dynamic Reccurent Neural Netowrk (LSTM) Implementation example using TensorFlow libray.
This example is using a toy dataset to classify linear sequences. The generated sequence have variable length.

'''
# 가변적인 데이터 배열이 linear 한지 random한지를 판별해주는 rnn
# 여기서 중요한 것은 가변적인 데이터를 학습시킨다
# tf.nn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)에서
# sequence_length에 가변적인 길이 값을 제공하여서 가변적인 값에 대한 학습을 시켜서 random인지 linear한지
# 판단을 한다.

# 가변 길이의 데이터를 어떻게 학습을 시킬 수 있을까?
# 어떻게 사용하냐 말고 어떻게 학습을 시키셨나요?
# LSTM  / RNN을 이용하는 것은 알겠습니다 그 안에 내용을 보여주시죠??




import tensorflow as tf
import random

# =====================
# TODY DATA GENERATOR
# =====================

class ToySequenceData(object):
    # class를 초기화 할 때 data, labels, seqlen에 대한 값을 저장한다
    # next함수를 통해서 3가지를 반환한다 batch_x, batch_y, batch_seqlen

    # batch_x의 경우 linear 또는 random한 배열을 가진 data배열을 반환한고
    # batch_y의 경우 [0., 1.](random을 표현) 또는 [1. , 0.](linear을 표현)의 레이블을 반환한다.
    # batch_seqlen 은 data가 가지고 있는 값의 길이를 반환한다.
    # 클래스에 대한 초기값으로 n_samples, max_seql_len, min_seq_len, max_value를 받게 되는데
    # i/max_value의 값들로 data가 만들어지고 i는 randdom값이거나, 0~(1000-max_seq_len) 시작으로 연속된 값으로 선택이 된다
    # len의 경우 가변적으로 선택이 되는데 min,max seq_len 사이의 길이가 선택되고
    # 길이만큼 데이터를 채운후 나머지 빈자리는 0으로 패딩을 채워준다.
    #

    """Generate sequence of data with dynamic length.
    This class generate of data with dynamic length.
    -Class 0: linear sequences(i.e. [0, 1, 2, 3, ...])
    -Class 1: random sequences(i.e. [1, 3, 10, 7, ...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with unconsistent dimensions).
    The dynamic caculation will then be perform thanks to 'seqlen' attribute that records
    every actual sequence length.
    """

    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data=[]
        self.labels=[]
        self.seqlen=[]
        for i in range(n_samples):
            #Random sequence length
            len=random.randint(min_seq_len, max_seq_len)
            #Monitor sequenc length for TensorFlow dynamic caculation
            self.seqlen.append(len)
            #Add a random or linear int sequence (50% prob)
            if random.random()<.5:
                #Generate a linear sequence
                rand_start=random.randint(0, max_value-len)
                s=[[float(i)/max_value] for i in range(rand_start, rand_start+len)]
                #Pad sequence for dimension consistency
                s+=[[0.] for i in range(max_seq_len-len)]
                self.data.append(s)
                self.labels.append([1.,0.])
            else:
                #Generate a random sequence
                s=[[float(random.randint(0, max_value))/max_value] for i in range(len)]
                #Pad seuence for dimension consistency
                s+=[[0.] for i in range(max_seq_len-len)]
                self.data.append(s)
                self.labels.append([0.,1.])
        self.batch_id=0

    def next(self, batch_size):
        """Return a batch of data. When dataset end is reached, start over."""

        if self.batch_id==len(self.data):
            self.batch_id=0
        batch_data=(self.data[self.batch_id:min(self.batch_id+batch_size,len(self.data))])
        batch_labels=(self.labels[self.batch_id:min(self.batch_id+batch_size, len(self.data))])
        batch_seqlen=(self.seqlen[self.batch_id:min(self.batch_id+batch_size, len(self.data))])
        self.batch_id=min(self.batch_id+batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ==============
# MODEL
# ==============

# Parameterrs
learning_rate=0.01
training_iters=1000000
batch_size=128
display_step=10


#Network Parameters
seq_max_len=20 #Sequence max length
n_hidden=64 #hidden layer num of features
n_classes =2 #linear sequence or not

trainset= ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset= ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

#tf Graph input
x=tf.placeholder("float", [None, seq_max_len, 1])
y=tf.placeholder("float", [None, n_classes])

#A placeholder for indicating each sequence length
seqlen=tf.placeholder(tf.int32, [None])

#Define weights
weights={'out':tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases={'out':tf.Variable(tf.random_normal([n_classes]))}

def dynamicRNN(x, seqlen, weights, biases):
    #Prepare data shape to match 'rnn' function requiremetns
    #Current data input shape:(batch_size, n_steps, n_input)
    #Required shpae:'n_steps' tensors list of shape (batch_size, n_input)

    #Permuting batch_size and n_steps
    x=tf.transpose(x,[1,0,2])
    #Reshaping to (n_steps*batch_size, n_input)
    x=tf.reshape(x,[-1,1])
    #Split to get a list of 'n_steps' tensor of shape (batch_size, n_input)
    x=tf.split(0, seq_max_len, x)

    #Define a lstm cell with tensorflow
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    #Get lstm cell output, providing 'sequence_length' will perform dynamic calculation
    outputs, states=tf.nn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    #When performing dynamic caculation we must retrieve the last
    # dynamically computed output, i.e, if a sequence length is 10, we need
    # to retrieve the 10th output
    #However TensorFlow doesn't support advanced indexing yet, so we build
    #a custom op that for each sample in batch size, get its length and
    #get the corresponding relevant output

    #'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_steps, n_input]

    outputs=tf.pack(outputs)
    outputs=tf.transpose(outputs, [1,0,2])

    #Hack to build the indexing and retrieve the right output
    batch_size=tf.shape(outputs)[0]
    #Start indices for each sample
    index=tf.range(0, batch_size)*seq_max_len+(seqlen-1)
    #Indexing
    outputs=tf.gather(tf.reshape(outputs,[-1, n_hidden]), index)

    #Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out'])+biases['out']

pred=dynamicRNN(x, seqlen, weights, biases)

#Define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred=tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initializing the variables
init=tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step=1
    #Keep training until reach max iterations
    while step*batch_size<training_iters:
        batch_x, batch_y, batch_seqlen=trainset.next(batch_size)
        #Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})

        if step%display_step==0:
            #Calculate batch accuracy
            acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})

            #Caculate batch loss
            loss=sess.run(cost, feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})

            print "Iter"+str(step*batch_size)+", Minibatch Loss= "+\
                "{:.6f}".format(loss)+", Training Accuracy= "+\
                "{:.5f}".format(acc)
        step+=1
    print "Optimization Finished!"

    #Caculate accuracy
    test_data=testset.data
    test_label=testset.labels
    test_seqlen=testset.seqlen
    print biases, sess.run(biases['out'])
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x:test_data, y:test_label, seqlen:test_seqlen})

