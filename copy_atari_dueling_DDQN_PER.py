import gym
import tensorflow as tf
#from tensorflow import keras
import keras
import random
import numpy as np
import datetime as dt
import imageio
import os
import keyboard
import multiprocessing as mp
import csv

from tensorflow.keras.mixed_precision import experimental as mixed_precision


STORE_PATH = 'C:\\TensorFlowBook\\TensorBoard\\More\\'
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
GAMMA = 0.99
BATCH_SIZE = 8
TAU = 0.05
POST_PROCESS_IMAGE_SIZE = (105, 80, 1)
DELAY_TRAINING = 10000
BETA_DECAY_ITERS = 500000
MIN_BETA = 0.4
MAX_BETA = 1.0
NUM_FRAMES = 4
GIF_RECORDING_FREQ = 500
MODEL_SAVE_FREQ = 100

has_human_file = False
row_count = 0

if os.path.isfile("saved_memory.csv"):
    with open("saved_memory.csv") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        row_count = sum(1 for row in reader)
        has_human_file = True
        print("Human File Found")
    

color = np.array([210, 164, 74]).mean()


NEW = True

env = gym.make("Phoenix-v4")
num_actions = env.action_space.n

input_shape =(num_actions,105, 80, 4)

print(tf.config.experimental.list_physical_devices('GPU'))

#prelu = tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25))
# huber_loss = keras.losses.Huber()
def huber_loss(loss):
    return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

class DQModel(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQModel, self).__init__()
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        self.dueling = dueling
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), (4, 4), padding="same",activation=None,use_bias=False,input_shape=(32,num_actions),kernel_initializer=keras.initializers.VarianceScaling(scale=2.0))
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), padding="same",activation='relu',use_bias=False,input_shape=(32,num_actions),kernel_initializer=keras.initializers.he_uniform())
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding="same",activation='relu',use_bias=False,input_shape=(32,num_actions),kernel_initializer=keras.initializers.he_uniform())
        self.conv4 = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), padding="same",activation='relu',use_bias=False,input_shape=(32,num_actions),kernel_initializer=keras.initializers.he_uniform())
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.18)

        self.pooling_layer1 = tf.keras.layers.MaxPool2D((2,2),strides=(1,1), padding="same")
        self.pooling_layer2 = tf.keras.layers.MaxPool2D((2,2),strides=(1,1), padding="same")
        self.pooling_layer3 = tf.keras.layers.MaxPool2D((2,2),strides=(1,1), padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.flatten1 = tf.keras.layers.Flatten()
        self.BatchNormalization = tf.keras.layers.BatchNormalization(momentum=.90,input_shape=(num_actions,105, 80, 4))
        self.BatchNormalization2 = tf.keras.layers.BatchNormalization()
        self.BatchNormalization3 = tf.keras.layers.BatchNormalization()
        self.BatchNormalization4 = tf.keras.layers.BatchNormalization()
        self.adv_dense1 = tf.keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_uniform())
        self.noise_layer = tf.keras.layers.GaussianNoise(0.1)
        self.adv_dense2 = tf.keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_uniform())
        self.adv_outa = tf.keras.layers.Dense(num_actions,activation='relu',
                                          kernel_initializer=keras.initializers.he_uniform())
        self.adv_outb = tf.keras.layers.Dense(num_actions,activation='softmax',
                                          kernel_initializer=keras.initializers.he_uniform())
        if dueling:
            self.LSTM1 = tf.keras.layers.LSTM(64)
            self.v_dense = tf.keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
            self.v_out = tf.keras.layers.Dense(1, activation='softmax',kernel_initializer=keras.initializers.he_uniform())
            self.lambda_layer = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = tf.keras.layers.Add()
            
    def call(self, input):
        input_shape = input.shape
        x = self.noise_layer(input)
        x = self.conv1(x)
    
        x = self.BatchNormalization(x)

       

        x = self.pooling_layer2(x)
        
        x = self.conv2(x)
        #x = self.BatchNormalization2(x)
        x = self.pooling_layer1(x)
        #x = self.dropout_layer(x)
        x = self.dropout_layer(x)
        
        x = self.conv3(x)
##        x = self.BatchNormalization3(x)
##        x = self.dropout_layer(x)
        x = self.conv4(x)
        #x = self.BatchNormalization4(x)
        x = self.pooling_layer3(x)
       # x = self.dropout_layer(x)
        x = self.flatten1(x)
        adv = self.adv_dense1(x)
        #adv = self.dropout_layer(adv)
##        adv = self.adv_dense2(x)
##        adv = self.dropout_layer(adv)
        self.LSTM1(adv)
        if self.dueling:
            adv = self.adv_outa(adv)
        else:
            adv = self.adv_outb(adv)
        
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv

if(NEW):
    primary_network = DQModel(32, num_actions, True)
    target_network = DQModel(32, num_actions, True)
else:
    primary_network = tf.keras.models.load_model(STORE_PATH + "/checkpoints2/cp_primary_network")
    target_network = tf.keras.models.load_model(STORE_PATH + "/checkpoints2/cp_target_network")



primary_network.compile(optimizer=keras.optimizers.Nadam(learning_rate = .0006), loss=tf.keras.losses.Huber())
target_network.compile(optimizer=keras.optimizers.Adamax(learning_rate = .0006), loss=tf.keras.losses.Huber())

# make target_network = primary_network
for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
    t.assign(e)

class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = sum(n.value for n in (left, right) if n is not None)
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    
    return nodes[0], leaf_nodes

def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    
    if node.left.value >= value: 
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)

def update(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change

    if node.parent is not None:
        propagate_changes(change, node.parent)


class Memory(object):
    def __init__(self, size: int):
        self.size = size
        self.curr_write_idx = 0
        self.available_samples = 0
        self.buffer = [(np.zeros((POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1]),
                                dtype=np.float32), 0.0, 0.0, 0.0) for i in range(self.size)]
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)])
        self.frame_idx = 0
        self.action_idx = 1
        self.reward_idx = 2
        self.terminal_idx = 3
        self.beta = 0.4
        self.alpha = 0.6
        self.min_priority = 0.01

    def append(self, experience: tuple, priority: float):
        self.buffer[self.curr_write_idx] = experience
        self.update(self.curr_write_idx, priority)
        self.curr_write_idx += 1
        # reset the current writer position index if creater than the allowed size
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        if self.available_samples + 1 < self.size:
            self.available_samples += 1
        else:
            self.available_samples = self.size - 1

    def update(self, idx: int, priority: float):
        update(self.leaf_nodes[idx], self.adjust_priority(priority))

    def adjust_priority(self, priority: float):
        return np.power(abs(priority + self.min_priority), self.alpha)

    def sample(self, num_samples: int):
        sampled_idxs = []
        is_weights = []
        sample_no = 0
        while sample_no < num_samples:
            sample_val = np.random.uniform(0, self.base_node.value)
            samp_node = retrieve(sample_val, self.base_node)
            if NUM_FRAMES - 1 < samp_node.idx < self.available_samples - 1:
                sampled_idxs.append(samp_node.idx)
                p = samp_node.value / self.base_node.value
                is_weights.append((self.available_samples + 1) * p)
                sample_no += 1
        # apply the beta factor and normalise so that the maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)
        # now load up the state and next state variables according to sampled idxs
        states = np.zeros((num_samples, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                             dtype=np.float32)
        next_states = np.zeros((num_samples, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                            dtype=np.float32)
        actions, rewards, terminal = [], [], [] 
        for i, idx in enumerate(sampled_idxs):
            for j in range(NUM_FRAMES):
                states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 1][self.frame_idx][:, :, 0]
                next_states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 2][self.frame_idx][:, :, 0]
            actions.append(self.buffer[idx][self.action_idx])
            rewards.append(self.buffer[idx][self.reward_idx])
            terminal.append(self.buffer[idx][self.terminal_idx])
        actions = np.array(actions)
##        for a in actions:
##            print(a)
        #print(actions[0])
        return states, actions, np.array(rewards), next_states, np.array(terminal), sampled_idxs, is_weights

memory = Memory(10000)

def image_preprocess(image, new_size=(105, 80)):
    # convert to greyscale, resize and normalize the image
    np.mean(image, axis=2).astype(np.uint8)
    image = tf.image.rgb_to_grayscale(image)
##    image[image==color] = 0
    image = tf.image.resize(image, new_size)
    image = image / 255
    return image


ii = 0


def replayHumanAction(ii):
    a_replay = 0
    print("getting human action")
    with open("saved_memory.csv") as file:
        print("opened file")
        reader = csv.reader(file, delimiter=',', quotechar='"')
        header = next(reader)
        rows = [header] + [[row[0], float(row[1])] for row in reader if row]
        print(rows)
        for row in rows:
            print(row[0])
            a_replay = row[2]
   
    #ii=ii+1
    return a_replay

def replayHumanReward():
    
    r_replay = 0
    
    with open("saved_memory.csv") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        header = next(reader)
        rows = [header] + [[row[0], float(row[1])] for row in reader if row]
        for row in rows:
            print(row[1])
            r_replay = row[1]
    #ii+=1
    return r_replay

def replayHumanState():
    
    s_replay = 0
    
    with open("saved_memory.csv") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        header = next(reader)
        rows = [header] + [[row[0], float(row[1])] for row in reader if row]
        for row in rows:
            print(row[1])
            s_replay = row[0]
    #ii+=1
    return s_replay


def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        print("check1")
        if has_human_file:
            print("check2")
            if ii < row_count:
                print("check3")
                return replayHumanAction(ii)
        else:
            return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                           POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))


def update_network(primary_network, target_network):
    # update target network parameters slowly from primary network
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)


def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


def record_gif(frame_list, episode, fps=50):
    if(episode!=0):
        imageio.mimsave(STORE_PATH + f"/Berserk_EPISODE-{episode}.gif", frame_list, fps=fps) #duration=duration_per_frame)ation_per_frame)


def get_per_error(states, actions, rewards, next_states, terminal, primary_network, target_network):
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    # the action selection from the primary / online network
    prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
    # the q value for the prim_action_tp1 from the target network
    q_from_target = target_network(next_states)
    updates = rewards + (1 - terminal) * GAMMA * q_from_target.numpy()[:, prim_action_tp1]
    #print(updates.shape)
    target_q[:, actions] = updates
    # calculate the loss / error to update priorites
    error = [huber_loss(target_q[i, actions[i]] - prim_qt.numpy()[i, actions[i]]) for i in range(states.shape[0])]
    return target_q, error


def train(primary_network, memory, target_network):
    states, actions, rewards, next_states, terminal, idxs, is_weights = memory.sample(BATCH_SIZE)
    target_q, error = get_per_error(states, actions, rewards, next_states, terminal,
                                    primary_network, target_network)
    for i in range(len(idxs)):
        memory.update(idxs[i], error[i])
    loss = primary_network.train_on_batch(states, target_q, is_weights)
    return loss


num_episodes = 1000000
eps = MAX_EPSILON
render = True
train_writer = tf.summary.create_file_writer(STORE_PATH + "/DuelingQPERSI_{}".format(dt.datetime.now().strftime('%d%m%Y%H%M')))
steps = 0


for i in range(num_episodes):
    state = env.reset()
    state = image_preprocess(state)
    state_stack = tf.Variable(np.repeat(state.numpy(), NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                            POST_PROCESS_IMAGE_SIZE[1],
                                                                            NUM_FRAMES)))
    if(i==100):
        BATCH_SIZE = 16
        print(BATCH_SIZE)
    if(i==500):
        BATCH_SIZE = 32
        print(BATCH_SIZE)
    if(i==2000):
        BATCH_SIZE = 64
        print(BATCH_SIZE)
    if(i==10000):
        BATCH_SIZE = 128
        print(BATCH_SIZE)
    if(i==100000):
        BATCH_SIZE = 256
        print(BATCH_SIZE)
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    if i % GIF_RECORDING_FREQ == 0:
        frame_list = []
    while True:
        if render:
            env.render()
        action = choose_action(state_stack, primary_network, eps, steps)
        ii+=1
        #print(action)
        next_state, reward, done, info = env.step(action)
        #print()
        tot_reward += reward
        if i % GIF_RECORDING_FREQ == 0:
            frame_list.append(tf.cast(tf.image.resize(next_state, (480, 320)), tf.uint8).numpy())
        next_state = image_preprocess(next_state)
        old_state_stack = state_stack
        state_stack = process_state_stack(state_stack, next_state)

        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory, target_network)
            update_network(primary_network, target_network)
            _, error = get_per_error(tf.reshape(old_state_stack, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                                  POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)),
                                     np.array([action]), np.array([reward]), 
                                     tf.reshape(state_stack, (1, POST_PROCESS_IMAGE_SIZE[0], 
                                                              POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)), np.array([done]),primary_network, target_network)


            # store in memory
            if has_human_file:
                if ii < row_count:
                    memory.append((replayHumanState(), action, replayHumanReward(ii), done), error[0])
            else:
            
                memory.append((next_state, action, reward, done), error[0])

        else:
            loss = -1
            # store in memory - default the priority to the reward
            if has_human_file:
                if ii < row_count:
                    memory.append((replayHumanState(), action, replayHumanReward(), done), reward)
            else:
                memory.append((next_state, action, reward, done), reward)
            
        avg_loss += loss

        # linearly decay the eps and PER beta values
        if steps > DELAY_TRAINING:
            eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                  (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                MIN_EPSILON
            beta = MIN_BETA + ((steps - DELAY_TRAINING) / BETA_DECAY_ITERS) * \
                  (MAX_BETA - MIN_BETA) if steps < BETA_DECAY_ITERS else \
                MAX_BETA
            memory.beta = beta
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print("Episode: {}, Reward: {}, avg loss: {:.5f}, eps: {:.3f}".format(i, tot_reward, avg_loss, eps))
                with open(STORE_PATH+"log.txt", "a") as file1: 
                    # Writing data to a file 
                    file1.write("Episode: {}, Reward: {}, avg loss: {:.5f}, eps: {:.3f}".format(i, tot_reward, avg_loss, eps)+"batch: {}".format(BATCH_SIZE)+"/n")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print("Pre-training...Episode: {}".format(i))
            if i % GIF_RECORDING_FREQ == 0:
                record_gif(frame_list, i, tot_reward)
            break
        
        cnt += 1
        if keyboard.is_pressed('c') and steps > DELAY_TRAINING:
            primary_network.save_weights(STORE_PATH+f"/weights-{i}/", overwrite=True)
    if i % MODEL_SAVE_FREQ == 0: # and i != 0:
        primary_network.save_weights(STORE_PATH + "/checkpoints2/cp_primary_network_episode_{}.ckpt".format(i), overwrite=True)
        target_network.save_weights(STORE_PATH + "/checkpoints2/cp_target_network_episode_{}.ckpt".format(i), overwrite=True)
        try:
            primary_network.build(input_shape)
            target_network.build(input_shape)
            primary_network.save(STORE_PATH + "/checkpoints2/cp_primary_network", save_format='tf',overwrite=True)
            target_network.save(STORE_PATH + "/checkpoints2/cp_target_network", save_format='tf',overwrite=True)
            print("Saved!")
        except:
            print("Failed to save :-(")
            pass
