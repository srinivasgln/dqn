import gym
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime as dt
import imageio
import keyboard
import os

STORE_PATH = 'C:\\TensorFlowBook\\TensorBoard'
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
GAMMA = 0.99
BATCH_SIZE = 256
POST_PROCESS_IMAGE_SIZE = (105, 84, 1)
DELAY_TRAINING = 500
NUM_FRAMES = 4
GIF_RECORDING_FREQ = 100
MODEL_SAVE_FREQ = 100


print(tf.config.experimental.list_physical_devices('GPU'))

def image_preprocess(image, new_size=(105, 84)):
    # convert to greyscale, resize and normalize the image
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, new_size)
    image = image / 255
    return image


env = gym.make("KungFuMaster-v0")
num_actions = env.action_space.n


class DQNModel(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQNModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu',use_bias=False)
        self.conv2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu',use_bias=False)
        self.flatten = keras.layers.Flatten()
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,activation='relu',
                                          kernel_initializer=keras.initializers.he_normal())

    def call(self, input):
        
        x = self.conv1(input)
        
        
        x = self.conv2(x)
        x = self.flatten(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        
        return adv

primary_network = DQNModel(1024, num_actions, True)




#compile models
primary_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())



class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._actions = np.zeros(max_memory, dtype=np.int32)
        self._rewards = np.zeros(max_memory, dtype=np.float32)
        self._frames = np.zeros((POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], max_memory), dtype=np.float32)
        self._terminal = np.zeros(max_memory, dtype=np.bool)
        self._i = 0

    def add_sample(self, frame, action, reward, terminal):
        self._actions[self._i] = action
        self._rewards[self._i] = reward
        self._frames[:, :, self._i] = frame[:, :, 0]
        self._terminal[self._i] = terminal
        if self._i % (self._max_memory - 1) == 0 and self._i != 0:
            self._i = BATCH_SIZE + NUM_FRAMES + 1
        else:
            self._i += 1

    def sample(self):
        if self._i < BATCH_SIZE + NUM_FRAMES + 1:
            raise ValueError("Not enough memory to extract a batch")
        else:
            rand_idxs = np.random.randint(NUM_FRAMES + 1, self._i, size=BATCH_SIZE)
            states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                             dtype=np.float32)
            next_states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
                             dtype=np.float32)
            for i, idx in enumerate(rand_idxs):
                states[i] = self._frames[:, :, idx - 1 - NUM_FRAMES:idx - 1]
                next_states[i] = self._frames[:, :, idx - NUM_FRAMES:idx]
            return states, self._actions[rand_idxs], self._rewards[rand_idxs], next_states, self._terminal[rand_idxs]
memory = Memory(10000)
def policy(state, t):
    p = np.array([q[(state,x)]/t for x in range(env.action_space.n)])
    prob_actions = np.exp(p) / np.sum(np.exp(p))
    cumulative_probability = 0.0
    choice = random.uniform(0,1)
    for a,pr in enumerate(prob_actions):
        cumulative_probability += pr
        if cumulative_probability > choice:
            return a

def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                           POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))




def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


def train(primary_network, memory, target_network=None):
    states, actions, rewards, next_states, terminal = memory.sample()
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    valid_idxs = terminal != True
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = target_network(next_states)
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates
    loss = primary_network.train_on_batch(states, target_q)
    return loss

def record_gif(frame_list, episode, fps=30):
    imageio.mimsave(STORE_PATH + f"/KUNGFU_EPISODE-{episode}.gif", frame_list, fps=fps) #duration=duration_per_frame)

num_episodes = 1000000
eps = MAX_EPSILON
render = True
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQSI_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
steps = 0
for i in range(num_episodes):
    state = env.reset()
    state = image_preprocess(state)
    state_stack = tf.Variable(np.repeat(state.numpy(), NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                            POST_PROCESS_IMAGE_SIZE[1],
                                                                            NUM_FRAMES)))


    
    
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    if i % GIF_RECORDING_FREQ == 0:
        frame_list = []
    while True:
        if render:
            env.render()
        action = choose_action(state_stack, primary_network, eps, steps)
        next_state, reward, done, info = env.step(action)
        tot_reward += reward
        if i % GIF_RECORDING_FREQ == 0:
            frame_list.append(tf.cast(tf.image.resize(next_state, (480, 320)), tf.uint8).numpy())
        next_state = image_preprocess(next_state)
        state_stack = process_state_stack(state_stack, next_state)
        # store in memory
        memory.add_sample(next_state, action, reward, done)

        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory, target_network if double_q else None)
            update_network(primary_network, target_network)
        else:
            loss = -1
        avg_loss += loss

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                  (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                MIN_EPSILON
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {tot_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}")
            if i % MODEL_SAVE_FREQ == 0:
                primary_network.save_weights(STORE_PATH+f"/weights-{i}/", overwrite=True)

            if i % GIF_RECORDING_FREQ == 0:
                record_gif(frame_list, i)
            break

        cnt += 1
        if keyboard.is_pressed('c') and steps > DELAY_TRAINING:
            primary_network.save_weights(STORE_PATH+f"/KungFuMaster-{i}/", overwrite=True)
        if i % MODEL_SAVE_FREQ == 0: # and i != 0:
            primary_network.save_weights(STORE_PATH + "/checkpoints2/cp_primary_network_episode_{}.ckpt".format(i), overwrite=True)
            target_network.save_weights(STORE_PATH + "/checkpoints2/cp_target_network_episode_{}.ckpt".format(i), overwrite=True)
            try:
                param = primary_network.count_params()
                print(param)
                input_shape= (num_actions,105,80,4)
                primary_network.build(input_shape)
                primary_network.save(STORE_PATH + "/checkpoints2/cp_primary_network", save_format='tf',overwrite=True)
                print("Saved!")
            except:
                print("Failed to save :-(")
env.close()
