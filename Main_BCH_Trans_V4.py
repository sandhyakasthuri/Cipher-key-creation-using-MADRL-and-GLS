import tensorflow as tf
import numpy as np
from ModelV3 import Agent, Buffer, AgentRestore
import csv
import serial
import time
from hashlib import md5
from base64 import b64decode
from base64 import b64encode
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA512
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad
import bchlib
from datetime import datetime
import pandas as pd

tf.random.set_seed(1)
np.random.seed(1)
key_length = 10
attribute_length = key_length
num_agent = 2
action_bound = 1


class AESCipher:
    def __init__(self, key):
        self.key = md5(key.encode('utf8')).digest()

    def encrypt(self, data):
        iv = get_random_bytes(AES.block_size)
        self.cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return b64encode(iv + self.cipher.encrypt(pad(data.encode('utf-8'), AES.block_size)))

    def decrypt(self, data):
        raw = b64decode(data)
        self.cipher = AES.new(self.key, AES.MODE_CBC, raw[:AES.block_size])
        return unpad(self.cipher.decrypt(raw[AES.block_size:]), AES.block_size)


def get_keys(obs, action):
    feature_vector = np.zeros(len(action))
    key = np.zeros((len(feature_vector), 2))
    for i in range(len(feature_vector)):
        if action[i] >= 0:
            feature_vector[i] = 1
        if feature_vector[i] == 1:
            if obs[i] >= np.mean(obs) + np.std(obs):
                key[i][0] = 1
                key[i][1] = 1
            elif np.mean(obs) + np.std(obs) > obs[i] > np.mean(obs) - np.std(obs):
                key[i][0] = 0
                key[i][1] = 1
            elif obs[i] <= np.mean(obs) - np.std(obs):
                key[i][0] = 1
                key[i][1] = 0
    key = np.array(key, dtype=int)
    key = np.reshape(key, (1, -1))[0]
    return key, feature_vector


def get_physical_dynamics(data, id):
    pressure = np.zeros(attribute_length)
    id = int(id)
    if data.shape[1] > 1:
        for i in range(len(data)):
            pressure[i] = float(data[i][2])
    else:
        for i in range(len(data)):
            row = str(data[i]).split(';')
            pressure[i] = float(row[2])
    try:
        min_pressure = min(pressure)
        max_pressure = max(pressure)
        if max_pressure - min_pressure != 0:
            pressure = (pressure - min_pressure) / (max_pressure - min_pressure)
    except:
        pressure[:] = np.nan
    return pressure


def run():
    # create a bch object
    BCH_POLYNOMIAL = 8219
    BCH_BITS = 10
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    epoch = 15000
    alice = AgentRestore(name='./R1/Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent,
                         epoch=epoch)

    sender = serial.Serial("COM9", 9600)
    idx = 1
    while True:
        # step = 11
        data = input('Input Text Message: ')
        # current_time = input('Input Time(H:M:S): ')
        current_time = datetime.now().strftime("%H:%M:%S")
        # current_time = '14:23:09'
        # data = 'hehe'
        time.sleep(15)
        df = pd.read_csv('uP428.csv')
        for i in range(len(df)):  # 2054
            row = df.values[i][0]
            row = str(row).split(';')[0:6]
            time_idx = row[0]
            if current_time == time_idx:
                # locate here
                break
            # idx += 1
        print(idx)
        pressure = get_physical_dynamics(df.values[idx:idx + attribute_length, :], idx)  # 2054:2064,:2
        print(f'pressure:{pressure}')
        # alice_attribute = pressure
        obs_alice = pressure
        # obs_alice = np.concatenate((alice_attribute, alice_zero_count, alice_one_count), axis=0)
        act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
        key_alice, k_action_a = get_keys(obs_alice, act_alice)
        key_alice_bytearray = bytearray(key_alice)
        key_alice_ecc = bch.encode(key_alice_bytearray)
        key_alice_ecc_temp = [i for i in key_alice_ecc]
        key_alice = ''.join(str(key_alice[i]) for i in range(len(key_alice)))
        salt = bytes.fromhex("10101010101010101010")
        # Convert the integer (idx) to its binary representation
        idx_binary = bin(int(idx))[2:]  # [2:] to remove the '0b' prefix
        # Convert the binary string to bytes
        idx_binary_bytes = int(idx_binary, 2).to_bytes((len(idx_binary) + 7) // 8, byteorder='big')
        salt = salt + idx_binary_bytes
        key_alice_stretched = PBKDF2(key_alice, salt, 16, count=10000, hmac_hash_module=SHA512)
        key_alice_binary = b''.join(format(byte, '08b').encode('utf-8') for byte in key_alice_stretched)
        key_alice_binary_str = key_alice_binary.decode('utf-8')  # Convert bytes to string
        key_alice = key_alice_binary_str
        encrypted_info = AESCipher(key_alice).encrypt(data).decode('utf-8')
        send_info = str.encode(str(current_time) + ';' + str(key_alice_ecc_temp) + ';' + encrypted_info + '\n')
        idx = idx + attribute_length
        print(idx)
        sender.write(send_info)
        print('Time: ', current_time, idx)
        print('Original text: ', data)
        print('Alice key: ', key_alice)
        print('Encrypted info: ', encrypted_info)
        print('Sent info', send_info)
        print('\n')

        # # Debugging information
        # print("Debugging information:")
        # print("Obs Alice:", obs_alice)
        # print("Act Alice:", act_alice)
        # print("Key Alice:", key_alice)
        # print("Encrypted Info:", encrypted_info)
        # print("Send Info:", send_info)
        # print("\n")
        # time.sleep(4)


if __name__ == '__main__':
    run()
