import random
import tensorflow as tf
import numpy as np
from ModelV3 import Agent, Buffer, AgentRestore
from nistrng import *
import csv
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA512
import serial
import time
from hashlib import md5
from base64 import b64decode
from base64 import b64encode
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad
import bchlib
from ast import literal_eval
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
    BCH_POLYNOMIAL = 8219
    BCH_BITS = 10
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    epoch = 15000
    bob = AgentRestore(name='./R1/Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, epoch=epoch)
    # Discount factor for future rewards
    reward_epoch = 0
    receiver = serial.Serial("COM11", 9600)
    idx=1
    while True:
        return_val = receiver.readline().decode().strip()
        return_val = return_val.split(';')
        current_time = str(return_val[0])
        ecc = literal_eval(return_val[1])
        ecc = bytearray(ecc)
        received_info = str(return_val[2])

        df = pd.read_csv('uP428.csv')
        for i in range(len(df)):
            row = df.values[i][0]
            row = str(row).split(';')[0:6]
            time_idx = row[0]
            if current_time == time_idx:
                break
            #idx += 1

        pressure = get_physical_dynamics(df.values[idx:idx + attribute_length, :], idx)
        print(f'pressure:{pressure}')
        obs_bob = pressure
        act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
        key_bob, k_action_b = get_keys(obs_bob, act_bob)
        key_bob_bytearray = bytearray(key_bob)

        bitflips = bch.decode_inplace(key_bob_bytearray, ecc)

        if bitflips <= BCH_BITS and bitflips >= 0:
            key_bob_final = bch.decode(key_bob, ecc)
            key_bob_final = key_bob_final[1]
            key_bob_final = np.array(key_bob_final)
            key_bob_final = np.reshape(key_bob_final, (-1, 4))
            key_bob_final = key_bob_final[:, 0]
            key_bob = ''.join(str(key_bob_final[i]) for i in range(len(key_bob_final)))
            salt = bytes.fromhex("10101010101010101010")
            # Convert the integer (idx) to its binary representation
            idx_binary = bin(int(idx))[2:]  # [2:] to remove the '0b' prefix
            # Convert the binary string to bytes
            idx_binary_bytes = int(idx_binary, 2).to_bytes((len(idx_binary) + 7) // 8, byteorder='big')
            salt = salt + idx_binary_bytes
            key_bob_stretched = (PBKDF2(key_bob, salt, 16, count=10000, hmac_hash_module=SHA512))
            key_bob_binary = b''.join(format(byte, '08b').encode('utf-8') for byte in key_bob_stretched)
            key_bob_binary_str = key_bob_binary.decode('utf-8')  # Convert bytes to string
            key_bob = key_bob_binary_str
            decrypted_info = AESCipher(key_bob).decrypt(received_info).decode('utf-8')
            idx = idx + attribute_length
            print("Decrypted Information:", decrypted_info)
            print("Success!")
            print("Time:", current_time)
            print("Index:", idx)
            print("Received Info:", return_val)
            print("Bob Key:", key_bob)
            #print("Original Bob Key:", key_bob_ori)
            print("Decrypted Data:", decrypted_info)
            print("Wrong Bits:", bitflips)
            print("\n")
        else:
            print("Failed!")
            print("Wrong Bits:", bitflips)
            print("Time:", current_time)
            print("Index:", idx)
            print("Received Info:", return_val)
            print("Bob Key:", key_bob)
            print("\n")

        time.sleep(4)


if __name__ == '__main__':
    run()
