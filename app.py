import base64
import time
from ast import literal_eval
from base64 import b64decode, b64encode
from hashlib import md5
import serial
import tensorflow as tf
import numpy as np
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA512
from flask import Flask, request, render_template, session
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad
from datetime import datetime
import pandas as pd
import bchlib
import matplotlib.pyplot as plt
import numpy as np
import math
from ModelV3 import AgentRestore

app = Flask(__name__)
tf.random.set_seed(1)
np.random.seed(1)

key_length = 10
attribute_length = key_length
num_agent = 2
action_bound = 1
app.secret_key = '101'


class AESCipher:
    def __init__(self, key):
        self.key = md5(key.encode('utf8')).digest()

    def encrypt(self, data):
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return b64encode(iv + cipher.encrypt(pad(data.encode('utf-8'), AES.block_size))).decode('utf-8')

    def decrypt(self, data):
        raw = b64decode(data.encode('utf-8'))
        cipher = AES.new(self.key, AES.MODE_CBC, raw[:AES.block_size])
        return unpad(cipher.decrypt(raw[AES.block_size:]), AES.block_size).decode('utf-8')


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


def get_physical_dynamics(data, idx):
    pressure = np.zeros(attribute_length)

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


def calculate_entropy(key):
    num_possible_combinations = 2 ** len(key)
    return len(key) * math.log2(num_possible_combinations)


def plot_entropy_graph(key_name, entropies):
    plt.hist(entropies, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Entropy Distribution of {key_name} Key')
    plt.grid(True)
    plt.show()


@app.route('/', methods=['GET', 'POST'])
def process_data():
    encrypted_info = ''
    key_alice = ''
    decrypted_info = ''
    key_bob = ''
    BCH_POLYNOMIAL = 8219
    BCH_BITS = 10
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    epoch = 15000
    alice = AgentRestore(name='./R1/Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent,
                         epoch=epoch)
    bob = AgentRestore(name='./R1/Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, epoch=epoch)

    sender = serial.Serial("COM9", 9600)
    receiver = serial.Serial("COM11", 9600)
    # Get the current value of idx from the session, default to 1 if not set
    idx = session.get('idx', 1)
    if request.method == 'POST':
        if 'message' in request.form:
            # Encryption
            data = request.form['message']
            current_time = datetime.now().strftime("%H:%M:%S")
            time.sleep(15)
            df = pd.read_csv('uP428.csv')

            for i in range(len(df)):
                row = df.values[i][0]
                row = str(row).split(';')[0:6]
                time_idx = row[0]
                if current_time == time_idx:
                    break
            pressure = get_physical_dynamics(df.values[idx:idx + attribute_length, :], idx)
            obs_alice = pressure
            print(obs_alice)
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            key_alice, _ = get_keys(obs_alice, act_alice)
            key_alice_bytearray = bytearray(key_alice)
            key_alice_ecc = bch.encode(key_alice_bytearray)
            key_alice_ecc_temp = [i for i in key_alice_ecc]
            key_alice = ''.join(str(key_alice[i]) for i in range(len(key_alice)))
            print(key_alice)
            key_alice.encode('utf-8')
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
            encrypted_info = AESCipher(key_alice).encrypt(data)
            send_info = str.encode(str(current_time) + ';' + str(key_alice_ecc_temp) + ';' + encrypted_info + '\n')
            sender.write(send_info)
            idx = idx + attribute_length

            # Calculate entropy for Alice's key
            entropy_alice = calculate_entropy(key_alice)

            # Plot the entropy graph for Alice's key
            plot_entropy_graph("Alice's", entropy_alice)

            # Decryption
            return_val = receiver.readline().decode().strip()
            return_val = return_val.split(';')
            current_time = str(return_val[0])
            ecc = literal_eval(return_val[1])
            ecc = bytearray(ecc)
            received_info = return_val[2]

            df = pd.read_csv('uP428.csv')
            for i in range(len(df)):
                row = df.values[i][0]
                row = str(row).split(';')[0:6]
                time_idx = row[0]
                if current_time == time_idx:
                    break

            pressure = get_physical_dynamics(df.values[idx:idx + attribute_length, :], idx)
            obs_bob = pressure
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            key_bob, _ = get_keys(obs_bob, act_bob)
            key_bob_bytearray = bytearray(key_bob)
            bitflips = bch.decode_inplace(key_bob_bytearray, ecc)

            if bitflips <= BCH_BITS and bitflips >= 0:
                key_bob_final = bch.decode(key_bob, ecc)
                key_bob_final = key_bob_final[1]
                key_bob_final = np.array(key_bob_final)
                key_bob_final = np.reshape(key_bob_final, (-1, 4))
                key_bob_final = key_bob_final[:, 0]
                key_bob = ''.join(str(key_bob_final[i]) for i in range(len(key_bob_final)))
                print(key_bob)
                key_bob_stretched = (PBKDF2(key_bob, salt, 16, count=10000, hmac_hash_module=SHA512))
                key_bob_binary = b''.join(format(byte, '08b').encode('utf-8') for byte in key_bob_stretched)
                key_bob_binary_str = key_bob_binary.decode('utf-8')  # Convert bytes to string
                key_bob = key_bob_binary_str
                decrypted_info = AESCipher(key_bob).decrypt(received_info)
            session['idx'] = idx + attribute_length

    return render_template('index.html', encrypted_data=encrypted_info, key_alice=key_alice,
                           decrypted_data=decrypted_info, key_bob=key_bob)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
