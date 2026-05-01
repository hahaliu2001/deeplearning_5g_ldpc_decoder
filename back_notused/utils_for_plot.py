import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import os
import pickle

def process_floating_point_simu_result():
    """
    
    """
    
    #1. read test_bler_list,test_config_list,snr_db_list from the file
    file_name = "simulate_result/LDPC_decoder_simulation_results.csv"
    df = pd.read_csv(file_name)
    return df

def process_int16_fixed_point_simu_result():
    """
    sim_int16_c_ldpc_decoder_performance.py is to run LDPC simulation test for two algorithms:
    (1)python-based floating point layered min-sum (2) c-based 16bits fixed point layered min-sum
    the result is in  simulate_result/int16_c_ldpc_decoder_result.pkl
    """
    
    #1. read test_bler_list,test_config_list,snr_db_list from the file
    file_name = "simulate_result/int16_c_ldpc_decoder_result.pkl"
    with open(file_name, 'rb') as f:
        test_bler_list,test_config_list,snr_db_list = pickle.load(f)
    
    #2, generate pandas dataframe 
    cols = ['rate', 'payload', 'L','snr', 'f_BLER', '16bits_BLER']
    df = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        for j in range(len(snr_db_list)):
            snr_db = snr_db_list[j]
            f_bler = test_bler_list[i][j][0]
            fixed_bler = test_bler_list[i][j][1]
            df.loc[len(df)] = [datarate, payload_size, L, snr_db, f_bler, fixed_bler] 
    
    df['payload']=df['payload'].astype('int64')
    df['L']=df['L'].astype('int64')

    #3. another dataframe
    cols = ['rate', 'payload', 'L'] + ['BLER']*len(snr_db_list)
    df2 = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        
        line = [datarate, payload_size, L]
        line.extend(x for x in test_bler_list[i])
        df2.loc[len(df2)] = line
    return df, df2

def process_int8_fixed_point_simu_result():
    """
    sim_int8_c_ldpc_decoder_performance.py is to run LDPC simulation test for two algorithms:
    (1)python-based floating point layered min-sum (2) c-based 8bits fixed point layered min-sum
    the result is in  simulate_result/int8_c_ldpc_decoder_result.pkl
    """
    
    #1. read test_bler_list,test_config_list,snr_db_list from the file
    file_name = "simulate_result/int8_c_ldpc_decoder_result.pkl"
    with open(file_name, 'rb') as f:
        test_bler_list,test_config_list,snr_db_list = pickle.load(f)
    
    #2, generate pandas dataframe 
    cols = ['rate', 'payload', 'L','snr', 'pf_BLER','pt_BLER', '8bits_BLER1', '8bits_BLER2', '8bits_BLER3']
    df = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        for j in range(len(snr_db_list)):
            snr_db = snr_db_list[j]
            tlist  = [datarate, payload_size, L, snr_db]
            tlist.extend(test_bler_list[i][j].tolist())
            df.loc[len(df)] = tlist
    
    df['payload']=df['payload'].astype('int64')
    df['L']=df['L'].astype('int64')

    #3. another dataframe
    cols = ['rate', 'payload', 'L'] + ['BLER']*len(snr_db_list)
    df2 = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        
        line = [datarate, payload_size, L]
        line.extend(x for x in test_bler_list[i])
        df2.loc[len(df2)] = line
    return df, df2

def process_int8_int16_L_fixed_point_simu_result():
    """
    sim_int8_c_ldpc_decoder_performance.py is to run LDPC simulation test for two algorithms:
    (1)python-based floating point layered min-sum (2) c-based 8bits fixed point layered min-sum
    the result is in  simulate_result/int8_c_ldpc_decoder_result.pkl
    """
    
    #1. read test_bler_list,test_config_list,snr_db_list from the file
    file_name = "simulate_result/int16_int8_L_c_ldpc_decoder_result.pkl"
    with open(file_name, 'rb') as f:
        test_bler_list,test_config_list,snr_db_list = pickle.load(f)
    
    #2, generate pandas dataframe 
    cols = ['rate', 'payload', 'L','snr', 'i8_L8','i16_L8','i8_L16','i16_L16','i8_L24','i16_L24','i8_L32','i16_L32',]
    df = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        for j in range(len(snr_db_list)):
            snr_db = snr_db_list[j]
            tlist  = [datarate, payload_size, L, snr_db]
            tlist.extend(test_bler_list[i][j].tolist())
            df.loc[len(df)] = tlist
    
    df['payload']=df['payload'].astype('int64')
    df['L']=df['L'].astype('int64')

    #3. another dataframe
    cols = ['rate', 'payload', 'L'] + ['BLER']*len(snr_db_list)
    df2 = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        
        line = [datarate, payload_size, L]
        line.extend(x for x in test_bler_list[i])
        df2.loc[len(df2)] = line
    return df, df2

def process_int8_BER_fixed_point_simu_result():
    """
    sim_int8_c_ldpc_decoder_performance.py is to run LDPC simulation test for two algorithms:
    (1)python-based floating point layered min-sum (2) c-based 8bits fixed point layered min-sum
    the result is in  simulate_result/int8_c_ldpc_decoder_result.pkl
    """
    
    #1. read test_bler_list,test_config_list,snr_db_list from the file
    file_name = "simulate_result/int16_int8_BER_c_ldpc_decoder_result.pkl"
    with open(file_name, 'rb') as f:
        test_bler_list,test_config_list,snr_db_list,test_ber_list = pickle.load(f)
    
    #2, generate pandas dataframe 
    cols = ['rate', 'payload', 'L','snr', '8bits_BLER', '16bits_BLER', '8bits_BER', '16bits_BER']
    df = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        for j in range(len(snr_db_list)):
            snr_db = snr_db_list[j]
            tlist  = [datarate, payload_size, L, snr_db]
            tlist.extend(test_bler_list[i][j].tolist())
            tlist.extend(test_ber_list[i][j].tolist())
            df.loc[len(df)] = tlist
    
    df['payload']=df['payload'].astype('int64')
    df['L']=df['L'].astype('int64')

    #3. another dataframe
    cols = ['rate', 'payload', 'L'] + ['BLER']*len(snr_db_list) + ['BER']*len(snr_db_list)
    df2 = pd.DataFrame(columns=cols)

    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = int(test_config_list[i][1])
        L = int(test_config_list[i][2]['L'])
        
        line = [datarate, payload_size, L]
        line.extend(x for x in test_bler_list[i])
        line.extend(x for x in test_ber_list[i])
        df2.loc[len(df2)] = line
    return df, df2

def get_ldpc_test_result_from_pickle(file_name_list, algo_list, snr_db_list):

    test_config_list_list = []
    test_bler_list_list = []
    for file_name in file_name_list:
        with open(file_name, 'rb') as f:
            test_config_list, test_bler_list = pickle.load(f)
            test_config_list_list.append(test_config_list)
            test_bler_list_list.append(test_bler_list)

    #generate pandas dataframe,
    # the format of the dataframe should be like this:
    # get datarate, payload size,L from test_config_list and get N blers from test_bler_list, then combine them into a dataframe
    #each line of the dataframe should be like this:
    # datarate, payload size, L, ,snr_db 0, bler 0
    # datarate, payload size, L, ,snr_db 1, bler 1
    # datarate, payload size, L, ,snr_db 2, bler 2

    
    #add columns name to the dataframe
    columns = ['datarate', 'payload_size', 'L', 'snr_db'] + [algo + '_bler' for algo in algo_list]
    df = pd.DataFrame(columns=columns)

    test_config_list = test_config_list_list[0]
    for i in range(len(test_config_list)):
        datarate = test_config_list[i][0]
        payload_size = test_config_list[i][1]
        L = test_config_list[i][2]['L']
        for j in range(len(snr_db_list)):
            snr_db = snr_db_list[j]
            bler = test_bler_list[i][j]
            #df.loc[len(df)] = {'datarate': datarate, 'payload_size': payload_size, 'L': L, 'snr_db': snr_db, algo + '_bler': bler}
            df.loc[len(df)] = [datarate, payload_size, L, snr_db] + [test_bler_list_list[k][i][j] for k in range(len(algo_list))]
    return df

if __name__ == "__main__":
    # example usage
    df = process_floating_point_simu_result()
    df,df2 = process_int8_int16_L_fixed_point_simu_result()
    df, df2 = process_int8_BER_fixed_point_simu_result()
    df,df2 = process_int8_fixed_point_simu_result()

    process_int16_fixed_point_simu_result()

    results_folder = "out"
    file_name_list = [
        os.path.join(results_folder,"LDPC_decoder_simulation_results_BP.pkl"),
        os.path.join(results_folder,"LDPC_decoder_simulation_results_layered-min-sum.pkl"),
        os.path.join(results_folder,"LDPC_decoder_simulation_results_min-sum.pkl")
        ]
    
    algo_list = ["BP", "layered-min-sum", "min-sum"]
    
    snr_db_list = np.arange(-3, -1, 0.5).tolist()

    df = get_ldpc_test_result_from_pickle(file_name_list, algo_list, snr_db_list)
    print(df.head())
    pass