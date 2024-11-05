from time import sleep
import torch 
from tqdm import trange, tqdm
import pandas as pd
import logging
import io
from functools import partial
import datetime
import time
import torch.multiprocessing as mp
from vision_processes.image_patch import ImagePatch,llm_query
import os
import openai
from prog_env import ProgWrapper
import traceback
from rich import print

# openai.api_base = 'https://weterminateai.xyz/v1'
def MyCollate(batch):
    res = {k:[sample[k] for sample in batch] for k in batch[0].keys()}
    return res

def beijing(sec,what,return_type='tuple'):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    if return_type == 'tuple':
        return beijing_time.timetuple()
    if return_type == 'stamp':
        return beijing_time

def init_logging_config(timestamp):
    logging.Formatter.converter = beijing
    log_dir = os.path.join('logs',timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir,f'process_{"_".join(tags)}.log')
    logging.basicConfig(filename=log_path,filemode='w',format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger,log_dir

def init_save_path(tags,timestamp):
    save_path = f'results/viper_gqa_sample2000_{"_".join(tags)}'
    save_path=os.path.join(save_path,timestamp)
    return save_path

def init_prog_wrapper_mp(save_dir,log_dir,process_id,queues):
    prog_wrapper = ProgWrapper(
        queues=queues,
        log_dir=log_dir,
        save_path=save_dir,
        process_id=process_id,
        inject_html=True,
        code_check=True,
        function_check_list=['simple_query','find']
    )

    return prog_wrapper

def worker_init(queues_in_,queue_results_,usable_keys_,using_keys_,save_dir,log_dir):
    global prog_wrapper,queue_results,queues_in,usable_keys,using_keys
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]
    queues_in = queues_in_
    usable_keys = usable_keys_
    using_keys = using_keys_    
    queues = [queues_in, queue_results]
    prog_wrapper = init_prog_wrapper_mp(save_dir,log_dir,index_queue,queues)


def test_task(query,image_path,image,sample_id,label):
    global prog_wrapper,queue_results,queues_in,usable_keys,using_keys
    import utils.key_utils
    import vision_processes.vision_processes
    utils.key_utils.get_random_key = partial(utils.key_utils.get_random_key,usable_keys=usable_keys,using_keys=using_keys) 
    utils.key_utils.delete_key = partial(utils.key_utils.delete_key,using_keys=using_keys)   

    queues = [queues_in, queue_results]
    vision_processes.vision_processes.forward = partial(vision_processes.vision_processes.forward,queues=queues)

    try:
        res = prog_wrapper.forward(query = query,
                                image_path = image_path,
                                file_name=sample_id,
                                label=label,
                                to_json=True).get('res')
        state = 'success'
    except Exception as e:
        state = 'fail'
        # traceback.print_exc()
        res = traceback.format_exc()
        print(f'Sample {sample_id} error:{res}')
    print(f'Sample {sample_id} done')
    return sample_id,state,res


if __name__ == "__main__":

    from vision_processes.vision_processes import *
    from torch.utils.data import DataLoader
    from datatsets.datasets import MyDataset
    batch_size = 3
    start = 1565
    queues_results = [manager.Queue() for _ in range(batch_size)]
    dataset = MyDataset('gqa_sample_2000',start_sample=start)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=MyCollate)
    tags=["single_agent_new"]
    timestamp = beijing('a','b','stamp').strftime('%Y-%m-%d-%H:%M')
    main_logger,log_dir = init_logging_config(timestamp)
    save_dir = init_save_path(tags=tags,timestamp=timestamp)

    with mp.Pool(processes=batch_size,
                 initializer=worker_init,
                 initargs=(queues_in,queues_results,usable_keys,using_keys,save_dir,log_dir)) as pool:
        for batch in tqdm(dataloader):
            query = batch.get('query')
            image_path = batch.get('image_name')
            image = batch.get('image')
            sample_id = batch.get('sample_id')
            label = batch.get('answer')
            inputs = zip(query,image_path,image,sample_id,label)
            result = pool.starmap(partial(test_task), inputs)
            for sample_id,state,res in result:
                if state == 'fail':
                    main_logger.log(logging.ERROR,f'Sample {sample_id}:{state},{res}')
                else:
                    main_logger.log(logging.INFO,f'Sample {sample_id}:{state},{res}')

            


