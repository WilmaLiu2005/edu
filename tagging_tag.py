import httpx
import asyncio
import json
import time
import os
from tqdm import tqdm
import random
random.seed(47)

input_file_path_list = ['/Users/vince/undergraduate/KEG/edu/Data/log.jsonl']
output_path =  '/Users/vince/undergraduate/KEG/edu/Data/tag_2nd.jsonl'

url1 = "https://117.50.182.117:8443/logmodel/tag1"
url2 = "https://117.50.182.117:8443/logmodel/tag2"
url3 = "https://117.50.182.117:8443/logmodel/tag3"
num_concurrent_requests = 100
semaphore = asyncio.Semaphore(num_concurrent_requests)
client = httpx.AsyncClient(verify=False)
headers = {'Host':'api-eval-service.glm.ai', 'Content-Type': 'application/json'}

async def send_request(test_data):
    
    async def _get_tag(url, json_data):
        try:
            json_data['prompt'] = json_data['prompt'][:7000]
            response = await client.post(url, json=json_data, timeout=30, headers=headers)
            return response.json()

        except Exception as e:
            print(repr(e))
            return {"code":-100, "error": repr(e), "tag":None}
    
    test_data['tags'] = [None]*4
    try:
        input_data = {"prompt":test_data['prompt'].strip()}
        tag1 = await _get_tag(url1, input_data) 
        if tag1.get('tag', None):
            test_data['tags'][0] = tag1['tag']
            if tag1['tag'] != '其他':
                input_data2 = {"prompt":input_data['prompt'], "tag1":tag1['tag']}
                tag2 = await _get_tag(url2, input_data2)
                if tag2.get('tag', None) and tag2.get('code') != -100:
                    test_data['tags'][1] = tag2['tag']
                    if tag2['tag'] != '其他':
                        input_data3 = {"prompt":input_data['prompt'], "tag2":tag2['tag']}
                        tag3 = await _get_tag(url3, input_data3)
                        if tag3.get('tag', None) and tag3.get('code') != -100:
                            test_data['tags'][2] = tag3['tag']
                        else:
                            test_data['tags_error'] = "tag3 error:" + tag3.get('error', None)
                else:
                    test_data['tags_error'] = "tag2 error:" + tag2.get('error', None)
        else:
            test_data['tags_error'] = "tag1 error:" + tag1.get('error', 'none')
        return test_data
    except Exception as e:
        print(f"Request failed: {repr(e)}")
                
async def limited_concurrency_wrapper(coro, semaphore: asyncio.Semaphore):
    async with semaphore:
        return await coro

async def process_data_chunk(chunk, start_time):
    tasks = [asyncio.create_task(limited_concurrency_wrapper(send_request(test_data), semaphore)) for test_data in chunk]
    return await asyncio.gather(*tasks)

async def process_data(input_file_path_list, output_path, chunk_size=1000):
    start_time = time.time()  # Record the start time
    test_data_list = []
    if os.path.exists(output_path):
        output_data_prompt_set = set()
        with open(output_path, 'r') as file:
            output_data = [json.loads(line) for line in file]
            output_data_prompt_set =  set([data['prompt'] for data in output_data if len(data['tags']) > 0])
        for file_path in input_file_path_list:
            with open(file_path, 'r') as file:
                temp = [json.loads(line) for line in file]
                temp = [item for item in temp if item['prompt'] not in output_data_prompt_set]
                test_data_list.extend(temp)
    else:
        for file_path in input_file_path_list:
            with open(file_path, 'r') as file:
                temp_data = [json.loads(line) for line in file]
                # temp_data = random.sample(temp_data, min(200000, len(temp_data)))
                test_data_list.extend(temp_data)
    
    # test_data_list = [item for item in test_data_list if not item['input_img_list'] and not item['output_img_list']]
    chunks = [test_data_list[i:i + chunk_size] for i in range(0, len(test_data_list), chunk_size)]
    print(f"[Total number]: {len(test_data_list)}")
    print(f"[Total chunk]: {len(chunks)}")

    for chunk_index, chunk in tqdm(enumerate(chunks), ncols=100, desc= "[tagging]", total= len(chunks)):
        cur_time = time.perf_counter()
        print(cur_time)
        updated_test_data_chunk = await process_data_chunk(chunk, start_time)
        # Open the file in append mode and write the updated data
        with open(output_path, 'a') as file:
            for data in updated_test_data_chunk:
                if data:
                    file.write(json.dumps(data, ensure_ascii=False) + '\n')

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"\nAll requests completed. Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(process_data(input_file_path_list, output_path))
