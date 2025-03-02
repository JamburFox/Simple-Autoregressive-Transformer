import os
import argparse
import h5py
import html

def create_dataset(save_path: str, chunk_size: int):
    with h5py.File(save_path,'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        dataset = f.create_dataset("text_data", (0,), maxshape=(None,), dtype=dt, chunks=(chunk_size,))

def expand_dataset(f, text: str):
    dataset = f['text_data']
    num_samples = dataset.shape[0]
    dataset.resize(num_samples + 1, axis=0)
    dataset[-1] = text

#convert html to normal text, e.x. "<h1>test</h1>"" becomes "test"
def decode_characters(text: str):
    decoded_text = html.unescape(text)
    return decoded_text

#apply all text filters here
def filter_text(text: str):
    return decode_characters(text)

#extracts full text from buffer and returns text and new remaining buffer after hop
def read_text_buffer(text_buffer: str, window_size: int = 128, hop_size: int = 64):
    text = text_buffer[:window_size].lstrip()
    text = filter_text(text)

    #find last space to use as end as the following text may be a cutup word
    last_space = text.rfind(' ')
    if last_space == -1:
        text = text
    else:
        text = text[:last_space].strip()

    hop_space_index = text_buffer[:hop_size].lstrip().rfind(' ')
    if hop_space_index == -1:#no white space
        text_buffer = text_buffer[hop_size:]
    else:
        text_buffer = text_buffer[hop_space_index:].strip()

    return text, text_buffer

def build_dataset(text_file_path: str, h5py_path: str, window_size: int = 512, hop_size: int = 256):
    text_buffer = ""
    end_of_file = False
    buffer_index = 0

    with h5py.File(h5py_path,'a') as dataset_file:
        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            while not end_of_file:
                chunk = file.read(window_size)
                if not chunk:
                    end_of_file = True
                else:
                    text_buffer += chunk

                #expand dataset with chunks of window_size and return new buffer after hop
                while len(text_buffer) >= window_size:
                    text, text_buffer = read_text_buffer(text_buffer, window_size, hop_size)
                    expand_dataset(dataset_file, text)

                buffer_index += 1
                print(f"Current Buffer: {buffer_index}", end='\r', flush=True)
            print("")

        #read remaining data into dataset
        if len(text_buffer) > 0:
            expand_dataset(dataset_file, filter_text(text_buffer))
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the tokenizer model.')
    parser.add_argument('--text_file_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt"), help='Text input path')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.hdf5"), help='File save path')
    parser.add_argument('--window_size', type=int, default=512, help='sliding window size')
    parser.add_argument('--hop_size', type=int, default=256, help='window hop size')
    args = parser.parse_args()

    create_dataset(args.save_path, 2048)
    build_dataset(args.text_file_path, args.save_path, args.window_size, args.hop_size)

    #load and read a few sample to test that its working
    with h5py.File(args.save_path, 'r') as f:
        dataset = f['text_data']
        num_samples = dataset.shape[0]
        print(num_samples)
        for i in range(1, 10):
            sample = dataset[i].decode('utf-8')
            print(f"{i}:", sample)