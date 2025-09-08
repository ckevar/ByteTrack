import os
import cv2

def seqinfo_mot17(sequence_dir):
    info_filename = f"{sequence_dir}/seqinfo.ini"
    info_dict = None
    if os.path.exists(info_filename):
        with open(info_filename, 'r') as fd:
            line_splits = [l.split('=') for l in fd.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

    return info_dict

def get_image_filenames(image_dir):
    image_filenames = os.listdir(image_dir)
    image_filenames.sort()
    image_path = {
        int(f.split('.')[0]): f"{image_dir}/{f}"
        for f in image_filenames
    }
    return image_path
    

def gather_info_sequence(sequence_dir, data_type):
    update_ms = None
    image_size = None
    
    if "MOT17" == data_type: 
        image_dir = os.path.join(sequence_dir, "img1")
        info_ini = seqinfo_mot17(sequence_dir)

        if info_ini:
            update_ms = 1000 / int(info_ini["frameRate"])
       
    elif "KITTI" == data_type: 
        image_dir = sequence_dir
        update_ms = 1000 / 15 # 15 FPS
    else: 
        raise ValueError("--data_type only supports MOT17 or KITTI.\n")

    image_filenames = get_image_filenames(image_dir)

    # Image_size
    if len(image_filenames) > 0:

        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape

    seq_info = {
        "sequence_name" : os.path.basename(sequence_dir),
        "image_filenames" : image_filenames,
        "image_size" : image_size,
        "update_ms" : update_ms,
        "data_type" : data_type
    }
    return seq_info


def print_debug(txt):
    print(f"\n --> [DEBUG:] {txt}\n")

class SequenceLoader(object):
    def __init__(self, data_dir:str, 
                 data_type:str, 
                 experiment_name="sequence_loader_default", 
                 overwrite=False):
        self.cache_filename = ".cache_" + experiment_name
        if not os.path.isfile(self.cache_filename) or overwrite:
            fd = open(self.cache_filename, 'w')
            fd.close()

        self.seqs = os.listdir(data_dir)
        self.seqs.append(None)
        self.data_dir = data_dir
        self.out_dir = self.__mk_output_dir__(data_dir, experiment_name, data_type)
        self.data_type = data_type

    def __mk_output_dir__(self, data_dir, results_dir, data_type):
        if '/' == data_dir[-1]:
            data_dir = data_dir[:-1]

        dirs = data_dir.split('/')
        out_dir = '/'.join(dirs[:-1]) + f"/results/{results_dir}"

        try:
            os.mkdir(out_dir)

        except FileExistsError:
            print("\n [SEQUENCE LOADER - WARNING:] Files will either overwritten or experiment resume.\n")
        except Exception as e:
            raise e

        return out_dir

    def __cache__(self, query_seq, ack_seq):
        found_seq = True
        with open(self.cache_filename, 'r+') as fd:
            cache_list = fd.read()
            if not (query_seq is None):
                found_seq = query_seq in cache_list

            if not (ack_seq is None):
                fd.write(f"{ack_seq}\n")

        return found_seq

    def next_sequence(self):
        ack_seq = None
        data_dir = self.data_dir
        out_dir = self.out_dir
        data_type = self.data_type

        for seq_name in self.seqs:
            if self.__cache__(seq_name, ack_seq):
                ack_seq = None
                continue

            ack_seq = seq_name
            seq_path = f"{data_dir}/{seq_name}"
            seq_info = gather_info_sequence(seq_path, data_type)
            seq_info["output_file"] = f"{out_dir}/{seq_name}.txt"

            yield seq_info

# Test section
if "__main__" == __name__:
    import sys
    sequences = SequenceLoader(sys.argv[1], # data_dir
                               sys.argv[2]) # data_type

    for seq in sequences.next_sequence():
        print(f"seq name: {seq['sequence_name']}, seq len: {len(seq['image_filenames'])}, out: {seq['output_file']}")



