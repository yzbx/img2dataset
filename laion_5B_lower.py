import argparse
import glob
import os
import os.path as osp
import shutil

from tqdm import tqdm

from img2dataset import download


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry', type=int, default=0)
    parser.add_argument('--processes', type=int ,default=16)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--root_dir', default='/mnt/download/dataset/laion5B', help='urls root dir')
    parser.add_argument('--out_dir', default='/mnt/download/laion-5B')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=3)
    parser.add_argument('--sub_dir', nargs="*", default=['laion1B-nolang', 'laion2B-en', 'laion2B-multi'])

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)

    for sub_dir in args.sub_dir:
        pfiles = glob.glob(osp.join(args.root_dir, sub_dir, '*.parquet')) + glob.glob(osp.join(args.root_dir, sub_dir, '*', '*.parquet'))
        pfiles.sort()

        rank_pfiles = pfiles[args.rank::args.world]
        output_dir = osp.join(args.out_dir, sub_dir, f'rank{args.rank}')
        os.makedirs(output_dir, exist_ok=True)

        log_file = osp.join(output_dir, f'log.txt')
        if osp.exists(log_file):
            with open(log_file, 'r') as fr:
                downloaded_pfiles = [line.strip() for line in fr.readlines()]
        else:
            downloaded_pfiles = []

        for pfile in tqdm(rank_pfiles):
            if pfile in downloaded_pfiles:
                print(f'skip downloaded pfile {pfile}')
                continue

            # laion5B/laion1B-nolang/part-00003-d6a94da9-d368-4d5b-9ab7-3f6d3c7abdb3-c000.snappy.parquet
            pid = osp.basename(pfile).split('-')[1]
            output_folder = osp.join(output_dir, pid)
            if osp.exists(output_folder):
                print(f'resume downloaded pfile {pfile} to {output_folder}')
            else:
                os.makedirs(output_folder, exist_ok=True)

            download(
                processes_count=args.processes,
                thread_count=args.threads,
                retries=args.retry,
                url_list = pfile,
                image_size=384,
                resize_only_if_bigger=True,
                resize_mode="keep_ratio",
                skip_reencode=True,
                output_folder=output_folder,
                output_format="webdataset",
                input_format="parquet",
                url_col="url",
                caption_col="text",
                enable_wandb=False,
                number_sample_per_shard=1000,
                distributor="multiprocessing",
                save_additional_columns=["NSFW","similarity","LICENSE"],
                oom_shard_count=6,
                incremental_mode='incremental'
            )

            with open(log_file, 'a') as fa:
                fa.write(pfile + '\n')
