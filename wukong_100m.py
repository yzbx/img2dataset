import argparse
import glob
import os
import os.path as osp
import shutil

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel
from tqdm import tqdm

from img2dataset import download


def create_spark_session():
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client")
        .config("spark.executor.cores", "16")
        .config("spark.driver.memory", "16G")
        .config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.executor.memory", "16G") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .appName("spark-wukong")
        .getOrCreate()
    )
    return spark

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry', type=int, default=0)
    parser.add_argument('--processes', type=int, default=16)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--root_dir', default='/mnt/download/wukong100M')
    parser.add_argument('--out_dir', default='/mnt/download/wukong100M/images')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=2)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    for sub_dir in ['wukong_release']:
        pfiles = glob.glob(osp.join(args.root_dir, sub_dir, '*.csv'))
        pfiles.sort()

        rank_pfiles = pfiles[args.rank::args.world]
        output_dir = osp.join(args.out_dir)
        os.makedirs(output_dir, exist_ok=True)

        log_file = osp.join(args.out_dir, f'log_rank{args.rank}.txt')
        if osp.exists(log_file):
            with open(log_file, 'r') as fr:
                downloaded_pfiles = [line.strip() for line in fr.readlines()]
        else:
            downloaded_pfiles = []

        for pfile in tqdm(rank_pfiles):
            if pfile in downloaded_pfiles:
                print(f'skip downloaded pfile {pfile}')
                continue

            # wukong_100m_154.csv
            pid = osp.basename(pfile).split('_')[2].split('.')[0]
            output_folder = osp.join(output_dir, pid)
            os.makedirs(output_folder, exist_ok=True)

            spark = create_spark_session()
            download(
                processes_count=args.processes,
                thread_count=args.threads,
                retries=args.retry,
                url_list = pfile,
                image_size = 1024,
                resize_only_if_bigger=True,
                resize_mode="keep_ratio",
                skip_reencode=True,
                output_folder=output_folder,
                output_format="webdataset",
                input_format="csv",
                url_col="url",
                caption_col="caption",
                enable_wandb=False,
                number_sample_per_shard=1000,
                distributor="pyspark",
                save_additional_columns=None,
                oom_shard_count=6,
            )
            spark.stop()

            with open(log_file, 'a') as fa:
                fa.write(pfile + '\n')
