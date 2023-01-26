"""
A script for running run_summarization.py on the CCC. This script submits a job using the cvar_pyutils library.
to install this library, call:
pip install --upgrade git+ssh://git@github.ibm.com/CVAR-DL-DET/pyutils.git

Usage examples:
---------------
# python transformers_src_multilingual/submit_run_summarization_new.py train
 --args transformers_src_multilingual/template_training_args.json
  --experiment-name summ_tryout --log-dir /logs
"""

import os
import argparse
import subprocess
from pathlib import Path

# some parameters related to job submission
MACHINE_TYPE = 'x86'
NUM_CORES = 16
NUM_GPUS = 1
MEMORY = '64g'  # amount of memory we will request for the job
GPU_TYPE = 'v100'  # currently we can't run on a100, see README.md

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--log-dir', type=str, default=None,
                        help='the dir to write the ccc job log into. If none is provided, the default is the curr dir')
    parser.add_argument('--duration', type=str, default='24h',
                        help='the desired job duration. possible values: 1h, 6h, 12h, 24h')
    parser.add_argument('-n', '--experiment-name', type=str, help='The name tag for the ccc job.',
                        default="contrast_classifier")
    parser.add_argument('--dummy', action='store_true',
                        help='disable job submission, just print the command to run')
    parser.add_argument('--local', action='store_true',
                        help='disable job submission, run the command locally')

    args, extra_args = parser.parse_known_args()

    if args.log_dir is None:
        logdir = os.getcwd()
    else:
        logdir = args.log_dir
    log_dir = os.path.join(logdir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    experiment_name = f'{args.experiment_name}'
    i = 0
    while os.path.isfile(os.path.join(log_dir, experiment_name + '_out.txt')):
        experiment_name = f'{args.experiment_name}_{i}'
        i += 1
    print(f'Logging to {log_dir}')
    out_fname = os.path.join(log_dir, experiment_name + '_out.txt')
    err_fname = os.path.join(log_dir, experiment_name + '_err.txt')

    data_dir = os.path.join(logdir, 'data', experiment_name)
    os.makedirs(data_dir, exist_ok=True)

    # prepare the parameters for run_summarization.py
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fname_to_run = 'example_script.py'
    full_fname_to_run = os.path.join(cur_dir, fname_to_run)
    params_str = " ".join(extra_args)
    command_to_run = f'python {full_fname_to_run} train -o {data_dir} {params_str}'

    print(f'experiment_name: {args.experiment_name}')
    print(f'duration: {args.duration}')
    print(f'out_fname: {out_fname}')
    print(f'err_fname: {err_fname}')
    print(f'command_to_run: {command_to_run}')

    if args.dummy:
        print('NOTICE: job submission is disabled - job was not submitted')
    elif args.local:
        subprocess.call(command_to_run.split())
    else:
        # submit the job
        from cvar_pyutils.ccc import submit_job

        job_id, jbsub_output = submit_job(command_to_run=command_to_run,
                                          machine_type=MACHINE_TYPE,
                                          duration=args.duration,
                                          num_cores=NUM_CORES,
                                          num_gpus=NUM_GPUS,
                                          name=args.experiment_name,
                                          mem=MEMORY,
                                          gpu_type=GPU_TYPE,
                                          out_file=out_fname,
                                          err_file=err_fname,
                                          )
        print(f'job_id: {job_id}')
        print(f'jbsub_output: {jbsub_output}')

        # write to log file
        log_fname = os.path.join(log_dir, experiment_name + '_log.txt')
        Path(log_fname).parent.mkdir(parents=True, exist_ok=True)
        with open(log_fname, 'w') as log_file:
            log_file.write(f'args passed to {__file__}:\n')
            for k, v in args.__dict__.items():
                log_file.write(f'{k}: {v}\n')

            log_file.write(f'\nsubmitted job id: {job_id}\n\n')
            log_file.write(f'jbsub output: {jbsub_output}\n')
