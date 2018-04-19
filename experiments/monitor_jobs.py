#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
# vim:foldmethod=marker
import argparse
from itertools import product
from os.path import join as path_join
import smtplib
from subprocess import check_call, check_output
from email.mime.text import MIMEText
import time


def notify_mail(recipient, batch_index, finished=False):
    text = (
        "Hello there,\n"
        "I just noticed that your job batch with id {id} "
        "is no longer running.\n"
        "{continuation}\n\n"
        "Best,\n"
        "Your Job Monitor Assistant (Making clusters sane since 1992)".format(
            id=batch_index, continuation="{}".format("This was the last batch for now, my work is done here!" if finished else "I've got some batches left though, so this is only an intermediate update.")
        )
    )
    msg = MIMEText(text)
    me = "uni_freiburglecturescraper@web.de"
    msg["Subject"] = "Binac batch with id {id} finished.".format(id=batch_index)
    msg["From"] = me
    msg["To"] = recipient

    username = me
    password = 'RobotMappingAndOtherStuff'
    server = smtplib.SMTP('smtp.web.de', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password)
    server.sendmail(me, recipient, msg.as_string())
    server.quit()


def submit_jobs(jobs):
    for job in jobs:
        check_call(job)


def jobs_running():
    return bool(check_output(("qstat", "-u", "fr_mn119")))


def main():
    parser = argparse.ArgumentParser(description="Submit a batch of jobs (200 jobs per batch by default), checks every minute if a batch terminates and resubmits a new one. Optionally informs about batch termination via email.")
    parser.add_argument("--notify", help="E-mail address to notify when a batch has terminated.")
    parser.add_argument("--batch-size", help="Number of jobs to submit in a batch.", default=200)

    parser.add_argument(
        "--datasets",
        help="(Comma-seperated) string of datasets to use. Defaults to `cifar10`.",
        default="cifar10"
    )

    parser.add_argument(
        "--benchmarks",
        help="(Comma-seperated) string of models to use. Defaults to `AlexNet`.",
        default="AlexNet"
    )

    parser.add_argument(
        "--no-runs", help="Number of runs to perform.", type=int, dest="no_runs",
        default=1
    )

    parser.add_argument(
        "--time-budget", default=7200, help="Maximum time budget to train a network",
        type=int
    )
    parser.add_argument(
        "--pipeline",
        dest="pipeline",
        default="best_config",
        help="Data augmentation pipeline to use, choice:{default, no_augment, best_config}"
    )

    parser.add_argument(
        "--configuration-file", help="Configuration file (json) to use.",
    )

    parser.add_argument(
        "submit_script", help="Submit script for this job."
    )

    parser.add_argument(
        "output_directory", help="Directory for output files."
    )
    subparsers = parser.add_subparsers()
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument(
        "optimizers",
        help="(Comma-seperated) string of optimizers to use. Defaults to `hyperband`.",
    )

    args = parser.parse_args()

    #  Construct job list {{{ #

    datasets = args.datasets.split(",") * args.no_runs
    benchmarks = args.benchmarks.split(",")
    try:
        optimizers = args.optimizers.split(",")
    except AttributeError:
        if args.configuration_file is not None:  # runs default or best found config experiments
            jobs = [(
                "qsub", "-v",
                "dataset={},run_id={},benchmark={},time_budget={},configuration_file={}".format(dataset, run_id, benchmark, args.time_budget, args.configuration_file),
                "-q", "gpu", "-l", "nodes=1:ppn=1:gpus=1",

                "-o", path_join(args.output_directory, "of_run_{pipeline}_{benchmark}_{dataset}_{run_id}.txt".format(pipeline=args.pipeline, benchmark=benchmark, dataset=dataset, run_id=run_id)),
                "-e", path_join(args.output_directory, "ef_run_{pipeline}_{benchmark}_{dataset}_{run_id}.txt".format(pipeline=args.pipeline, benchmark=benchmark, dataset=dataset, run_id=run_id)),
                args.submit_script

            ) for run_id, (dataset, benchmark) in enumerate(product(datasets, benchmarks))
            ]
        else:
            jobs = [(
                "qsub", "-v",
                "dataset={},run_id={},benchmark={},time_budget={}".format(dataset, run_id, benchmark, args.time_budget),
                "-q", "gpu", "-l", "nodes=1:ppn=1:gpus=1",
                "-o", path_join(args.output_directory, "of_run_{pipeline}_{benchmark}_{dataset}_{run_id}.txt".format(pipeline=args.pipeline, benchmark=benchmark, dataset=dataset, run_id=run_id)),

                "-e", path_join(args.output_directory, "ef_run_{pipeline}_{benchmark}_{dataset}_{run_id}.txt".format(pipeline=args.pipeline, benchmark=benchmark, dataset=dataset, run_id=run_id)),
                args.submit_script
            ) for run_id, (dataset, benchmark) in enumerate(product(datasets, benchmarks))
            ]

    else:
        jobs = [(
            "qsub", "-v",
            "dataset={},opt={},benchmark={},run_id={},time_budget={}".format(dataset, optimizer, benchmark, run_id, args.time_budget),
            "-q", "gpu", "-l", "nodes=1:ppn=1:gpus=1",
            "-o", path_join(args.output_directory, "of_run_{dataset}_{optimizer}_{benchmark}_{run_id}.txt".format(dataset=dataset, optimizer=optimizer, benchmark=benchmark, run_id=run_id)),
            "-e", path_join(args.output_directory, "ef_run_{dataset}_{optimizer}_{benchmark}_{run_id}.txt".format(dataset=dataset, optimizer=optimizer, benchmark=benchmark, run_id=run_id)),
            args.submit_script
        ) for run_id, (dataset, optimizer, benchmark) in enumerate(product(datasets, optimizers, benchmarks))
        ]
    #  }}} Construct job list #

    batches = [jobs[i:i + args.batch_size] for i in range(0, len(jobs), args.batch_size)]

    for (batch_index, job_batch) in enumerate(batches, 1):
        submit_jobs(job_batch)
        while False:
            time.sleep(1)

        if args.notify:
            notify_mail(
                recipient=args.notify,
                batch_index=batch_index,
                finished=batch_index == len(batches)
            )


if __name__ == "__main__":
    main()
