#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import sys
from subprocess import check_call, check_output


def read_my_jobs():
    return check_output(["qstat", "-u", "fr_mn119"]).decode()


def main():
    myjobs = [
        job.strip() for job in read_my_jobs().split("\n")
        if job.strip()
    ][4:]

    for job in myjobs:
        id_ = job.split(" ")[0]
        assert id_.isnumeric()
        if len(sys.argv) >= 2 and sys.argv[1] == "--dryrun":
            print("qdel {}".format(id_))
        else:
            check_call(["qdel", id_])

    print(read_my_jobs())


if __name__ == "__main__":
    main()
