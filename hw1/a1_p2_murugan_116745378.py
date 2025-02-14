#!/usr/bin/env python3

import sys
import argparse

from typing import List


def getConllTags(filename: str) -> List[List]:
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]

    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding="utf8") as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="script to run cse538 assignment 1 part 2"
    )
    parser.add_argument("filepath", type=str, help="path to the input file")
    args = parser.parse_args()

    # Read and process input data
    with open(args.filepath, "r") as f:
        data = f.read().splitlines()

    # Create and open output file
    outfile = open("a1_p2_murugan_116745378_OUTPUT.txt", "w")

    outfile.write("Checkpoint 2.1:\n")
    outfile.write("Checkpoint 2.2:\n")
    outfile.write("Checkpoint 2.3:\n")
    outfile.write("Checkpoint 2.4:\n")

    # Close output file
    outfile.close()

    return None


if __name__ == "__main__":
    main()
else:
    sys.exit(0)
