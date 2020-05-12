import argparse
import json

# Argument parser
parser = argparse.ArgumentParser(description="project_ingestion")
parser.add_argument("source_path", help="The path of the json data source file")
parser.add_argument("output_path", help="The path of the output json file")

# Argument variables
sourcePath = parser.parse_args().source_path
outputPath = parser.parse_args().output_path

with open(sourcePath, 'r') as sourceFile:
    with open(outputPath, "w") as outputFile:

        data = {}

        for row in sourceFile:
            raw = json.loads(row)

            data["body"] = raw["body"]
            data["subreddit"] = raw["subreddit"]

            json.dump(data, outputFile)
            outputFile.write("\n")
