import argparse
from pathlib import Path


def count_words_in_file(file_path):
    with open(file_path) as fp:
        contents = fp.read()
        words = contents.split()
        return len(words)


def count_words_in_directory(directory_path):
    directory = Path(directory_path)
    total_word_count = 0

    if not directory.is_dir():
        return

    for file_path in directory.glob("*.txt"):
        word_count = count_words_in_file(file_path)
        total_word_count += word_count
        print(f"The file {file_path.name} contains {word_count} words")

    print(f"The total number of words in all the files is {total_word_count}")


def run():
    parser = argparse.ArgumentParser(
        description="Count words in text files in a directory"
    )
    parser.add_argument(
        "--input_directory", "-d", required=True, help="Directory containing text files"
    )

    args = parser.parse_args()

    directory_path = args.input_directory
    count_words_in_directory(directory_path)


if __name__ == "__main__":
    run()
