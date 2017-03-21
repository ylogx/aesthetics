import click

def read_ava_data(data_folder):
    import pandas as pd
    import os

    columns = 'Index ImageId r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 SemanticTag1 SemanticTag2 ChallengeId'.split()
    df = pd.read_csv(os.path.join(data_folder, 'AVA.txt'), delimiter=' ', names=columns)
    return df


def to_url(img_id):
    return 'http://www.dpchallenge.com/image.php?IMAGE_ID={img_id}'.format(img_id=img_id)


@click.command()
@click.option('--data-folder', '-d', default='./data', help='AVA Data Folder (default: ./data)')
def stats(data_folder):
    df = read_ava_data(data_folder)
    print(df.head())
    print(df.info())
    print(df.describe())


@click.command()
@click.option('--data-folder', '-d', default='./data', help='AVA Data Folder (default: ./data)')
def download_ava(data_folder):
    df = read_ava_data(data_folder)
    urls = map(to_url, df['ImageId'].values)
    for i, url in enumerate(urls):
        print(url)
        if i == 10:
            break



@click.group()
def main():
    return 0

main.add_command(download_ava, 'ava')
main.add_command(stats)

