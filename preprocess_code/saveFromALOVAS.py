import requests
import re
import os
import argparse

def download_file(url, save_path):
    """下載單一檔案並儲存到指定路徑。"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"檔案已儲存到 {save_path}")
    except Exception as e:
        print(f"下載失敗: {url}\n錯誤訊息: {e}")

def parse_record(line):
    """解析一行紀錄，回傳tif_url, tif_filename, json_url, json_filename。"""
    tif_url, json_url = line.split(',')
    tif_match = re.search(r'/([^/]+\.tif)', tif_url)
    tif_filename = tif_match.group(1) if tif_match else None

    json_match = re.search(r'/([^/]+)/annotation\.json\?', json_url)
    json_filename = f"{json_match.group(1)}.json" if json_match else None

    return tif_url, tif_filename, json_url, json_filename

def read_links(path=None, default_records=None):
    """從檔案或字串讀取所有紀錄，回傳字串。"""
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return default_records or ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--links_record_path', default='./dataset/alovas_datalist.txt')
    parser.add_argument('--default_records', default='''http://cos.twcc.ai/alovas-platform-image-bucket-4/image/560efaf8-caf7-4e02-bfed-9e6dc4f35dc2.tif?AWSAccessKeyId=7GAL0RQVP8WEBC0Z4IQ8&Signature=bgGb3YfqX3FEKVZ4wxXEBTn0z%2FQ%3D&Expires=1749525389,http://cos.twcc.ai/alovas-platform-image-bucket-4/image/data/560efaf8-caf7-4e02-bfed-9e6dc4f35dc2/annotation.json?AWSAccessKeyId=7GAL0RQVP8WEBC0Z4IQ8&Signature=xADXX8fUPj%2BvM8iqj72wXfn1SJY%3D&Expires=1749525389
http://cos.twcc.ai/alovas-platform-image-bucket-4/image/fcb948df-d853-44ff-ba65-724fee6b1973.tif?AWSAccessKeyId=7GAL0RQVP8WEBC0Z4IQ8&Signature=DjemthESJt59EKqxEgVbY%2F1fxtE%3D&Expires=1749525389,http://cos.twcc.ai/alovas-platform-image-bucket-4/image/data/fcb948df-d853-44ff-ba65-724fee6b1973/annotation.json?AWSAccessKeyId=7GAL0RQVP8WEBC0Z4IQ8&Signature=mqfjBov16rDgvcGndj5bzgF9tm0%3D&Expires=1749525389
''')
    parser.add_argument('--save_path', default="./dataset/")
    args = parser.parse_args()

    links_record_path = args.links_record_path
    default_records = args.default_records
    data_records = read_links(links_record_path, default_records)

    img_folder = os.path.join(args.save_path, 'images')
    label_folder = os.path.join(args.save_path, 'annotations')
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for line in data_records.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        try:
            tif_url, tif_filename, json_url, json_filename = parse_record(line)
            if not tif_filename or not json_filename:
                print(f"檔名解析失敗: {line}")
                continue

            tif_save_path = os.path.join(img_folder, tif_filename)
            json_save_path = os.path.join(label_folder, json_filename)

            download_file(tif_url, tif_save_path)
            download_file(json_url, json_save_path)
        except Exception as e:
            print(f"處理紀錄時發生錯誤: {line}\n錯誤訊息: {e}")

if __name__ == '__main__':
    main()
