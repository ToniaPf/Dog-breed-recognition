import os
import re
import requests

"""
MIT License
Copyright (c) 2020 zqthu
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class THUCloud:
    """
    Based on https://github.com/zqthu/thu_cloud_download/blob/master/thu_cloud_download.py
    Changed to download in chunks
    """
    def __init__(self, shared_link, outdir=None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}

        if "/f/" in shared_link:  # single file
            archive = shared_link.split("/f/")[-1].split("/")[0]
            self.api_link = "https://cloud.tsinghua.edu.cn/f/{}/".format(archive)
            self.file_link = "https://cloud.tsinghua.edu.cn/f/{}/?dl=1".format(archive)
        else:
            raise ValueError("Cannot parse the shared link.")

        if outdir is None:
            self.current_dir = os.getcwd()
        else:
            self.current_dir = os.path.abspath(outdir)
        if not os.path.exists(self.current_dir):
            os.mkdir(self.current_dir)

    def _retrieve_file(self, url, name):  # for small files
        file_path = os.path.join(self.current_dir, name)
        with requests.get(url, stream=True, headers=self.headers) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    print('.', end='')
                    f.write(chunk)

    def download(self):
        response = requests.get(url=self.api_link, headers=self.headers)
        assert response.status_code == 200
        content = response.content.decode()
        name = re.search(r"fileName: '(.*)',", content).group(1)
        self._retrieve_file(self.file_link, name)


if __name__ == "__main__":
    url_low = 'https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/'
    url_high_1 = 'https://cloud.tsinghua.edu.cn/f/d2031efb239c4dde9c6c/'
    url_high_2 = 'https://cloud.tsinghua.edu.cn/f/6a242a6bba664537ba45/'
    url_high_3 = 'https://cloud.tsinghua.edu.cn/f/d17034fa14f54e4381d8/'
    url_high_4 = 'https://cloud.tsinghua.edu.cn/f/3740fc44cd484e1cb089/'
    url_high_5 = 'https://cloud.tsinghua.edu.cn/f/ff5d96a0bc4e4dba9004/'
    url_high_6 = 'https://cloud.tsinghua.edu.cn/f/d5fe5c88198c4387a7bb/'
    url_high_7 = 'https://cloud.tsinghua.edu.cn/f/b13d6710ac85487e9487/'
    url_high_8 = 'https://cloud.tsinghua.edu.cn/f/b6cf354fd04b4fe0b909/'
    url_high_9 = 'https://cloud.tsinghua.edu.cn/f/06a421a528044b15838c/'

    high_urls = [
        url_high_1, url_high_2, url_high_3, url_high_4, url_high_5,
        url_high_6, url_high_7, url_high_8, url_high_9
    ]

    for d_url in high_urls:
        print(d_url)
        THUCloud(d_url, 'high').download()

