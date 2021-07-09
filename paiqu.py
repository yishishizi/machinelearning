from IPython.display import Image
Image("./Images/pdf_0.png",width="600px",height="400px")
Image("./Images/pdf_1.png",width="600px",height="400px")
Image("./Images/pdf_2.png",width="600px",height="400px")

import requests
import json
import re
import json

session=requests.session()

url=input("请输入下载的文件URL地址：")

content=session.get(url).content.decode('gbk')
doc_id=re.findall('view/(.*?).html',url)[0]
types=re.findall(r"docType.*?:.*?'(.*?)'",content)[0]
title=re.findall(r"title.*?:.*?'(.*?)'",content)[0]

#请输入下载的文件URL地址： https://wenku.baidu.com/view/155d684b773231126edb6f1aff00bed5b9f37387.html
#title
#'【中考】2019年徐州市中考英语作文万能写作模板【高分必备】'

url_list=re.findall(r'(https.*?0.json.*?)\\x22}',content)
url_list=[addr.replace("\\\\\\/","/") for addr in url_list]

result = ""

for url in set(url_list):
    content = session.get(url).content.decode('gbk')

    y = 0
    txtlists = re.findall(r'"c":"(.*?)".*?"y":(.*?),', content)
    for item in txtlists:
        # 当item[1]的值与前面不同时，代表要换行了
        if not y == item[1]:
            y = item[1]
            n = '\n'
        else:
            n = ''
        result += n
        result += item[0].encode('utf-8').decode('unicode_escape', 'ignore')

#D:\Anaconda\envs\mypython\lib\site-packages\ipykernel_launcher.py:16: DeprecationWarning: invalid escape sequence '\/'
  #app.launch_new_instance()

filename = "./Files/" + title + '.txt'

with open(filename, 'w', encoding="utf-8") as f:
    f.write(result)

f.close()

import requests
import json
import re
import os

session = requests.session()

path = "E:\\桌面文件"

if not os.path.exists(path):
    os.mkdir(path)


def parse_txt1(code, doc_id):
    content_url = 'https://wenku.baidu.com/api/doc/getdocinfo?callback=cb&doc_id=' + doc_id

    content = session.get(content_url).content.decode(code)
    md5sum = re.findall('"md5sum":"(.*?)",', content)[0]
    rsign = re.findall('"rsign":"(.*?)"', content)[0]
    pn = re.findall('"totalPageNum":"(.*?)"', content)[0]

    content_url = 'https://wkretype.bdimg.com/retype/text/' + doc_id + '?rn=' + pn + '&type=txt' + md5sum + '&rsign=' + rsign
    content = json.loads(session.get(content_url).content.decode('gbk'))

    result = ''

    for item in content:
        for i in item['parags']:
            result += i['c']

    return result


def parse_txt2(content, code, doc_id):
    md5sum = re.findall('"md5sum":"(.*?)",', content)[0]
    rsign = re.findall('"rsign":"(.*?)"', content)[0]
    pn = re.findall('"show_page":"(.*?)"', content)[0]

    content_url = 'https://wkretype.bdimg.com/retype/text/' + doc_id + '?rn=' + pn + '&type=txt' + md5sum + '&rsign=' + rsign
    content = json.loads(session.get(content_url).content.decode('utf-8'))

    result = ''

    for item in content:
        for i in item['parags']:
            result += i['c']

    return result


def parse_doc(content):
    url_list = re.findall(r'(https.*?0.json.*?)\\x22}', content)
    url_list = [addr.replace("\\\\\\/", "/") for addr in url_list]

    result = ""

    for url in set(url_list):
        content = session.get(url).content.decode('gbk')

        y = 0
        txtlists = re.findall(r'"c":"(.*?)".*?"y":(.*?),', content)
        for item in txtlists:
            # 当item[1]的值与前面不同时，代表要换行了
            if not y == item[1]:
                y = item[1]
                n = '\n'
            else:
                n = ''
            result += n
            result += item[0].encode('utf-8').decode('unicode_escape', 'ignore')

    return result


def parse_pdf(content):
    url_list = re.findall(r'(https.*?0.json.*?)\\x22}', content)
    url_list = [addr.replace("\\\\\\/", "/") for addr in url_list]

    result = ""

    for url in set(url_list):
        content = session.get(url).content.decode('gbk')

        y = 0
        txtlists = re.findall(r'"c":"(.*?)".*?"y":(.*?),', content)
        for item in txtlists:
            # 当item[1]的值与前面不同时，代表要换行了
            if not y == item[1]:
                y = item[1]
                n = '\n'
            else:
                n = ''
            result += n
            result += item[0].encode('utf-8').decode('unicode_escape', 'ignore')

    return result


def save_file(title, filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        print("文件" + title + "保存成功")
    f.close()


def main():
    print("欢迎来到百度文库文件下载：")
    print("-----------------------\r\n")

    while True:
        try:
            print("1.doc \n 2.txt \n 3.ppt \n 4.xls\n 5.ppt\n")
            types = input("请输入需要下载文件的格式(0退出)：")

            if types == "0":
                break

            if types not in ['txt', 'doc', 'pdf']:
                print("抱歉功能尚未开发")
                continue

            url = input("请输入下载的文库URL地址：")

            # 网页内容
            response = session.get(url)

            code = re.findall('charset=(.*?)"', response.text)[0]

            if code.lower() != 'utf-8':
                code = 'gbk'

            content = response.content.decode(code)

            # 文件id
            doc_id = re.findall('view/(.*?).html', url)[0]
            # 文件类型
            # types=re.findall(r"docType.*?:.*?'(.*?)'",content)[0]
            # 文件主题
            # title=re.findall(r"title.*?:.*?'(.*?)'",content)[0]

            if types == 'txt':
                md5sum = re.findall('"md5sum":"(.*?)",', content)
                if md5sum != []:
                    result = parse_txt2(content, code, doc_id)
                    title = re.findall(r'<title>(.*?). ', content)[0]
                    # filename=os.getcwd()+"\\Files\\"+title+'.txt'
                    filename = path + "\\" + title + ".txt"
                    save_file(title, filename, result)
                else:
                    result = parse_txt1(code, doc_id)
                    title = re.findall(r"title.*?:.*?'(.*?)'", content)[0]
                    # filename=os.getcwd()+"\\Files\\"+title+'.txt'
                    filename = path + "\\" + title + ".txt"
                    save_file(title, filename, result)
            elif types == 'doc':
                title = re.findall(r"title.*?:.*?'(.*?)'", content)[0]
                result = parse_doc(content)
                filename = path + "\\" + title + ".doc"
                save_file(title, filename, result)
            elif types == 'pdf':
                title = re.findall(r"title.*?:.*?'(.*?)'", content)[0]
                result = parse_pdf(content)
                filename = path + "\\" + title + ".txt"
                save_file(title, filename, result)


        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()

